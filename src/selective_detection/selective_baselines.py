"""Phase 3 selective-classification baseline utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DETECTORS = ("univfd", "safe")
SPLITS = ("split_a", "split_b")
EVAL_ROLES = ("protocol_seen", "protocol_held_out")
MANDATORY_BASELINES = ("msp", "entropy", "energy", "temp_msp", "mahalanobis", "knn")


@dataclass(frozen=True)
class RuntimeRecord:
    detector: str
    baseline: str
    split: str
    stage: str
    seconds: float
    sample_count: int
    peak_ram_mb: float
    peak_gpu_memory_mb: float
    throughput_samples_per_second: float
    reference_bank_size: int
    disk_size_bytes: int


def now_local_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Bangkok").isoformat()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def file_size_sum(paths: list[str | Path]) -> int:
    return int(sum(Path(path).stat().st_size for path in paths if Path(path).exists()))


def gpu_memory_mb(device_index: int = 1) -> float:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return float(output.splitlines()[0])
    except Exception:
        return 0.0


def process_ram_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        return 0.0


def sigmoid_prob(logits: np.ndarray) -> np.ndarray:
    return expit(np.asarray(logits, dtype=np.float64))


def msp_risk(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    return 1.0 - np.maximum(p, 1.0 - p)


def entropy_risk(probabilities: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0 - epsilon)
    return -((p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))


def energy_risk(logits: np.ndarray, energy_temperature: float = 1.0) -> np.ndarray:
    """Binary symmetric energy risk: -T logsumexp([-s/2, s/2] / T)."""
    s = np.asarray(logits, dtype=np.float64)
    two_logits = np.stack([-s / 2.0, s / 2.0], axis=1)
    return -energy_temperature * logsumexp(two_logits / energy_temperature, axis=1)


def temp_msp_risk(logits: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    p = sigmoid_prob(np.asarray(logits, dtype=np.float64) / float(temperature))
    return msp_risk(p), p


def binary_nll(labels: np.ndarray, logits: np.ndarray, temperature: float) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    p = np.clip(sigmoid_prob(np.asarray(logits, dtype=np.float64) / float(temperature)), 1e-12, 1.0 - 1e-12)
    return float(-(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    min_temperature: float,
    max_temperature: float,
) -> dict[str, float | int | bool]:
    result = minimize_scalar(
        lambda t: binary_nll(labels, logits, float(t)),
        bounds=(float(min_temperature), float(max_temperature)),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    return {
        "temperature": float(result.x),
        "nll": float(result.fun),
        "success": bool(result.success),
        "iterations": int(result.nit),
    }


def cross_validate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
    min_temperature: float,
    max_temperature: float,
) -> pd.DataFrame:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(logits, labels), start=1):
        fitted = fit_temperature(logits[train_idx], labels[train_idx], min_temperature, max_temperature)
        valid_nll = binary_nll(labels[valid_idx], logits[valid_idx], float(fitted["temperature"]))
        rows.append(
            {
                "fold": fold,
                "temperature": fitted["temperature"],
                "train_nll": fitted["nll"],
                "valid_nll": valid_nll,
                "success": fitted["success"],
            }
        )
    return pd.DataFrame(rows)


def l2_normalize(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise ValueError("zero-norm or non-finite embedding encountered")
    return arr / norms[:, None], norms


def fit_mahalanobis(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or Inf")
    mean = x.mean(axis=0, dtype=np.float64)
    std = x.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    z = ((x.astype(np.float64) - mean) / std).astype(np.float64)
    stats: dict[str, Any] = {"standardization_mean": mean, "standardization_std": std, "classes": {}}
    for cls in (0, 1):
        cls_z = z[labels == cls]
        if len(cls_z) < 2:
            raise ValueError(f"class {cls} has too few samples for Ledoit-Wolf covariance")
        try:
            estimator = LedoitWolf().fit(cls_z.astype(np.float32))
        except Exception:
            estimator = LedoitWolf().fit(cls_z.astype(np.float64))
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        precision = np.asarray(estimator.precision_, dtype=np.float64)
        center = cls_z.mean(axis=0, dtype=np.float64)
        if not (np.isfinite(covariance).all() and np.isfinite(precision).all()):
            raise ValueError(f"class {cls} covariance or precision contains NaN/Inf")
        stats["classes"][str(cls)] = {
            "mean": center,
            "covariance": covariance,
            "precision": precision,
            "sample_count": int(len(cls_z)),
            "shrinkage": float(estimator.shrinkage_),
            "condition_number": float(np.linalg.cond(covariance)),
        }
    return stats


def score_mahalanobis(embeddings: np.ndarray, stats: dict[str, Any]) -> dict[str, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float64)
    z = (x - stats["standardization_mean"]) / stats["standardization_std"]
    distances: dict[str, np.ndarray] = {}
    for cls in ("0", "1"):
        center = stats["classes"][cls]["mean"]
        precision = stats["classes"][cls]["precision"]
        diff = z - center
        squared = np.einsum("ij,jk,ik->i", diff, precision, diff, optimize=True)
        distances[cls] = np.sqrt(np.maximum(squared, 0.0))
    minimum = np.minimum(distances["0"], distances["1"])
    return {
        "risk_score": minimum.astype(np.float64),
        "distance_to_real": distances["0"].astype(np.float64),
        "distance_to_fake": distances["1"].astype(np.float64),
    }


def exact_knn_neighbors(
    bank: np.ndarray,
    query: np.ndarray,
    k: int,
    bank_ids: np.ndarray | None = None,
    query_ids: np.ndarray | None = None,
    device: str | None = None,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Exact cosine kNN neighbor distances/indexes with optional torch/CUDA acceleration."""
    bank_norm, _ = l2_normalize(bank)
    query_norm, _ = l2_normalize(query)
    if device is None:
        device = "cuda:1"
    try:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if device.startswith("cuda"):
            requested_index = int(device.split(":", 1)[1]) if ":" in device else 0
            if requested_index >= torch.cuda.device_count():
                device = "cuda:0"
        torch_device = torch.device(device)
        bank_t = torch.as_tensor(bank_norm, dtype=torch.float32, device=torch_device).T.contiguous()
        distances = np.empty((query_norm.shape[0], int(k)), dtype=np.float32)
        neighbor_indexes = np.empty((query_norm.shape[0], int(k)), dtype=np.int64)
        extra = 1 if bank_ids is not None and query_ids is not None else 0
        search_k = min(int(k) + extra, bank_norm.shape[0])
        bank_ids_arr = np.asarray(bank_ids).astype(str) if bank_ids is not None else None
        query_ids_arr = np.asarray(query_ids).astype(str) if query_ids is not None else None
        with torch.no_grad():
            for start in range(0, query_norm.shape[0], batch_size):
                end = min(start + batch_size, query_norm.shape[0])
                q = torch.as_tensor(query_norm[start:end], dtype=torch.float32, device=torch_device)
                sims = q @ bank_t
                values, indexes = torch.topk(sims, k=search_k, dim=1, largest=True, sorted=True)
                values_np = values.cpu().numpy()
                indexes_np = indexes.cpu().numpy()
                for row in range(end - start):
                    idx = indexes_np[row]
                    sim = values_np[row]
                    if bank_ids_arr is not None and query_ids_arr is not None:
                        keep = bank_ids_arr[idx] != query_ids_arr[start + row]
                        idx = idx[keep]
                        sim = sim[keep]
                    idx = idx[:k]
                    sim = sim[:k]
                    if len(idx) < k:
                        raise RuntimeError("not enough neighbors after self-neighbor removal")
                    dist = 1.0 - sim
                    distances[start + row] = dist
                    neighbor_indexes[start + row] = idx
                del q, sims, values, indexes
        return {"distances": distances, "indexes": neighbor_indexes}
    except Exception:
        import faiss

        index = faiss.IndexFlatIP(bank_norm.shape[1])
        index.add(bank_norm.astype(np.float32))
        extra = 1 if bank_ids is not None and query_ids is not None else 0
        search_k = min(int(k) + extra, bank_norm.shape[0])
        sims, indexes = index.search(query_norm.astype(np.float32), search_k)
        bank_ids_arr = np.asarray(bank_ids).astype(str) if bank_ids is not None else None
        query_ids_arr = np.asarray(query_ids).astype(str) if query_ids is not None else None
        distances = np.empty((query_norm.shape[0], int(k)), dtype=np.float32)
        neighbor_indexes = np.empty((query_norm.shape[0], int(k)), dtype=np.int64)
        for row in range(query_norm.shape[0]):
            idx = indexes[row]
            sim = sims[row]
            if bank_ids_arr is not None and query_ids_arr is not None:
                keep = bank_ids_arr[idx] != query_ids_arr[row]
                idx = idx[keep]
                sim = sim[keep]
            idx = idx[:k]
            sim = sim[:k]
            if len(idx) < k:
                raise RuntimeError("not enough neighbors after self-neighbor removal")
            dist = 1.0 - sim
            distances[row] = dist
            neighbor_indexes[row] = idx
        return {"distances": distances, "indexes": neighbor_indexes}


def exact_knn_distance(
    bank: np.ndarray,
    query: np.ndarray,
    k: int,
    bank_ids: np.ndarray | None = None,
    query_ids: np.ndarray | None = None,
    device: str | None = None,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    """Exact cosine kNN average distance with optional torch/CUDA acceleration."""
    neighbors = exact_knn_neighbors(bank, query, k, bank_ids, query_ids, device, batch_size)
    distances = neighbors["distances"]
    indexes = neighbors["indexes"]
    return {
        "risk_score": distances.mean(axis=1).astype(np.float32),
        "neighbor_distance_1": distances[:, 0].astype(np.float32),
        "neighbor_index_1": indexes[:, 0].astype(np.int64),
    }


def select_knn_k_cv(
    embeddings: np.ndarray,
    errors: np.ndarray,
    sample_ids: np.ndarray,
    candidate_k: list[int],
    folds: int,
    seed: int,
    device: str = "cuda:1",
) -> tuple[int, pd.DataFrame]:
    labels_for_split = np.asarray(errors, dtype=np.int64)
    if len(np.unique(labels_for_split)) < 2:
        labels_for_split = np.zeros_like(labels_for_split)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    max_k = int(max(candidate_k))
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(embeddings, labels_for_split), start=1):
        neighbors = exact_knn_neighbors(
            embeddings[train_idx],
            embeddings[valid_idx],
            max_k,
            bank_ids=sample_ids[train_idx],
            query_ids=None,
            device=device,
        )
        cumulative = np.cumsum(neighbors["distances"], axis=1)
        for k in sorted(set(candidate_k)):
            risk = cumulative[:, int(k) - 1] / float(k)
            status = "ok"
            auroc = float("nan")
            if len(np.unique(errors[valid_idx])) < 2:
                status = "undefined_due_to_single_error_class"
            else:
                auroc = float(roc_auc_score(errors[valid_idx], risk))
            rows.append({"fold": fold, "k": int(k), "error_detection_AUROC": auroc, "status": status})
    cv = pd.DataFrame(rows)
    summary = (
        cv[cv["status"] == "ok"]
        .groupby("k", as_index=False)["error_detection_AUROC"]
        .mean()
        .sort_values(["error_detection_AUROC", "k"], ascending=[False, True], kind="mergesort")
    )
    selected = int(summary["k"].iloc[0]) if len(summary) else int(min(candidate_k))
    cv["selected_k"] = selected
    return selected, cv


class Phase2Cache:
    """Loader for frozen Phase 2 predictions and embedding shards."""

    def __init__(self, project_root: str | Path, detector: str):
        self.project_root = Path(project_root)
        self.detector = detector
        self.cache_dir = self.project_root / "artifacts" / "cache" / detector / "clean"
        self.index = pd.read_parquet(self.cache_dir / "index.parquet")
        self.index["sample_id"] = self.index["sample_id"].astype(str)
        self.predictions = self._load_predictions()

    def _load_predictions(self) -> pd.DataFrame:
        frames = []
        for shard in sorted(self.index["prediction_shard"].unique()):
            frame = pd.read_parquet(shard)
            frame["sample_id"] = frame["sample_id"].astype(str)
            frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        predictions["sample_id"] = predictions["sample_id"].astype(str)
        return predictions

    def prediction_rows(self, sample_ids: pd.Series | np.ndarray | list[str]) -> pd.DataFrame:
        requested = pd.DataFrame({"sample_id": pd.Series(sample_ids, dtype="string"), "_order": range(len(sample_ids))})
        merged = requested.merge(self.predictions, on="sample_id", how="left")
        if merged["raw_logit"].isna().any():
            missing = merged.loc[merged["raw_logit"].isna(), "sample_id"].head(5).tolist()
            raise RuntimeError(f"missing predictions for {self.detector}: {missing}")
        return merged.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)

    def embeddings_for(self, sample_ids: pd.Series | np.ndarray | list[str]) -> np.ndarray:
        requested = pd.DataFrame({"sample_id": pd.Series(sample_ids, dtype="string"), "_order": range(len(sample_ids))})
        coords = requested.merge(self.index, on="sample_id", how="left")
        if coords["embedding_shard"].isna().any():
            missing = coords.loc[coords["embedding_shard"].isna(), "sample_id"].head(5).tolist()
            raise RuntimeError(f"missing embedding coordinates for {self.detector}: {missing}")
        first = np.load(coords["embedding_shard"].iloc[0], mmap_mode="r")
        out = np.empty((len(coords), int(first.shape[1])), dtype=np.float32)
        for shard, group in coords.groupby("embedding_shard", sort=False):
            arr = np.load(shard, mmap_mode="r")
            out[group["_order"].to_numpy(dtype=np.int64)] = arr[group["row_offset"].to_numpy(dtype=np.int64)]
        return out


def base_thresholds(project_root: str | Path) -> pd.DataFrame:
    thresholds = pd.read_csv(Path(project_root) / "artifacts" / "phase2_clean_thresholds.csv")
    return thresholds[thresholds["threshold_source"] == "threshold_cal"].copy()


def add_base_predictions(df: pd.DataFrame, predictions: pd.DataFrame, thresholds: pd.DataFrame, detector: str) -> pd.DataFrame:
    out = df.merge(
        predictions[
            [
                "sample_id",
                "raw_logit",
                "fake_probability",
                "predicted_label",
                "checkpoint_sha256",
                "preprocessing_id",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if out["raw_logit"].isna().any():
        raise RuntimeError(f"{detector} prediction join incomplete")
    threshold_map = thresholds[thresholds["detector"] == detector].set_index("split")["decision_threshold"].to_dict()
    out["phase2_decision_threshold"] = out["split"].map(threshold_map)
    if out["phase2_decision_threshold"].isna().any():
        raise RuntimeError(f"missing phase2 threshold for {detector}")
    out["base_prediction"] = (out["fake_probability"].astype(float) >= out["phase2_decision_threshold"].astype(float)).astype(int)
    out["base_error"] = (out["base_prediction"].astype(int) != out["label"].astype(int)).astype(int)
    return out


def make_score_frame(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    detector: str,
    baseline: str,
    split: str,
    partition: str,
    evaluation_role: str,
    risk_score: np.ndarray,
    fit_artifact_sha256: str,
    config_sha256: str,
    diagnostics: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    base_cols = [
        "sample_id",
        "sha256",
        "canonical_generator",
        "label",
        "evaluation_split",
        "evaluation_role",
        "source_riskguard_partition",
    ]
    available = [col for col in base_cols if col in manifest.columns]
    df = manifest[available].copy()
    if "sha256" not in df.columns:
        df = df.merge(predictions[["sample_id", "sha256"]], on="sample_id", how="left", validate="one_to_one")
    df = df.rename(columns={"canonical_generator": "generator"})
    df["split"] = split
    df["partition"] = partition
    df["evaluation_role"] = evaluation_role
    df = add_base_predictions(df, predictions, pd.DataFrame(), detector) if False else df
    pred_cols = predictions[
        [
            "sample_id",
            "raw_logit",
            "fake_probability",
            "checkpoint_sha256",
            "preprocessing_id",
        ]
    ].copy()
    df = df.merge(pred_cols, on="sample_id", how="left", validate="one_to_one")
    if df["raw_logit"].isna().any():
        raise RuntimeError(f"{detector}/{baseline}/{partition} prediction join incomplete")
    df["detector"] = detector
    df["baseline"] = baseline
    df["base_logit"] = df["raw_logit"].astype(float)
    df["base_probability"] = df["fake_probability"].astype(float)
    df["risk_score"] = np.asarray(risk_score, dtype=np.float64)
    df["risk_orientation"] = "higher_risk_score_more_likely_to_reject"
    df["fit_artifact_sha256"] = fit_artifact_sha256
    df["config_sha256"] = config_sha256
    if diagnostics:
        for key, value in diagnostics.items():
            df[key] = value
    return df


def required_phase2_inputs(project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    relative = [
        "reports/verified_v2_audit_report.md",
        "artifacts/verified_v2_audit_summary.json",
        "artifacts/verified_v2_immutable_artifact_hashes.csv",
        "artifacts/verified_v2_evaluation_config.json",
        "artifacts/verified_v2_checkpoint_provenance.csv",
        "artifacts/verified_v2_fit_provenance_audit.csv",
        "artifacts/verified_v2_transformation_lineage_audit.csv",
        "artifacts/verified_v2_explicit_eval_table.csv",
        "artifacts/phase2_clean_thresholds.csv",
        "artifacts/detector_runtime_registry.json",
        "datasets/manifests/verified_v2_split_a_protocol_seen_eval.csv",
        "datasets/manifests/verified_v2_split_a_protocol_held_out_eval.csv",
        "datasets/manifests/verified_v2_split_b_protocol_seen_eval.csv",
        "datasets/manifests/verified_v2_split_b_protocol_held_out_eval.csv",
        "datasets/manifests/split_a_risk_fit.csv",
        "datasets/manifests/split_a_threshold_cal.csv",
        "datasets/manifests/split_b_risk_fit.csv",
        "datasets/manifests/split_b_threshold_cal.csv",
        "datasets/manifests/bfree_viral_verified_snapshot.csv",
    ]
    paths = [root / item for item in relative]
    for detector in DETECTORS:
        cache = root / "artifacts" / "cache" / detector / "clean"
        paths.append(cache / "index.parquet")
        paths.extend(sorted(cache.glob("predictions_*.parquet")))
        paths.extend(sorted(cache.glob("embeddings_*.npy")))
    registry = json.loads((root / "artifacts" / "detector_runtime_registry.json").read_text(encoding="utf-8"))
    for detector in DETECTORS:
        checkpoint = registry["detectors"][detector].get("checkpoint_path")
        if checkpoint:
            paths.append(root / checkpoint)
    return paths


def freeze_phase2_inputs(project_root: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(project_root)
    paths = required_phase2_inputs(root)
    rows = []
    for path in paths:
        rows.append(
            {
                "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else "",
            }
        )
    hashes = pd.DataFrame(rows)
    hashes_path = root / "artifacts" / "phase2_frozen_artifact_hashes.csv"
    hashes_path.parent.mkdir(parents=True, exist_ok=True)
    hashes.to_csv(hashes_path, index=False)

    registry = json.loads((root / "artifacts" / "detector_runtime_registry.json").read_text(encoding="utf-8"))
    thresholds = pd.read_csv(root / "artifacts" / "phase2_clean_thresholds.csv")
    detector_payload = {}
    for detector in DETECTORS:
        info = registry["detectors"][detector]
        detector_payload[detector] = {
            "prediction_index": f"artifacts/cache/{detector}/clean/index.parquet",
            "embedding_index": f"artifacts/cache/{detector}/clean/index.parquet",
            "checkpoint_sha256": info.get("checkpoint_sha256", ""),
            "repository_commit": info.get("repository_commit", ""),
            "preprocessing_id": info.get("preprocessing_id", ""),
            "score_orientation": info.get("official_score_orientation", ""),
            "base_decision_thresholds": thresholds[thresholds["detector"] == detector].to_dict("records"),
        }
    try:
        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        repo_commit = "unavailable_not_git_repository"
    config_hash = hashes.loc[
        hashes["relative_path"] == "artifacts/verified_v2_evaluation_config.json", "sha256"
    ].iloc[0]
    payload = {
        "created_at": now_local_iso(),
        "phase": "phase2_frozen_for_phase3",
        "repository_commit": repo_commit,
        "phase2_manifest_paths": [
            "datasets/manifests/verified_v2_split_a_protocol_seen_eval.csv",
            "datasets/manifests/verified_v2_split_a_protocol_held_out_eval.csv",
            "datasets/manifests/verified_v2_split_b_protocol_seen_eval.csv",
            "datasets/manifests/verified_v2_split_b_protocol_held_out_eval.csv",
        ],
        "detectors": detector_payload,
        "threshold_provenance": "artifacts/phase2_clean_thresholds.csv threshold_source=threshold_cal",
        "evaluation_configuration_hash": config_hash,
        "artifact_hash_registry": "artifacts/phase2_frozen_artifact_hashes.csv",
    }
    write_yaml(root / "configs" / "phase2_frozen.yaml", payload)
    return hashes, payload


def verify_phase2_frozen_hashes(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    registry_path = root / "artifacts" / "phase2_frozen_artifact_hashes.csv"
    if not registry_path.exists():
        freeze_phase2_inputs(root)
    frozen = pd.read_csv(registry_path)
    rows = []
    for row in frozen.to_dict("records"):
        path = root / str(row["relative_path"])
        observed_exists = path.exists()
        observed_sha = sha256_file(path) if observed_exists else ""
        status = "pass" if bool(row["exists"]) == observed_exists and str(row["sha256"]) == observed_sha else "fail"
        rows.append({**row, "observed_sha256": observed_sha, "status": status})
    audit = pd.DataFrame(rows)
    if audit["status"].eq("fail").any():
        failed = audit[audit["status"] == "fail"]["relative_path"].head(10).tolist()
        raise RuntimeError(f"Frozen Phase 2 input changed unexpectedly: {failed}")
    return audit


def save_npz(path: str | Path, **arrays: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def load_mahalanobis_npz(path: str | Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=False)
    return {
        "standardization_mean": loaded["standardization_mean"],
        "standardization_std": loaded["standardization_std"],
        "classes": {
            "0": {
                "mean": loaded["class0_mean"],
                "covariance": loaded["class0_covariance"],
                "precision": loaded["class0_precision"],
                "sample_count": int(loaded["class0_sample_count"]),
                "shrinkage": float(loaded["class0_shrinkage"]),
                "condition_number": float(loaded["class0_condition_number"]),
            },
            "1": {
                "mean": loaded["class1_mean"],
                "covariance": loaded["class1_covariance"],
                "precision": loaded["class1_precision"],
                "sample_count": int(loaded["class1_sample_count"]),
                "shrinkage": float(loaded["class1_shrinkage"]),
                "condition_number": float(loaded["class1_condition_number"]),
            },
        },
    }


def save_mahalanobis(path: str | Path, stats: dict[str, Any]) -> str:
    return save_npz(
        path,
        standardization_mean=stats["standardization_mean"],
        standardization_std=stats["standardization_std"],
        class0_mean=stats["classes"]["0"]["mean"],
        class0_covariance=stats["classes"]["0"]["covariance"],
        class0_precision=stats["classes"]["0"]["precision"],
        class0_sample_count=np.array(stats["classes"]["0"]["sample_count"]),
        class0_shrinkage=np.array(stats["classes"]["0"]["shrinkage"]),
        class0_condition_number=np.array(stats["classes"]["0"]["condition_number"]),
        class1_mean=stats["classes"]["1"]["mean"],
        class1_covariance=stats["classes"]["1"]["covariance"],
        class1_precision=stats["classes"]["1"]["precision"],
        class1_sample_count=np.array(stats["classes"]["1"]["sample_count"]),
        class1_shrinkage=np.array(stats["classes"]["1"]["shrinkage"]),
        class1_condition_number=np.array(stats["classes"]["1"]["condition_number"]),
    )


def elapsed_record(
    detector: str,
    baseline: str,
    split: str,
    stage: str,
    start_time: float,
    sample_count: int,
    reference_bank_size: int = 0,
    artifact_paths: list[str | Path] | None = None,
    gpu_device: int = 1,
) -> RuntimeRecord:
    seconds = max(time.perf_counter() - start_time, 1e-12)
    return RuntimeRecord(
        detector=detector,
        baseline=baseline,
        split=split,
        stage=stage,
        seconds=float(seconds),
        sample_count=int(sample_count),
        peak_ram_mb=process_ram_mb(),
        peak_gpu_memory_mb=gpu_memory_mb(gpu_device),
        throughput_samples_per_second=float(sample_count / seconds),
        reference_bank_size=int(reference_bank_size),
        disk_size_bytes=file_size_sum(list(artifact_paths or [])),
    )
