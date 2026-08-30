#!/usr/bin/env python3
"""Fit Phase 3 selective baselines from frozen risk_fit partitions only."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from selective_detection.tabular_input_schema import read_manifest_csv
from selective_detection.selective_baselines import (
    DETECTORS,
    SPLITS,
    Phase2Cache,
    base_thresholds,
    cross_validate_temperature,
    elapsed_record,
    fit_mahalanobis,
    fit_temperature,
    freeze_phase2_inputs,
    load_yaml,
    now_local_iso,
    save_mahalanobis,
    save_npz,
    select_knn_k_cv,
    sha256_file,
    sha256_text,
    verify_phase2_frozen_hashes,
    write_json,
)
from selective_detection.selective_metrics import sha256_deduplicate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/phase3/selective_baselines.yaml"
ARTIFACTS = PROJECT_ROOT / "artifacts/phase3"
REPORTS = PROJECT_ROOT / "reports/phase3"
LOGS = PROJECT_ROOT / "logs/phase3"
MANIFESTS = PROJECT_ROOT / "datasets/manifests"


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def risk_fit_frame(split: str, cache: Phase2Cache, thresholds: pd.DataFrame, detector: str) -> pd.DataFrame:
    manifest = read_manifest_csv(MANIFESTS / f"{split}_risk_fit.csv")
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    predictions = cache.prediction_rows(manifest["sample_id"])
    df = manifest[
        [
            "sample_id",
            "canonical_generator",
            "label",
            "riskguard_split",
            "riskguard_partition",
        ]
    ].merge(
        predictions[
            [
                "sample_id",
                "sha256",
                "raw_logit",
                "fake_probability",
                "checkpoint_sha256",
                "preprocessing_id",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    row = thresholds[(thresholds["detector"] == detector) & (thresholds["split"] == split)]
    if len(row) != 1:
        raise RuntimeError(f"Missing Phase 2 threshold for {detector}/{split}")
    threshold = float(row["decision_threshold"].iloc[0])
    df["split"] = split
    df["partition"] = "risk_fit"
    df["base_prediction"] = (df["fake_probability"].astype(float) >= threshold).astype(int)
    df["base_error"] = (df["base_prediction"].astype(int) != df["label"].astype(int)).astype(int)
    return df


def materialize_dedup_maps() -> None:
    rows = []
    summary_rows = []
    for split in SPLITS:
        for role in ("protocol_seen", "protocol_held_out"):
            path = MANIFESTS / f"verified_v2_{split}_{role}_eval.csv"
            df = read_manifest_csv(path)
            kept, dedup_map = sha256_deduplicate(df)
            dedup_map.insert(0, "evaluation_role", role)
            dedup_map.insert(0, "split", split)
            rows.append(dedup_map)
            out_path = ARTIFACTS / "dedup_manifests" / f"verified_v2_{split}_{role}_eval_sha256_dedup.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            kept.to_csv(out_path, index=False)
            summary_rows.append(
                {
                    "split": split,
                    "evaluation_role": role,
                    "source_manifest": str(path.relative_to(PROJECT_ROOT)),
                    "source_manifest_sha256": sha256_file(path),
                    "row_count": int(len(df)),
                    "sha256_deduplicated_count": int(len(kept)),
                    "removed_alias_count": int(len(df) - len(kept)),
                    "dedup_manifest": str(out_path.relative_to(PROJECT_ROOT)),
                    "dedup_manifest_sha256": sha256_file(out_path),
                    "canonical_policy": "lexicographically_smallest_sample_id_per_sha256",
                }
            )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(ARTIFACTS / "eval_sha256_dedup_map.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(ARTIFACTS / "eval_manifest_dedup_summary.csv", index=False)


def mc_dropout_availability() -> pd.DataFrame:
    rows = []
    sources = {
        "univfd": PROJECT_ROOT / "third_party/UniversalFakeDetect",
        "safe": PROJECT_ROOT / "third_party/SAFE",
    }
    for detector, root in sources.items():
        text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*.py"))
        dropout_mentions = len(re.findall(r"\bDropout\b|dropout_p\s*=\s*[1-9]", text))
        batchnorm_mentions = len(re.findall(r"\bBatchNorm[123]?d\b", text))
        if detector == "univfd":
            status = "unsupported_by_official_architecture"
            reason = (
                "Pinned Phase 2 UnivFD adapter uses CLIP ViT-L/14 image encoder plus a linear head; "
                "available local CLIP path has dropout_p=0 and no active fitted dropout head."
            )
            dropout_count = 0
            probs = ""
        else:
            status = "unsupported_by_official_architecture"
            reason = "Pinned SAFE ResNet/DWT architecture source contains batch normalization but no Dropout modules."
            dropout_count = 0
            probs = ""
        rows.append(
            {
                "detector": detector,
                "dropout_module_count": dropout_count,
                "dropout_probability_values": probs,
                "batchnorm_module_count": batchnorm_mentions,
                "status": status,
                "reason": reason,
                "source_dropout_mentions": dropout_mentions,
            }
        )
    availability = pd.DataFrame(rows)
    availability.to_csv(ARTIFACTS / "mc_dropout_availability.csv", index=False)
    return availability


def registry_row(
    component: str,
    detector: str,
    baseline: str,
    split: str,
    source_manifest: Path,
    source_df: pd.DataFrame,
    embedding_index: Path | str,
    output_artifact: Path,
) -> dict[str, object]:
    return {
        "component": component,
        "detector": detector,
        "baseline": baseline,
        "split": split,
        "source_partition": "risk_fit",
        "source_manifest": str(source_manifest.relative_to(PROJECT_ROOT)),
        "source_manifest_sha256": sha256_file(source_manifest),
        "sample_count": int(len(source_df)),
        "class_counts": json.dumps({str(k): int(v) for k, v in source_df["label"].value_counts().sort_index().items()}),
        "embedding_index": str(embedding_index),
        "embedding_index_sha256": sha256_file(embedding_index) if Path(embedding_index).exists() else "",
        "fit_seed": 20260916,
        "fit_config_sha256": config_hash(),
        "output_artifact": str(output_artifact.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output_artifact),
        "created_at": now_local_iso(),
    }


def fit_detector_split(
    detector: str,
    split: str,
    cache: Phase2Cache,
    thresholds: pd.DataFrame,
    cfg: dict,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fit_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    source_manifest = MANIFESTS / f"{split}_risk_fit.csv"
    df = risk_fit_frame(split, cache, thresholds, detector)
    labels = df["label"].to_numpy(dtype=np.int64)
    logits = df["raw_logit"].to_numpy(dtype=np.float64)
    errors = df["base_error"].to_numpy(dtype=np.int64)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    fit_dir = ARTIFACTS / "fits" / detector / split
    fit_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    temp_cfg = cfg["temperature_scaling"]
    temp_cv = cross_validate_temperature(
        logits,
        labels,
        folds=int(cfg["knn"]["cross_validation_folds"]),
        seed=int(cfg["seed"]),
        min_temperature=float(temp_cfg["min_temperature"]),
        max_temperature=float(temp_cfg["max_temperature"]),
    )
    temp_cv_path = fit_dir / "temperature_cv.csv"
    temp_cv.to_csv(temp_cv_path, index=False)
    temp = fit_temperature(
        logits,
        labels,
        min_temperature=float(temp_cfg["min_temperature"]),
        max_temperature=float(temp_cfg["max_temperature"]),
    )
    temp_payload = {
        "detector": detector,
        "split": split,
        "baseline": "temp_msp",
        "component": "temperature scalar",
        "source_partition": "risk_fit",
        "temperature": temp["temperature"],
        "risk_fit_nll": temp["nll"],
        "optimizer_success": temp["success"],
        "cv_artifact": str(temp_cv_path.relative_to(PROJECT_ROOT)),
        "cv_artifact_sha256": sha256_file(temp_cv_path),
        "config_sha256": config_hash(),
        "created_at": now_local_iso(),
    }
    temp_path = fit_dir / "temperature.json"
    write_json(temp_path, temp_payload)
    fit_rows.append(registry_row("temperature scalar", detector, "temp_msp", split, source_manifest, df, cache.cache_dir / "index.parquet", temp_path))
    runtime_rows.append(elapsed_record(detector, "temp_msp", split, "fit", start, len(df), artifact_paths=[temp_path, temp_cv_path]).__dict__)

    start = time.perf_counter()
    embeddings = cache.embeddings_for(df["sample_id"])
    mahala = fit_mahalanobis(embeddings, labels)
    mahala_path = fit_dir / "mahalanobis_stats.npz"
    save_mahalanobis(mahala_path, mahala)
    diag_rows = []
    for cls in ("0", "1"):
        cls_stats = mahala["classes"][cls]
        diag_rows.append(
            {
                "detector": detector,
                "split": split,
                "class_label": int(cls),
                "sample_count": int(cls_stats["sample_count"]),
                "shrinkage": float(cls_stats["shrinkage"]),
                "condition_number": float(cls_stats["condition_number"]),
                "covariance_finite": bool(np.isfinite(cls_stats["covariance"]).all()),
                "precision_finite": bool(np.isfinite(cls_stats["precision"]).all()),
            }
        )
    diag_path = fit_dir / "mahalanobis_diagnostics.csv"
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    fit_rows.append(registry_row("embedding z-score statistics", detector, "mahalanobis", split, source_manifest, df, cache.cache_dir / "index.parquet", mahala_path))
    fit_rows.append(registry_row("real Mahalanobis statistics", detector, "mahalanobis", split, source_manifest, df, cache.cache_dir / "index.parquet", mahala_path))
    fit_rows.append(registry_row("fake Mahalanobis statistics", detector, "mahalanobis", split, source_manifest, df, cache.cache_dir / "index.parquet", mahala_path))
    runtime_rows.append(elapsed_record(detector, "mahalanobis", split, "fit", start, len(df), artifact_paths=[mahala_path, diag_path]).__dict__)

    start = time.perf_counter()
    knn_cfg = cfg["knn"]
    selected_k, cv = select_knn_k_cv(
        embeddings,
        errors,
        sample_ids,
        candidate_k=[int(k) for k in knn_cfg["candidate_k"]],
        folds=int(knn_cfg["cross_validation_folds"]),
        seed=int(cfg["seed"]),
        device=str(knn_cfg.get("cuda_device", "cuda:1")),
    )
    cv_path = fit_dir / "knn_k_selection_cv.csv"
    cv.to_csv(cv_path, index=False)
    bank = df[["sample_id", "sha256", "label", "canonical_generator", "base_error"]].merge(
        cache.index, on="sample_id", how="left", validate="one_to_one"
    )
    bank_path = fit_dir / "knn_reference_bank.parquet"
    bank.to_parquet(bank_path, index=False)
    knn_payload = {
        "detector": detector,
        "split": split,
        "baseline": "knn",
        "component": "kNN selected k",
        "source_partition": "risk_fit",
        "selected_k": selected_k,
        "candidate_k": knn_cfg["candidate_k"],
        "selection_objective": "error_detection_AUROC",
        "reference_bank": str(bank_path.relative_to(PROJECT_ROOT)),
        "reference_bank_sha256": sha256_file(bank_path),
        "cv_artifact": str(cv_path.relative_to(PROJECT_ROOT)),
        "cv_artifact_sha256": sha256_file(cv_path),
        "config_sha256": config_hash(),
        "created_at": now_local_iso(),
    }
    knn_path = fit_dir / "knn_selected_k.json"
    write_json(knn_path, knn_payload)
    fit_rows.append(registry_row("kNN selected k", detector, "knn", split, source_manifest, df, cache.cache_dir / "index.parquet", knn_path))
    fit_rows.append(registry_row("kNN reference bank", detector, "knn", split, source_manifest, df, cache.cache_dir / "index.parquet", bank_path))
    runtime_rows.append(
        elapsed_record(
            detector,
            "knn",
            split,
            "fit",
            start,
            len(df),
            reference_bank_size=len(bank),
            artifact_paths=[knn_path, cv_path, bank_path],
        ).__dict__
    )
    return fit_rows, runtime_rows


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(CONFIG_PATH)
    freeze_phase2_inputs(PROJECT_ROOT)
    verify_phase2_frozen_hashes(PROJECT_ROOT)
    materialize_dedup_maps()
    thresholds = base_thresholds(PROJECT_ROOT)
    fit_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    for detector in DETECTORS:
        cache = Phase2Cache(PROJECT_ROOT, detector)
        for split in SPLITS:
            rows, runtimes = fit_detector_split(detector, split, cache, thresholds, cfg)
            fit_rows.extend(rows)
            runtime_rows.extend(runtimes)
    availability = mc_dropout_availability()
    for row in availability.to_dict("records"):
        output = ARTIFACTS / "mc_dropout_availability.csv"
        fit_rows.append(
            {
                "component": "MC Dropout availability decision",
                "detector": row["detector"],
                "baseline": "mc_dropout",
                "split": "all",
                "source_partition": "risk_fit",
                "source_manifest": "",
                "source_manifest_sha256": "",
                "sample_count": 0,
                "class_counts": "{}",
                "embedding_index": "",
                "embedding_index_sha256": "",
                "fit_seed": int(cfg["seed"]),
                "fit_config_sha256": config_hash(),
                "output_artifact": str(output.relative_to(PROJECT_ROOT)),
                "output_sha256": sha256_file(output),
                "created_at": now_local_iso(),
            }
        )
    registry = pd.DataFrame(fit_rows)
    registry.to_csv(ARTIFACTS / "baseline_fit_registry.csv", index=False)
    write_json(ARTIFACTS / "baseline_fit_registry.json", {"rows": fit_rows})
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(ARTIFACTS / "runtime_resource_audit.csv", index=False)
    print(f"Wrote {len(registry)} fit registry rows and {len(runtime)} runtime rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
