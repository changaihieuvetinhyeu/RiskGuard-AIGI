#!/usr/bin/env python3
"""Score Phase 3 selective baselines on calibration and evaluation manifests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from selective_detection.tabular_input_schema import read_manifest_csv
from selective_detection.selective_baselines import (
    DETECTORS,
    MANDATORY_BASELINES,
    SPLITS,
    Phase2Cache,
    base_thresholds,
    elapsed_record,
    energy_risk,
    entropy_risk,
    exact_knn_distance,
    load_mahalanobis_npz,
    load_yaml,
    msp_risk,
    score_mahalanobis,
    sha256_file,
    temp_msp_risk,
    verify_phase2_frozen_hashes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/phase3/selective_baselines.yaml"
ARTIFACTS = PROJECT_ROOT / "artifacts/phase3"
MANIFESTS = PROJECT_ROOT / "datasets/manifests"


def config_hash() -> str:
    return sha256_file(CONFIG_PATH)


def load_scope_manifest(split: str, scope: str) -> tuple[pd.DataFrame, str, str]:
    if scope == "risk_fit":
        path = MANIFESTS / f"{split}_risk_fit.csv"
        df = read_manifest_csv(path)
        role = "risk_fit"
        partition = "risk_fit"
    elif scope == "threshold_cal":
        path = MANIFESTS / f"{split}_threshold_cal.csv"
        df = read_manifest_csv(path)
        role = "threshold_cal"
        partition = "threshold_cal"
    elif scope in {"protocol_seen", "protocol_held_out"}:
        path = MANIFESTS / f"verified_v2_{split}_{scope}_eval.csv"
        df = read_manifest_csv(path)
        role = scope
        partition = str(df["source_riskguard_partition"].iloc[0])
    else:
        raise ValueError(f"unknown scope {scope}")
    df["sample_id"] = df["sample_id"].astype(str)
    return df, role, partition


def base_scoring_frame(
    manifest: pd.DataFrame,
    cache: Phase2Cache,
    thresholds: pd.DataFrame,
    detector: str,
    split: str,
    baseline: str,
    role: str,
    partition: str,
) -> pd.DataFrame:
    predictions = cache.prediction_rows(manifest["sample_id"])
    keep_cols = ["sample_id", "canonical_generator", "label"]
    if "sha256" in manifest.columns:
        keep_cols.append("sha256")
    if "evaluation_split" in manifest.columns:
        keep_cols.extend(["evaluation_split", "evaluation_role", "source_riskguard_partition"])
    df = manifest[keep_cols].copy()
    if "sha256" not in df.columns:
        df = df.merge(predictions[["sample_id", "sha256"]], on="sample_id", how="left", validate="one_to_one")
    pred_cols = [
        "sample_id",
        "raw_logit",
        "fake_probability",
        "predicted_label",
        "checkpoint_sha256",
        "preprocessing_id",
    ]
    df = df.merge(predictions[pred_cols], on="sample_id", how="left", validate="one_to_one")
    threshold_row = thresholds[(thresholds["detector"] == detector) & (thresholds["split"] == split)]
    if len(threshold_row) != 1:
        raise RuntimeError(f"missing Phase 2 threshold for {detector}/{split}")
    phase2_threshold = float(threshold_row["decision_threshold"].iloc[0])
    df["detector"] = detector
    df["baseline"] = baseline
    df["split"] = split
    df["partition"] = partition
    df["evaluation_role"] = role
    df["generator"] = df["canonical_generator"].astype(str)
    df["base_logit"] = df["raw_logit"].astype(float)
    df["base_probability"] = df["fake_probability"].astype(float)
    df["base_prediction"] = (df["base_probability"] >= phase2_threshold).astype(int)
    df["base_error"] = (df["base_prediction"].astype(int) != df["label"].astype(int)).astype(int)
    df["phase2_decision_threshold"] = phase2_threshold
    df["threshold_source_partition"] = "threshold_cal"
    return df


def fit_sha(detector: str, split: str, baseline: str) -> str:
    fit_dir = ARTIFACTS / "fits" / detector / split
    if baseline == "temp_msp":
        return sha256_file(fit_dir / "temperature.json")
    if baseline == "mahalanobis":
        return sha256_file(fit_dir / "mahalanobis_stats.npz")
    if baseline == "knn":
        return sha256_file(fit_dir / "knn_selected_k.json")
    return config_hash()


def score_baseline(
    detector: str,
    split: str,
    baseline: str,
    scope: str,
    cache: Phase2Cache,
    thresholds: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest, role, partition = load_scope_manifest(split, scope)
    start = time.perf_counter()
    df = base_scoring_frame(manifest, cache, thresholds, detector, split, baseline, role, partition)
    diagnostics: dict[str, np.ndarray] = {}
    if baseline == "msp":
        risk = msp_risk(df["base_probability"].to_numpy(dtype=np.float64))
    elif baseline == "entropy":
        risk = entropy_risk(df["base_probability"].to_numpy(dtype=np.float64))
    elif baseline == "energy":
        risk = energy_risk(df["base_logit"].to_numpy(dtype=np.float64))
    elif baseline == "temp_msp":
        temp = json.loads((ARTIFACTS / "fits" / detector / split / "temperature.json").read_text(encoding="utf-8"))
        risk, temp_prob = temp_msp_risk(df["base_logit"].to_numpy(dtype=np.float64), float(temp["temperature"]))
        diagnostics["temperature_scaled_probability"] = temp_prob
        diagnostics["temperature"] = np.full(len(df), float(temp["temperature"]))
    elif baseline == "mahalanobis":
        stats = load_mahalanobis_npz(ARTIFACTS / "fits" / detector / split / "mahalanobis_stats.npz")
        embeddings = cache.embeddings_for(df["sample_id"])
        scored = score_mahalanobis(embeddings, stats)
        risk = scored["risk_score"]
        diagnostics["distance_to_real"] = scored["distance_to_real"]
        diagnostics["distance_to_fake"] = scored["distance_to_fake"]
        diagnostics["distance_to_predicted_class"] = np.where(
            df["base_prediction"].to_numpy(dtype=np.int64) == 0,
            scored["distance_to_real"],
            scored["distance_to_fake"],
        )
        diagnostics["minimum_class_distance"] = scored["risk_score"]
    elif baseline == "knn":
        fit_dir = ARTIFACTS / "fits" / detector / split
        selected = json.loads((fit_dir / "knn_selected_k.json").read_text(encoding="utf-8"))
        bank_manifest = pd.read_parquet(fit_dir / "knn_reference_bank.parquet")
        bank_embeddings = cache.embeddings_for(bank_manifest["sample_id"])
        query_embeddings = cache.embeddings_for(df["sample_id"])
        query_ids = df["sample_id"].astype(str).to_numpy() if scope == "risk_fit" else None
        scored = exact_knn_distance(
            bank_embeddings,
            query_embeddings,
            int(selected["selected_k"]),
            bank_ids=bank_manifest["sample_id"].astype(str).to_numpy(),
            query_ids=query_ids,
            device=str(cfg["knn"].get("cuda_device", "cuda:1")),
            batch_size=int(cfg["knn"].get("query_batch_size", 1024)),
        )
        risk = scored["risk_score"]
        diagnostics["neighbor_distance_1"] = scored["neighbor_distance_1"]
        diagnostics["neighbor_index_1"] = scored["neighbor_index_1"]
        diagnostics["selected_k"] = np.full(len(df), int(selected["selected_k"]))
    else:
        raise ValueError(f"unknown baseline {baseline}")
    df["risk_score"] = np.asarray(risk, dtype=np.float64)
    df["risk_orientation"] = "higher_risk_score_more_likely_to_reject"
    df["fit_artifact_sha256"] = fit_sha(detector, split, baseline)
    df["config_sha256"] = config_hash()
    for key, value in diagnostics.items():
        df[key] = value
    required = [
        "sample_id",
        "sha256",
        "detector",
        "baseline",
        "split",
        "partition",
        "evaluation_role",
        "generator",
        "label",
        "base_logit",
        "base_probability",
        "base_prediction",
        "base_error",
        "risk_score",
        "risk_orientation",
        "fit_artifact_sha256",
        "config_sha256",
        "phase2_decision_threshold",
        "threshold_source_partition",
    ]
    extra = [col for col in df.columns if col not in required and col not in {"canonical_generator", "raw_logit", "fake_probability", "predicted_label"}]
    out = df[required + extra].copy()
    if out["sample_id"].isna().any() or out["risk_score"].isna().any() or not np.isfinite(out["risk_score"]).all():
        raise RuntimeError(f"invalid scores for {detector}/{split}/{baseline}/{scope}")
    runtime = elapsed_record(detector, baseline, split, f"score_{scope}", start, len(out)).__dict__
    return out, runtime


def output_name(split: str, scope: str) -> str:
    if scope in {"risk_fit", "threshold_cal"}:
        return f"{split}_{scope}.parquet"
    return f"{split}_{scope}.parquet"


def main() -> int:
    verify_phase2_frozen_hashes(PROJECT_ROOT)
    cfg = load_yaml(CONFIG_PATH)
    thresholds = base_thresholds(PROJECT_ROOT)
    runtime_rows = []
    scopes = ("risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out")
    for detector in DETECTORS:
        cache = Phase2Cache(PROJECT_ROOT, detector)
        for baseline in MANDATORY_BASELINES:
            score_dir = ARTIFACTS / "scores" / detector / baseline
            score_dir.mkdir(parents=True, exist_ok=True)
            for split in SPLITS:
                for scope in scopes:
                    scored, runtime = score_baseline(detector, split, baseline, scope, cache, thresholds, cfg)
                    out_path = score_dir / output_name(split, scope)
                    scored.to_parquet(out_path, index=False)
                    runtime["disk_size_bytes"] = out_path.stat().st_size
                    runtime_rows.append(runtime)
                    print(f"wrote {out_path.relative_to(PROJECT_ROOT)} rows={len(scored)}")
            # Compatibility files for the path names listed in the Phase 3 spec.
            for scope in ("risk_fit", "threshold_cal"):
                combined = pd.concat(
                    [pd.read_parquet(score_dir / f"{split}_{scope}.parquet") for split in SPLITS],
                    ignore_index=True,
                )
                combined.to_parquet(score_dir / f"{scope}.parquet", index=False)
    runtime_path = ARTIFACTS / "selective_baseline_runtime.csv"
    pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False)
    existing_runtime = ARTIFACTS / "runtime_resource_audit.csv"
    if existing_runtime.exists():
        combined = pd.concat([pd.read_csv(existing_runtime), pd.DataFrame(runtime_rows)], ignore_index=True)
        combined.to_csv(existing_runtime, index=False)
    else:
        pd.DataFrame(runtime_rows).to_csv(existing_runtime, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
