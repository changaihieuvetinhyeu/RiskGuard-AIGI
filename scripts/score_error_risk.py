#!/usr/bin/env python3
"""Materialize frozen Phase 5 RiskGuard scores."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from selective_detection.error_probability_calibrator import PRIMARY_FEATURES, load_riskguard_json, risk_logit, risk_probability
from selective_detection.calibrator_artifact_io import (
    DETECTORS,
    PARTITIONS,
    SPLITS,
    combo_slug,
    phase4_feature_path,
    read_json,
    sha256_file,
    verify_frozen_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=[*DETECTORS, "all"], default="all")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    return parser.parse_args()


def selected(values: tuple[str, ...], requested: str) -> tuple[str, ...]:
    return values if requested == "all" else (requested,)


def sigmoid(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700.0, 700.0)))


def score_audit_row(
    *,
    detector: str,
    split: str,
    artifact_name: str,
    partition: str,
    expected_rows: int,
    scored: pd.DataFrame,
) -> dict[str, Any]:
    probs = scored["risk_probability"].to_numpy(dtype=np.float64)
    logits = scored["risk_logit"].to_numpy(dtype=np.float64)
    sigmoid_gap = np.max(np.abs(sigmoid(logits) - probs)) if len(scored) else 0.0
    duplicate_rows = int(scored.duplicated(["sample_id", "sha256"]).sum())
    missing_rows = int(expected_rows - len(scored))
    ok = (
        expected_rows == len(scored)
        and missing_rows == 0
        and duplicate_rows == 0
        and np.isfinite(logits).all()
        and np.isfinite(probs).all()
        and (probs >= 0.0).all()
        and (probs <= 1.0).all()
        and sigmoid_gap <= 1.0e-12
    )
    return {
        "detector": detector,
        "split": split,
        "artifact": artifact_name,
        "partition": partition,
        "expected_rows": int(expected_rows),
        "actual_rows": int(len(scored)),
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "nonfinite_logit_count": int((~np.isfinite(logits)).sum()),
        "nonfinite_probability_count": int((~np.isfinite(probs)).sum()),
        "probability_below_zero_count": int((probs < 0.0).sum()),
        "probability_above_one_count": int((probs > 1.0).sum()),
        "max_sigmoid_probability_gap": float(sigmoid_gap),
        "status": "pass" if ok else "fail",
    }


def base_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "sha256",
        "detector",
        "split",
        "partition",
        "evaluation_role",
        "generator",
        "label",
        "base_prediction",
        "base_error",
        *PRIMARY_FEATURES,
    ]
    return df.loc[:, cols].copy()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    verify_frozen_inputs(PROJECT_ROOT, output_root, raise_on_fail=True)
    audit_rows: list[dict[str, Any]] = []
    for detector in selected(DETECTORS, args.detector):
        for split in selected(SPLITS, args.split):
            slug = combo_slug(detector, split)
            model_path = output_root / "models" / f"{slug}_riskguard.json"
            model = load_riskguard_json(model_path)
            model_hash = str(model["model_hash"])
            config_hash = str(model["config_sha256"])
            fold_model_set = read_json(output_root / "models" / f"{slug}_riskguard_oof_folds.json")
            fold_model_hash = str(fold_model_set["model_set_hash"])
            score_dir = output_root / "scores" / detector / split
            score_dir.mkdir(parents=True, exist_ok=True)

            for partition in PARTITIONS:
                feature_path = phase4_feature_path(PROJECT_ROOT, detector, split, partition)
                df = pd.read_parquet(feature_path)
                feature_hash = sha256_file(feature_path)
                if partition == "risk_fit":
                    oof_path = output_root / "oof_scores" / f"{slug}_risk_fit.parquet"
                    oof = pd.read_parquet(oof_path, columns=["sample_id", "sha256", "risk_logit", "risk_probability"])
                    scored = base_score_frame(df).merge(oof, on=["sample_id", "sha256"], how="left", validate="one_to_one")
                    scored["score_source"] = "oof_fold_model"
                    scored["model_sha256"] = fold_model_hash
                    scored["config_sha256"] = config_hash
                    scored["phase4_feature_artifact_sha256"] = feature_hash
                    out_path = score_dir / "risk_fit_oof.parquet"
                    scored.to_parquet(out_path, index=False)
                    audit_rows.append(
                        score_audit_row(
                            detector=detector,
                            split=split,
                            artifact_name="risk_fit_oof",
                            partition=partition,
                            expected_rows=len(df),
                            scored=scored,
                        )
                    )

                    full = base_score_frame(df)
                    full["risk_logit"] = risk_logit(df.loc[:, list(PRIMARY_FEATURES)], model)
                    full["risk_probability"] = risk_probability(df.loc[:, list(PRIMARY_FEATURES)], model)
                    full["score_source"] = "full_risk_fit_model"
                    full["model_sha256"] = model_hash
                    full["config_sha256"] = config_hash
                    full["phase4_feature_artifact_sha256"] = feature_hash
                    full.to_parquet(score_dir / "risk_fit_fullfit.parquet", index=False)
                    audit_rows.append(
                        score_audit_row(
                            detector=detector,
                            split=split,
                            artifact_name="risk_fit_fullfit",
                            partition=partition,
                            expected_rows=len(df),
                            scored=full,
                        )
                    )
                else:
                    scored = base_score_frame(df)
                    scored["risk_logit"] = risk_logit(df.loc[:, list(PRIMARY_FEATURES)], model)
                    scored["risk_probability"] = risk_probability(df.loc[:, list(PRIMARY_FEATURES)], model)
                    scored["score_source"] = "full_risk_fit_model"
                    scored["model_sha256"] = model_hash
                    scored["config_sha256"] = config_hash
                    scored["phase4_feature_artifact_sha256"] = feature_hash
                    scored.to_parquet(score_dir / f"{partition}.parquet", index=False)
                    audit_rows.append(
                        score_audit_row(
                            detector=detector,
                            split=split,
                            artifact_name=partition,
                            partition=partition,
                            expected_rows=len(df),
                            scored=scored,
                        )
                    )
    audit = pd.DataFrame(audit_rows)
    if output_root.joinpath("score_artifact_audit.csv").exists() and args.resume and not args.force:
        old = pd.read_csv(output_root / "score_artifact_audit.csv")
        key = ["detector", "split", "artifact"]
        old_marker = old[key].astype(str).agg("\x1f".join, axis=1)
        new_marker = set(audit[key].astype(str).agg("\x1f".join, axis=1))
        audit = pd.concat([old[~old_marker.isin(new_marker)], audit], ignore_index=True)
    audit.to_csv(output_root / "score_artifact_audit.csv", index=False)
    failures = audit[audit["status"] != "pass"]
    if len(failures):
        raise SystemExit(f"score artifact audit failed: {failures.head(5).to_dict('records')}")


if __name__ == "__main__":
    main()
