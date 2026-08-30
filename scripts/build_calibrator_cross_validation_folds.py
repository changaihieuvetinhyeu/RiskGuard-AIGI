#!/usr/bin/env python3
"""Build deterministic Phase 5 SHA-grouped CV folds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from selective_detection.error_probability_calibrator import FEATURE_TRANSFORMATIONS, PRIMARY_FEATURES, transformed_feature_names
from selective_detection.grouped_cross_validation import assign_sha_grouped_folds, fold_audit_rows
from selective_detection.calibrator_artifact_io import (
    DETECTORS,
    SPLITS,
    default_config_payload,
    phase4_feature_path,
    sha256_file,
    verify_frozen_inputs,
    write_default_config,
    write_json,
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


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    if args.force or not config_path.exists():
        write_default_config(PROJECT_ROOT)
    verify_frozen_inputs(PROJECT_ROOT, output_root, raise_on_fail=True)

    schema = {
        "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "primary_feature_count": 4,
        "feature_order": list(PRIMARY_FEATURES),
        "transformed_feature_order": transformed_feature_names(PRIMARY_FEATURES),
        "feature_transformations": {name: FEATURE_TRANSFORMATIONS[name] for name in PRIMARY_FEATURES},
        "forbidden_model_inputs": [
            "generator",
            "label",
            "base_prediction",
            "detector",
            "split",
            "partition",
            "source_id",
            "near_duplicate_group",
            "prediction_flip_diagnostics",
            "raw_image_features",
            "other_phase4_diagnostic_features",
        ],
        "config_sha256": sha256_file(config_path),
        "config": default_config_payload(),
    }
    write_json(output_root / "primary_feature_schema.json", schema)

    fold_dir = output_root / "cv_fold_assignments"
    fold_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    config = default_config_payload()
    n_splits = int(config["cross_validation"]["folds"])
    seed = int(config["cross_validation"]["seed"])
    for detector in selected(DETECTORS, args.detector):
        for split in selected(SPLITS, args.split):
            path = fold_dir / f"{detector}_{split}.parquet"
            if path.exists() and args.resume and not args.force:
                folds = pd.read_parquet(path)
                df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
                merged = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            else:
                df = pd.read_parquet(phase4_feature_path(PROJECT_ROOT, detector, split, "risk_fit"))
                folds = assign_sha_grouped_folds(df, n_splits=n_splits, seed=seed)
                folds.to_parquet(path, index=False)
                merged = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            audit_rows.extend(fold_audit_rows(merged, detector=detector, split=split))
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_root / "cv_fold_audit.csv", index=False)
    failures = audit[audit["status"] != "pass"]
    if len(failures):
        raise SystemExit(f"CV fold audit failed: {json.dumps(failures.head(5).to_dict('records'), sort_keys=True)}")


if __name__ == "__main__":
    main()
