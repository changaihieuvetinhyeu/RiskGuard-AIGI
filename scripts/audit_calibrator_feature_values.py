#!/usr/bin/env python3
"""Audit Phase 5 raw feature finiteness and near-zero negative anomalies."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from selective_detection.error_probability_calibrator import PRIMARY_FEATURES, RAW_FEATURE_NEGATIVE_TOLERANCE
from selective_detection.calibrator_artifact_io import DETECTORS, PARTITIONS, SPLITS, phase4_feature_path, verify_frozen_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    verify_frozen_inputs(PROJECT_ROOT, output_root, raise_on_fail=True)
    rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            for partition in PARTITIONS:
                df = pd.read_parquet(
                    phase4_feature_path(PROJECT_ROOT, detector, split, partition),
                    columns=["sample_id", "sha256", *PRIMARY_FEATURES],
                )
                for feature in PRIMARY_FEATURES:
                    values = df[feature].to_numpy(dtype=np.float64)
                    nonfinite = ~np.isfinite(values)
                    negative = values < 0.0
                    material_negative = values < -RAW_FEATURE_NEGATIVE_TOLERANCE
                    rows.append(
                        {
                            "detector": detector,
                            "split": split,
                            "partition": partition,
                            "feature": feature,
                            "row_count": int(len(df)),
                            "nonfinite_count": int(nonfinite.sum()),
                            "negative_count": int(negative.sum()),
                            "material_negative_count": int(material_negative.sum()),
                            "minimum": float(np.nanmin(values)),
                            "maximum": float(np.nanmax(values)),
                            "status": "pass" if int(nonfinite.sum()) == 0 and int(material_negative.sum()) == 0 else "fail",
                        }
                    )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_root / "raw_feature_value_audit.csv", index=False)
    examples = []
    for detector in DETECTORS:
        for split in SPLITS:
            for partition in PARTITIONS:
                df = pd.read_parquet(
                    phase4_feature_path(PROJECT_ROOT, detector, split, partition),
                    columns=["sample_id", "sha256", *PRIMARY_FEATURES],
                )
                for feature in PRIMARY_FEATURES:
                    part = df[df[feature] < 0.0].loc[:, ["sample_id", "sha256", feature]].copy()
                    if len(part):
                        part.insert(0, "feature", feature)
                        part.insert(0, "partition", partition)
                        part.insert(0, "split", split)
                        part.insert(0, "detector", detector)
                        part = part.rename(columns={feature: "raw_feature_value"})
                        examples.append(part)
    if examples:
        pd.concat(examples, ignore_index=True).to_csv(output_root / "raw_feature_negative_examples.csv", index=False)
    else:
        pd.DataFrame(columns=["detector", "split", "partition", "feature", "sample_id", "sha256", "raw_feature_value"]).to_csv(
            output_root / "raw_feature_negative_examples.csv", index=False
        )
    failures = audit[audit["status"] != "pass"]
    if len(failures):
        print(failures.to_string(index=False))
        return 1
    print(f"raw feature audit pass; numerical_negative_rows={int(audit['negative_count'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
