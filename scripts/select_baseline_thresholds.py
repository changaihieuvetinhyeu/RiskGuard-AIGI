#!/usr/bin/env python3
"""Select Phase 3 global pooled thresholds from threshold_cal only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from selective_detection.selective_baselines import DETECTORS, MANDATORY_BASELINES, SPLITS, load_yaml, sha256_file, verify_phase2_frozen_hashes
from selective_detection.selective_thresholds import select_global_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/phase3/selective_baselines.yaml"
ARTIFACTS = PROJECT_ROOT / "artifacts/phase3"
MANIFESTS = PROJECT_ROOT / "datasets/manifests"


def main() -> int:
    verify_phase2_frozen_hashes(PROJECT_ROOT)
    cfg = load_yaml(CONFIG_PATH)
    threshold_rows = []
    curve_frames = []
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                path = ARTIFACTS / "scores" / detector / baseline / f"{split}_threshold_cal.parquet"
                scores = pd.read_parquet(path)
                for alpha in cfg["alpha_values"]:
                    result, curve = select_global_threshold(
                        scores["risk_score"].to_numpy(dtype=float),
                        scores["base_error"].to_numpy(dtype=int),
                        scores["sample_id"].astype(str).to_numpy(),
                        alpha=float(alpha),
                        delta=float(cfg["delta"]),
                    )
                    manifest_path = MANIFESTS / f"{split}_threshold_cal.csv"
                    threshold_rows.append(
                        {
                            "detector": detector,
                            "baseline": baseline,
                            "split": split,
                            "alpha": float(alpha),
                            "delta": float(cfg["delta"]),
                            "threshold": result.threshold,
                            "accepted_count": result.accepted_count,
                            "total_count": result.total_count,
                            "coverage": result.coverage,
                            "accepted_errors": result.accepted_errors,
                            "empirical_risk": result.empirical_risk,
                            "cp_upper": result.cp_upper,
                            "selection_status": result.selection_status,
                            "threshold_cal_manifest_sha256": sha256_file(manifest_path),
                        }
                    )
                    curve = curve.copy()
                    curve.insert(0, "delta", float(cfg["delta"]))
                    curve.insert(0, "alpha", float(alpha))
                    curve.insert(0, "split", split)
                    curve.insert(0, "baseline", baseline)
                    curve.insert(0, "detector", detector)
                    curve_frames.append(curve)
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(ARTIFACTS / "global_thresholds.csv", index=False)
    pd.concat(curve_frames, ignore_index=True).to_parquet(ARTIFACTS / "global_threshold_search_curves.parquet", index=False)
    print(f"Wrote {len(thresholds)} threshold rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
