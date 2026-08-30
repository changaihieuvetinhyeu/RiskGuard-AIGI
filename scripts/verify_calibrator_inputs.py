#!/usr/bin/env python3
"""Verify frozen upstream inputs for Phase 5 without fitting calibrators."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selective_detection.calibrator_artifact_io import verify_frozen_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit, summary = verify_frozen_inputs(PROJECT_ROOT, Path(args.output_root))
    phase2 = audit[audit["phase"].eq("phase2")]
    phase3 = audit[audit["phase"].eq("phase3")]
    phase4 = audit[audit["phase"].eq("phase4")]
    required = audit[audit["phase"].eq("phase4_required")]
    scientific_changed = phase4[phase4["status"].ne("pass")]
    print(f"Phase 2 frozen mismatches: {int(phase2['status'].ne('pass').sum())}")
    print(f"Phase 3 frozen mismatches: {int(phase3['status'].ne('pass').sum())}")
    print(f"Phase 4 frozen mismatches: {int(phase4['status'].ne('pass').sum())}")
    print(f"Required Phase 4 artifacts missing: {int(required['observed_exists'].ne(True).sum())}")
    print(f"Scientific Phase 4 artifacts changed: {int(scientific_changed['status'].ne('pass').sum())}")
    print(f"PHASE4_REFREEZE_STATUS = {'PASS' if summary['status'] == 'pass' else 'FAIL'}")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
