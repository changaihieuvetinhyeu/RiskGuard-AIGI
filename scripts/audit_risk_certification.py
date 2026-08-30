#!/usr/bin/env python3
"""Run or audit the Phase 6 certified selective risk-control pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selective_detection.group_risk_certification import add_common_cli, run_stage


def main() -> int:
    parser = add_common_cli(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--stage",
        choices=["verify", "calibration_split", "candidates", "certify", "freeze_policy", "evaluate", "bootstrap", "audit", "run_all"],
        default="audit",
    )
    args = parser.parse_args()
    run_stage(args.stage, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

