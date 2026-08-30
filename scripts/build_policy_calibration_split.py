#!/usr/bin/env python3
"""Build Phase 6 policy-select/policy-certify calibration assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selective_detection.group_risk_certification import add_common_cli, run_stage


def main() -> int:
    parser = add_common_cli(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    run_stage("calibration_split", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

