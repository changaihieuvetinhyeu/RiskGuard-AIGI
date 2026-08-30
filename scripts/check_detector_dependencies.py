#!/usr/bin/env python
"""Check whether official detector adapters are ready for smoke inference."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selective_detection.detector_dependency_registry import default_adapters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-csv", default="artifacts/detector_adapter_smoke_status.csv")
    args = parser.parse_args()

    checks = [adapter.check() for adapter in default_adapters(args.project_root)]
    rows = [
        {
            "detector": check.name,
            "status": check.status,
            "missing_files": ";".join(check.missing_files),
            "missing_packages": ";".join(check.missing_packages),
            "smoke_command": " ".join(check.command),
            "message": check.message,
        }
        for check in checks
    ]
    fieldnames = ["detector", "status", "missing_files", "missing_packages", "smoke_command", "message"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['detector']}: {row['status']}")
        if row["missing_packages"]:
            print(f"  missing_packages={row['missing_packages']}")
        if row["missing_files"]:
            print(f"  missing_files={row['missing_files']}")
    print(f"status_csv={args.out_csv}")
    return 0 if all(check.ready for check in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
