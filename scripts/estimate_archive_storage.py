#!/usr/bin/env python
"""Create a staged GenImage acquisition storage plan."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-summary-csv", default="artifacts/genimage_drive_size_summary.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-csv", default="artifacts/genimage_storage_acquisition_plan.csv")
    parser.add_argument("--reserve-gib", type=float, default=150.0)
    parser.add_argument(
        "--extraction-ratio",
        type=float,
        default=2.5,
        help="Estimated extracted_bytes / compressed_bytes until empirical measurement exists.",
    )
    parser.add_argument(
        "--cache-gib",
        type=float,
        default=40.0,
        help="Reserved space for checkpoints, logits, embeddings, manifests, and failed-run retention.",
    )
    args = parser.parse_args()

    rows = [row for row in read_summary(Path(args.size_summary_csv)) if row["generator_folder"] != "ALL"]
    free_bytes = shutil.disk_usage(Path(args.project_root).resolve()).free
    reserve_bytes = int(args.reserve_gib * 1024**3)
    cache_bytes = int(args.cache_gib * 1024**3)
    available_for_stage = free_bytes - reserve_bytes - cache_bytes

    plan_rows = []
    cumulative = 0
    for row in sorted(rows, key=lambda item: int(item["total_content_length_bytes"])):
        compressed = int(row["total_content_length_bytes"])
        missing_size_files = int(row["missing_size_files"])
        remote_size_complete = missing_size_files == 0
        extracted = int(compressed * args.extraction_ratio)
        stage_required = compressed + extracted
        if remote_size_complete:
            cumulative += stage_required
        plan_rows.append(
            {
                "stage": str(len(plan_rows) + 1),
                "generator_folder": row["generator_folder"],
                "remote_size_complete": str(remote_size_complete).lower(),
                "missing_size_files": str(missing_size_files),
                "compressed_bytes": str(compressed),
                "estimated_extracted_bytes": str(extracted),
                "stage_required_bytes": str(stage_required),
                "cumulative_required_bytes": str(cumulative),
                "free_after_cumulative_bytes": str(free_bytes - cache_bytes - cumulative),
                "keeps_150gib_reserve": str(remote_size_complete and (free_bytes - cache_bytes - cumulative) >= reserve_bytes).lower(),
                "notes": (
                    "blocked_missing_remote_sizes; do not acquire this stage until every object has a byte size"
                    if not remote_size_complete
                    else "order is smallest-compressed-first; replace extraction_ratio with empirical value after representative extraction"
                ),
            }
        )

    total_compressed = sum(int(row["compressed_bytes"]) for row in plan_rows)
    total_extracted = sum(int(row["estimated_extracted_bytes"]) for row in plan_rows)
    total_required = total_compressed + total_extracted + cache_bytes
    missing_total = sum(int(row["missing_size_files"]) for row in plan_rows)
    complete_total = missing_total == 0
    plan_rows.append(
        {
            "stage": "ALL",
            "generator_folder": "ALL",
            "remote_size_complete": str(complete_total).lower(),
            "missing_size_files": str(missing_total),
            "compressed_bytes": str(total_compressed),
            "estimated_extracted_bytes": str(total_extracted),
            "stage_required_bytes": str(total_compressed + total_extracted),
            "cumulative_required_bytes": str(total_required),
            "free_after_cumulative_bytes": str(free_bytes - total_required),
            "keeps_150gib_reserve": str(complete_total and (free_bytes - total_required) >= reserve_bytes).lower(),
            "notes": (
                f"blocked_missing_remote_sizes={missing_total}; free_bytes={free_bytes}; reserve_bytes={reserve_bytes}; cache_bytes={cache_bytes}; available_for_stage_bytes={available_for_stage}"
                if not complete_total
                else f"free_bytes={free_bytes}; reserve_bytes={reserve_bytes}; cache_bytes={cache_bytes}; available_for_stage_bytes={available_for_stage}"
            ),
        }
    )

    fieldnames = [
        "stage",
        "generator_folder",
        "remote_size_complete",
        "missing_size_files",
        "compressed_bytes",
        "estimated_extracted_bytes",
        "stage_required_bytes",
        "cumulative_required_bytes",
        "free_after_cumulative_bytes",
        "keeps_150gib_reserve",
        "notes",
    ]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(plan_rows)

    print(f"free_gib={free_bytes / 1024**3:.3f}")
    print(f"reserve_gib={args.reserve_gib:.3f}")
    print(f"assumed_extraction_ratio={args.extraction_ratio:.3f}")
    print(f"missing_remote_size_files={missing_total}")
    print(f"all_generators_keep_reserve={plan_rows[-1]['keeps_150gib_reserve']}")
    print(f"plan_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
