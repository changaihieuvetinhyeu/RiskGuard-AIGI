#!/usr/bin/env python
"""Summarize a `gdown --folder --json` GenImage inventory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_inventory(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    start = text.find("[")
    if start < 0:
        raise ValueError(f"No JSON array found in {path}")
    return json.loads(text[start:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory_json")
    parser.add_argument("--out-csv", default="artifacts/genimage_drive_inventory_summary.csv")
    args = parser.parse_args()

    items = load_inventory(Path(args.inventory_json))
    counts = Counter(item["path"].split("/")[0] for item in items)
    paths = Counter(item["path"] for item in items)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["generator_folder", "file_count"])
        writer.writeheader()
        for folder in sorted(counts):
            writer.writerow({"generator_folder": folder, "file_count": counts[folder]})

    duplicates = [
        {"path": path, "count": count}
        for path, count in sorted(paths.items())
        if count > 1
    ]
    dup_path = out_path.with_name("genimage_drive_duplicate_paths.csv")
    with dup_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "count"])
        writer.writeheader()
        writer.writerows(duplicates)

    print(f"total_files={len(items)}")
    print(f"summary={out_path}")
    print(f"duplicate_paths={dup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
