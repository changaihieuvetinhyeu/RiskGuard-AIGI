#!/usr/bin/env python
"""Report duplicate GenImage Drive paths and whether local hashes resolve them."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def load_inventory(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    start = text.find("[")
    if start < 0:
        raise ValueError(f"No JSON array found in {path}")
    return json.loads(text[start:])


def drive_id(url: str) -> str:
    parsed = urlparse(url)
    query_id = parse_qs(parsed.query).get("id", [""])[0]
    if query_id:
        return query_id
    match = re.search(r"/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract Drive id from {url!r}")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_size_probe(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["drive_id"]: row for row in csv.DictReader(handle)}


def local_candidate(download_root: Path, remote_path: str, file_id: str) -> Path:
    raw = download_root / remote_path
    if raw.exists():
        return raw
    suffix = raw.suffix
    isolated = raw.with_name(f"{raw.stem}__drive_{file_id}{suffix}")
    return isolated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-json", default="logs/genimage_drive_inventory.json")
    parser.add_argument("--size-probe-csv", default="artifacts/genimage_drive_size_probe.csv")
    parser.add_argument("--download-root", default="datasets/_downloads/genimage_drive")
    parser.add_argument("--out-csv", default="artifacts/genimage_drive_duplicate_resolution.csv")
    args = parser.parse_args()

    items = load_inventory(Path(args.inventory_json))
    counts = Counter(item["path"] for item in items)
    duplicates = {path for path, count in counts.items() if count > 1}
    size_probe = read_size_probe(Path(args.size_probe_csv))
    by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in items:
        if item["path"] in duplicates:
            file_id = drive_id(item["url"])
            probe = size_probe.get(file_id, {})
            local_path = local_candidate(Path(args.download_root), item["path"], file_id)
            row = {
                "path": item["path"],
                "drive_id": file_id,
                "content_length": probe.get("content_length", ""),
                "last_modified": probe.get("last_modified", ""),
                "local_path": str(local_path),
                "local_exists": str(local_path.exists()).lower(),
                "local_bytes": str(local_path.stat().st_size) if local_path.exists() else "",
                "sha256": sha256_file(local_path) if local_path.exists() else "",
                "resolution": "",
                "message": "",
            }
            by_path[item["path"]].append(row)

    rows: list[dict[str, str]] = []
    for path, path_rows in sorted(by_path.items()):
        hashes = {row["sha256"] for row in path_rows if row["sha256"]}
        local_complete = all(row["local_exists"] == "true" for row in path_rows)
        if local_complete and len(hashes) == 1:
            resolution = "same_hash_duplicate"
            message = "All isolated local copies have identical SHA-256; either copy is equivalent after archive integrity passes."
        elif local_complete and len(hashes) > 1:
            resolution = "hash_conflict"
            message = "Duplicate path maps to different local bytes; full multi-part archive integrity is required."
        else:
            resolution = "unresolved_pending_isolated_download"
            message = "Download duplicate objects to ID-suffixed paths before hashing; do not overwrite either object."
        for row in path_rows:
            row["resolution"] = resolution
            row["message"] = message
            rows.append(row)

    fieldnames = [
        "path",
        "drive_id",
        "content_length",
        "last_modified",
        "local_path",
        "local_exists",
        "local_bytes",
        "sha256",
        "resolution",
        "message",
    ]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"duplicate_paths={len(by_path)}")
    print(f"duplicate_objects={len(rows)}")
    print(f"resolution_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
