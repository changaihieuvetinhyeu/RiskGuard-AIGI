#!/usr/bin/env python
"""Probe GenImage Google Drive object sizes without downloading file bodies."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests


DRIVE_USERCONTENT = "https://drive.usercontent.google.com/download"


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


def probe_url(file_id: str) -> str:
    return f"{DRIVE_USERCONTENT}?{urlencode({'id': file_id, 'export': 'download', 'confirm': 't'})}"


def content_length(headers: requests.structures.CaseInsensitiveDict[str]) -> int | None:
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def probe_one(item: dict[str, str], timeout: int) -> dict[str, str]:
    file_id = drive_id(item["url"])
    row = {
        "path": item["path"],
        "drive_id": file_id,
        "probe_url": probe_url(file_id),
        "http_status": "",
        "content_length": "",
        "content_type": "",
        "content_disposition": "",
        "accept_ranges": "",
        "last_modified": "",
        "status": "",
        "message": "",
    }
    try:
        response = requests.head(row["probe_url"], allow_redirects=True, timeout=timeout)
        row["http_status"] = str(response.status_code)
        row["content_type"] = response.headers.get("Content-Type", "")
        row["content_disposition"] = response.headers.get("Content-Disposition", "")
        row["accept_ranges"] = response.headers.get("Accept-Ranges", "")
        row["last_modified"] = response.headers.get("Last-Modified", "")
        size = content_length(response.headers)
        if response.status_code != 200:
            row["status"] = "http_error"
            row["message"] = response.reason
        elif size is None:
            row["status"] = "missing_content_length"
            row["message"] = "HEAD response did not include integer Content-Length"
        elif "text/html" in row["content_type"].lower():
            row["status"] = "html_interstitial"
            row["message"] = "Drive returned HTML rather than an archive object"
        else:
            row["status"] = "ok"
            row["content_length"] = str(size)
        return row
    except requests.RequestException as exc:
        row["status"] = "request_failed"
        row["message"] = repr(exc)
        return row


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generator_from_path(path: str) -> str:
    return path.split("/", 1)[0]


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_generator: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_generator[generator_from_path(row["path"])].append(row)

    summary = []
    for generator in sorted(by_generator):
        items = by_generator[generator]
        ok_items = [item for item in items if item["status"] == "ok"]
        total_bytes = sum(int(item["content_length"]) for item in ok_items)
        summary.append(
            {
                "generator_folder": generator,
                "listed_files": str(len(items)),
                "size_probed_files": str(len(ok_items)),
                "missing_size_files": str(len(items) - len(ok_items)),
                "total_content_length_bytes": str(total_bytes),
                "total_content_length_gib": f"{total_bytes / 1024**3:.3f}",
            }
        )
    total = sum(int(row["total_content_length_bytes"]) for row in summary)
    summary.append(
        {
            "generator_folder": "ALL",
            "listed_files": str(len(rows)),
            "size_probed_files": str(sum(1 for row in rows if row["status"] == "ok")),
            "missing_size_files": str(sum(1 for row in rows if row["status"] != "ok")),
            "total_content_length_bytes": str(total),
            "total_content_length_gib": f"{total / 1024**3:.3f}",
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inventory_json",
        nargs="?",
        default="logs/genimage_drive_inventory.json",
        help="Raw gdown --folder --json inventory.",
    )
    parser.add_argument("--out-csv", default="artifacts/genimage_drive_size_probe.csv")
    parser.add_argument("--summary-csv", default="artifacts/genimage_drive_size_summary.csv")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    items = load_inventory(Path(args.inventory_json))
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(probe_one, item, args.timeout) for item in items]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx % 10 == 0 or idx == len(futures):
                print(f"completed={idx}/{len(futures)}", file=sys.stderr, flush=True)

    order = {item["path"] + "::" + drive_id(item["url"]): idx for idx, item in enumerate(items)}
    rows.sort(key=lambda row: order.get(row["path"] + "::" + row["drive_id"], 0))

    fieldnames = [
        "path",
        "drive_id",
        "probe_url",
        "http_status",
        "content_length",
        "content_type",
        "content_disposition",
        "accept_ranges",
        "last_modified",
        "status",
        "message",
    ]
    write_csv(Path(args.out_csv), rows, fieldnames)
    write_csv(
        Path(args.summary_csv),
        summarize(rows),
        [
            "generator_folder",
            "listed_files",
            "size_probed_files",
            "missing_size_files",
            "total_content_length_bytes",
            "total_content_length_gib",
        ],
    )

    ok = sum(1 for row in rows if row["status"] == "ok")
    total_bytes = sum(int(row["content_length"]) for row in rows if row["content_length"])
    print(f"probed_files={ok}/{len(rows)}")
    print(f"total_content_length_gib={total_bytes / 1024**3:.3f}")
    print(f"probe_csv={args.out_csv}")
    print(f"summary_csv={args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
