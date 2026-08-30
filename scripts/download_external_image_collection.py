#!/usr/bin/env python
"""Download B-Free viral images from the pinned official CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


def md5_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_log(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "filename",
        "source_id",
        "url",
        "status",
        "http_status",
        "bytes",
        "expected_md5",
        "actual_md5",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_one(row: dict[str, str], out_root: Path, timeout: int) -> dict[str, str]:
    rel = Path(row["filename"])
    dest = out_root / rel
    part = dest.with_suffix(dest.suffix + ".part")
    expected_md5 = row.get("md5", "").lower()

    log_row = {
        "filename": row.get("filename", ""),
        "source_id": row.get("source_id", ""),
        "url": row.get("url", ""),
        "status": "",
        "http_status": "",
        "bytes": "0",
        "expected_md5": expected_md5,
        "actual_md5": "",
        "message": "",
    }

    if dest.exists() and expected_md5 and md5_file(dest) == expected_md5:
        log_row.update(status="ok_existing", bytes=str(dest.stat().st_size), actual_md5=expected_md5)
        return log_row

    headers = {}
    mode = "wb"
    existing = part.stat().st_size if part.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    try:
        with requests.get(
            row["url"],
            headers=headers,
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        ) as response:
            log_row["http_status"] = str(response.status_code)
            if response.status_code == 200 and existing:
                mode = "wb"
            if response.status_code not in {200, 206}:
                log_row.update(status="http_error", message=response.reason)
                return log_row
            dest.parent.mkdir(parents=True, exist_ok=True)
            with part.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        actual_md5 = md5_file(part)
        log_row["actual_md5"] = actual_md5
        log_row["bytes"] = str(part.stat().st_size)
        if expected_md5 and actual_md5 != expected_md5:
            log_row.update(status="md5_mismatch", message="downloaded file did not match CSV md5")
            return log_row
        part.replace(dest)
        log_row["status"] = "ok"
        return log_row
    except requests.RequestException as exc:
        log_row.update(status="request_failed", message=repr(exc))
        return log_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="third_party/B-Free/viral_images_dataset/BFree_viral_images.csv",
        help="Pinned B-Free viral metadata CSV.",
    )
    parser.add_argument("--out-root", default="datasets/external/bfree_viral")
    parser.add_argument("--log", default="logs/bfree_viral_download_log.csv")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 means all rows.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional pause between requests.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel download workers; 1 preserves serial behavior.")
    args = parser.parse_args()

    rows = read_rows(Path(args.csv))
    if args.limit:
        rows = rows[: args.limit]

    logs: list[dict[str, str]] = []
    if args.workers <= 1:
        for row in tqdm(rows, desc="B-Free viral"):
            logs.append(download_one(row, Path(args.out_root), args.timeout))
            if args.sleep:
                time.sleep(args.sleep)
            write_log(Path(args.log), logs)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_one, row, Path(args.out_root), args.timeout) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="B-Free viral"):
                logs.append(future.result())
                write_log(Path(args.log), logs)
                if args.sleep:
                    time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
