#!/usr/bin/env python
"""Audit acquired images without modifying raw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append(pixels[offset + col] > pixels[offset + col + 1])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def iter_images(roots: list[Path]):
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def audit_one(path: Path, root: Path) -> tuple[dict[str, str], str | None]:
    try:
        size = path.stat().st_size
        if size <= 0:
            return {}, "zero_byte"
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            row = {
                "absolute_path": str(path.resolve()),
                "relative_path": str(path.relative_to(root)),
                "root": str(root),
                "file_bytes": str(size),
                "sha256": sha256_file(path),
                "phash": dhash(rgb),
                "width": str(width),
                "height": str(height),
                "mode": img.mode,
                "file_format": img.format or "",
            }
        return row, None
    except Exception as exc:  # noqa: BLE001 - audit must record every decode failure.
        return {
            "absolute_path": str(path.resolve()),
            "relative_path": str(path),
            "root": str(root),
            "error": repr(exc),
        }, "decode_failed"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Image root to audit.")
    parser.add_argument("--out-dir", default="artifacts", help="Output artifact directory.")
    args = parser.parse_args()

    roots = [Path(item).resolve() for item in args.root]
    out_dir = Path(args.out_dir)
    rows: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []

    for root in roots:
        for path in iter_images([root]):
            row, error = audit_one(path, root)
            if error:
                row["failure"] = error
                invalid.append(row)
            else:
                rows.append(row)

    audit_fields = [
        "absolute_path",
        "relative_path",
        "root",
        "file_bytes",
        "sha256",
        "phash",
        "width",
        "height",
        "mode",
        "file_format",
    ]
    write_csv(out_dir / "image_audit.csv", rows, audit_fields)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(out_dir / "image_audit.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        (out_dir / "image_audit.parquet.warning.txt").write_text(
            f"Parquet export failed; CSV was written. Reason: {exc!r}\n",
            encoding="utf-8",
        )

    invalid_fields = ["absolute_path", "relative_path", "root", "failure", "error"]
    write_csv(out_dir / "invalid_images.csv", invalid, invalid_fields)

    duplicates = []
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_phash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
        by_phash[row["phash"]].append(row)

    for digest, items in by_sha.items():
        if len(items) > 1:
            for row in items:
                duplicates.append({"sha256": digest, "absolute_path": row["absolute_path"]})
    write_csv(out_dir / "duplicate_sha256.csv", duplicates, ["sha256", "absolute_path"])

    near_duplicates = []
    for digest, items in by_phash.items():
        if len(items) > 1:
            for row in items:
                near_duplicates.append({"phash": digest, "absolute_path": row["absolute_path"]})
    write_csv(out_dir / "duplicate_phash.csv", near_duplicates, ["phash", "absolute_path"])

    license_rows = [{"root": str(root), "license_id": "pending_manual_audit"} for root in roots]
    write_csv(out_dir / "license_registry.csv", license_rows, ["root", "license_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
