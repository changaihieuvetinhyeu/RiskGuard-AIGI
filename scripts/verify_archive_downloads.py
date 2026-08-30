#!/usr/bin/env python3
"""Verify GenImage files downloaded across two physical disks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_OBJECTS = 208
EXPECTED_BYTES = 656_414_841_547
DISK1_ROOT = Path("/home/llm/RiskGuard-AIGI-data/genimage_disk1")
DISK2_ROOT = Path("/home/llm/disk2/AnhNT/RiskGuard-AIGI-data/genimage_disk2")
PROJECT_DISK1_ROOT = Path("/home/llm/AnhNT/RiskGuard-AIGI/datasets/_downloads/genimage_disk1")
PROJECT_DISK2_ROOT = Path("/home/llm/AnhNT/RiskGuard-AIGI/datasets/_downloads/genimage_disk2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_hashes(path: Path, block_size: int = 8 * 1024 * 1024) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def samefile_any(path: Path, candidates: list[Path]) -> bool:
    if not path.exists():
        return False
    return any(candidate.exists() and os.path.samefile(path, candidate) for candidate in candidates)


def ensure_hardlink_or_symlink(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        if os.path.samefile(dest, source):
            return
        dest.unlink()
    try:
        os.link(source, dest)
    except OSError:
        dest.symlink_to(source)


def run_7z_test(zip_path: Path, log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if not seven_zip:
        log_path.write_text("7z/7za was not found on PATH\n", encoding="utf-8")
        return False
    result = subprocess.run(
        [seven_zip, "t", str(zip_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return result.returncode == 0


def build_local_inventory(paths: list[Path], out_path: Path) -> None:
    rows: list[tuple[str, int]] = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not path.name.endswith(".rclone-meta.json"):
                rows.append((str(path), path.stat().st_size))
    rows.sort()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for path, size in rows:
            handle.write(f"{path},{size}\n")


def duplicate_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("duplicate_remote_path") == "true":
            groups[row["remote_path"]].append(row)
    return groups


def canonical_duplicate_path(row: dict[str, str]) -> Path:
    return Path(row["physical_destination"]).with_name(Path(row["remote_path"]).name)


def resolve_duplicates(
    rows: list[dict[str, str]],
    row_hashes: dict[str, tuple[str, str]],
    log_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    duplicate_rows: list[dict[str, str]] = []
    row_status: dict[str, str] = {}

    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[row["archive_family"]].append(row)

    for remote_path, candidates in duplicate_groups(rows).items():
        complete = []
        for row in candidates:
            path = Path(row["physical_destination"])
            key = row["remote_object_id"]
            if not path.exists() or path.stat().st_size != int(row["size_bytes"]):
                duplicate_rows.append(
                    {
                        "remote_path": remote_path,
                        "remote_object_id": key,
                        "candidate_path": str(path),
                        "size_bytes": str(path.stat().st_size) if path.exists() else "",
                        "sha256": "",
                        "resolution": "pending_missing_or_size_mismatch",
                        "canonical_path": "",
                        "alias_of": "",
                    }
                )
                row_status[key] = "missing" if not path.exists() else "size_mismatch"
                continue
            md5, sha256 = row_hashes.get(key, ("", ""))
            complete.append((row, path, md5, sha256))

        if len(complete) != len(candidates):
            continue

        canonical_path = canonical_duplicate_path(candidates[0])
        hashes = {sha256 for _, _, _, sha256 in complete}
        selected = complete[0]
        resolution = "identical_duplicate_alias"

        if len(hashes) == 1:
            ensure_hardlink_or_symlink(selected[1], canonical_path)
        else:
            resolution = "candidate_archive_test"
            family_rows = by_family[candidates[0]["archive_family"]]
            zip_rows = [row for row in family_rows if row["remote_path"].endswith(".zip")]
            selected = complete[0]
            selected_passed = False
            if zip_rows:
                zip_path = Path(zip_rows[0]["physical_destination"])
                candidate_paths = [item[1] for item in complete]
                for candidate in complete:
                    if canonical_path.exists() and samefile_any(canonical_path, candidate_paths):
                        canonical_path.unlink()
                    ensure_hardlink_or_symlink(candidate[1], canonical_path)
                    log_path = log_dir / f"duplicate_test_{safe_name(candidate[0]['remote_object_id'])}.log"
                    if run_7z_test(zip_path, log_path):
                        selected = candidate
                        selected_passed = True
                        break
                if not selected_passed:
                    resolution = "no_candidate_passed_archive_integrity"
            ensure_hardlink_or_symlink(selected[1], canonical_path)

        for row, path, _md5, sha256 in complete:
            if row["remote_object_id"] == selected[0]["remote_object_id"]:
                state = "verified"
                alias_of = ""
            else:
                state = "duplicate_alias"
                alias_of = selected[0]["remote_object_id"]
            row_status[row["remote_object_id"]] = state
            duplicate_rows.append(
                {
                    "remote_path": remote_path,
                    "remote_object_id": row["remote_object_id"],
                    "candidate_path": str(path),
                    "size_bytes": str(path.stat().st_size),
                    "sha256": sha256,
                    "resolution": resolution,
                    "canonical_path": str(canonical_path),
                    "alias_of": alias_of,
                }
            )

    return duplicate_rows, row_status


def family_zip_path(family_rows: list[dict[str, str]]) -> Path | None:
    for row in family_rows:
        if row["remote_path"].endswith(".zip"):
            return Path(row["physical_destination"])
    return None


def mount_point_for_disk(disk: str) -> str:
    return "/" if disk == "disk1" else "/home/llm/disk2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-csv", default="artifacts/genimage_two_disk_assignment.csv")
    parser.add_argument("--summary-csv", default="artifacts/genimage_two_disk_assignment_summary.csv")
    parser.add_argument("--verification-csv", default="artifacts/genimage_download_verification.csv")
    parser.add_argument("--sha-csv", default="artifacts/genimage_archive_sha256.csv")
    parser.add_argument("--integrity-csv", default="artifacts/genimage_archive_integrity.csv")
    parser.add_argument("--duplicate-csv", default="artifacts/genimage_duplicate_resolution_final.csv")
    parser.add_argument("--manifest-csv", default="datasets/manifests/genimage_full_archive_manifest.csv")
    parser.add_argument("--local-inventory-csv", default="artifacts/genimage_local_two_disk_inventory.csv")
    parser.add_argument("--report", default="reports/genimage_two_disk_acquisition_report.md")
    parser.add_argument("--skip-integrity", action="store_true")
    args = parser.parse_args()

    assignment_rows = read_csv(Path(args.assignment_csv))
    summary_rows = read_csv(Path(args.summary_csv))
    if len(assignment_rows) != EXPECTED_OBJECTS:
        raise ValueError(f"Assignment has {len(assignment_rows)} rows, expected {EXPECTED_OBJECTS}")
    assigned_bytes = sum(int(row["size_bytes"]) for row in assignment_rows)
    if assigned_bytes != EXPECTED_BYTES:
        raise ValueError(f"Assignment byte total {assigned_bytes} != expected {EXPECTED_BYTES}")

    log_dir = Path("logs/genimage_download")
    build_local_inventory([DISK1_ROOT, DISK2_ROOT], Path(args.local_inventory_csv))

    row_hashes: dict[str, tuple[str, str]] = {}
    verification_rows: list[dict[str, str]] = []
    sha_rows: list[dict[str, str]] = []
    initial_status: dict[str, str] = {}

    for row in assignment_rows:
        path = Path(row["physical_destination"])
        remote_id = row["remote_object_id"]
        actual_size = path.stat().st_size if path.exists() else None
        md5 = ""
        sha256 = ""
        if not path.exists():
            status = "missing"
        elif actual_size != int(row["size_bytes"]):
            status = "size_mismatch"
        else:
            md5, sha256 = file_hashes(path)
            row_hashes[remote_id] = (md5, sha256)
            if row.get("remote_md5") and md5 != row["remote_md5"]:
                status = "hash_mismatch"
            else:
                status = "verified"
            sha_rows.append(
                {
                    "generator": row["generator"],
                    "archive_family": row["archive_family"],
                    "remote_path": row["remote_path"],
                    "remote_object_id": remote_id,
                    "local_path": str(path),
                    "size_bytes": row["size_bytes"],
                    "sha256": sha256,
                    "md5": md5,
                }
            )
        initial_status[remote_id] = status
        verification_rows.append(
            {
                "generator": row["generator"],
                "archive_family": row["archive_family"],
                "remote_path": row["remote_path"],
                "remote_object_id": remote_id,
                "expected_path": str(path),
                "expected_size_bytes": row["size_bytes"],
                "actual_size_bytes": str(actual_size) if actual_size is not None else "",
                "sha256": sha256,
                "md5": md5,
                "status": status,
                "archive_family_members": "",
            }
        )

    duplicate_rows, duplicate_status = resolve_duplicates(assignment_rows, row_hashes, log_dir)
    for item in verification_rows:
        if item["remote_object_id"] in duplicate_status:
            item["status"] = duplicate_status[item["remote_object_id"]]

    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in assignment_rows:
        by_family[row["archive_family"]].append(row)

    integrity_rows: list[dict[str, str]] = []
    family_status: dict[str, str] = {}
    for archive_family, family_rows in sorted(by_family.items()):
        statuses = [item["status"] for item in verification_rows if item["archive_family"] == archive_family]
        zip_path = family_zip_path(family_rows)
        if any(status in {"missing", "size_mismatch", "hash_mismatch"} for status in statuses):
            integrity_status = "missing"
            log_file = ""
        elif args.skip_integrity:
            integrity_status = "verified"
            log_file = ""
        elif zip_path is None or not zip_path.exists():
            integrity_status = "missing"
            log_file = ""
        else:
            log_path = log_dir / f"integrity_{safe_name(archive_family)}.log"
            passed = run_7z_test(zip_path, log_path)
            integrity_status = "verified" if passed else "archive_integrity_failed"
            log_file = str(log_path)
        family_status[archive_family] = integrity_status
        integrity_rows.append(
            {
                "generator": family_rows[0]["generator"],
                "archive_family": archive_family,
                "physical_disk": family_rows[0]["assigned_disk"],
                "mount_point": mount_point_for_disk(family_rows[0]["assigned_disk"]),
                "zip_path": str(zip_path) if zip_path else "",
                "file_count": str(len(family_rows)),
                "total_size_bytes": str(sum(int(row["size_bytes"]) for row in family_rows)),
                "integrity_status": integrity_status,
                "log_file": log_file,
            }
        )

    member_counts = Counter(row["archive_family"] for row in assignment_rows)
    for item in verification_rows:
        item["archive_family_members"] = str(member_counts[item["archive_family"]])
        if family_status.get(item["archive_family"]) == "archive_integrity_failed":
            item["status"] = "archive_integrity_failed"

    manifest_rows: list[dict[str, str]] = []
    duplicate_selected_by_path: dict[str, str] = {}
    for row in duplicate_rows:
        if row["alias_of"] == "" and row["canonical_path"]:
            duplicate_selected_by_path[row["remote_path"]] = row["canonical_path"]

    for row in assignment_rows:
        if row.get("duplicate_remote_path") == "true":
            canonical = duplicate_selected_by_path.get(row["remote_path"])
            if not canonical or Path(row["physical_destination"]) != Path(canonical) and row["remote_object_id"] not in duplicate_status:
                continue
            if duplicate_status.get(row["remote_object_id"]) == "duplicate_alias":
                continue
            archive_path = canonical
            path_for_hash = Path(row["physical_destination"])
        else:
            archive_path = row["physical_destination"]
            path_for_hash = Path(row["physical_destination"])
        md5, sha256 = row_hashes.get(row["remote_object_id"], ("", ""))
        manifest_rows.append(
            {
                "generator": row["generator"],
                "archive_family": row["archive_family"],
                "archive_path": archive_path,
                "physical_disk": row["assigned_disk"],
                "mount_point": mount_point_for_disk(row["assigned_disk"]),
                "size_bytes": row["size_bytes"],
                "sha256": sha256 if path_for_hash.exists() else "",
                "integrity_status": family_status.get(row["archive_family"], "missing"),
            }
        )

    write_csv(
        Path(args.verification_csv),
        [
            "generator",
            "archive_family",
            "remote_path",
            "remote_object_id",
            "expected_path",
            "expected_size_bytes",
            "actual_size_bytes",
            "sha256",
            "md5",
            "status",
            "archive_family_members",
        ],
        verification_rows,
    )
    write_csv(
        Path(args.sha_csv),
        [
            "generator",
            "archive_family",
            "remote_path",
            "remote_object_id",
            "local_path",
            "size_bytes",
            "sha256",
            "md5",
        ],
        sha_rows,
    )
    write_csv(
        Path(args.integrity_csv),
        [
            "generator",
            "archive_family",
            "physical_disk",
            "mount_point",
            "zip_path",
            "file_count",
            "total_size_bytes",
            "integrity_status",
            "log_file",
        ],
        integrity_rows,
    )
    write_csv(
        Path(args.duplicate_csv),
        [
            "remote_path",
            "remote_object_id",
            "candidate_path",
            "size_bytes",
            "sha256",
            "resolution",
            "canonical_path",
            "alias_of",
        ],
        duplicate_rows,
    )
    write_csv(
        Path(args.manifest_csv),
        [
            "generator",
            "archive_family",
            "archive_path",
            "physical_disk",
            "mount_point",
            "size_bytes",
            "sha256",
            "integrity_status",
        ],
        manifest_rows,
    )

    counts = Counter(row["status"] for row in verification_rows)
    failed_archive_count = sum(1 for row in integrity_rows if row["integrity_status"] != "verified")
    disk1_actual = sum(path.stat().st_size for path in DISK1_ROOT.rglob("*") if path.is_file())
    disk2_actual = sum(path.stat().st_size for path in DISK2_ROOT.rglob("*") if path.is_file())
    df_root = shutil.disk_usage("/")
    df_disk2 = shutil.disk_usage("/home/llm/disk2")
    summary_by_disk = {row["disk"]: row for row in summary_rows}

    represented_count = counts.get("verified", 0) + counts.get("duplicate_alias", 0)
    acquisition_complete = represented_count == EXPECTED_OBJECTS and failed_archive_count == 0

    report = [
        "# GenImage Two-Disk Acquisition Report",
        "",
        f"Status: {'complete' if acquisition_complete else 'in_progress_or_attention_required'}",
        "",
        f"Disk 1 physical root: {summary_by_disk.get('disk1', {}).get('physical_root', str(DISK1_ROOT))}",
        f"Disk 2 physical root: {summary_by_disk.get('disk2', {}).get('physical_root', str(DISK2_ROOT))}",
        f"Disk 1 assigned size: {summary_by_disk.get('disk1', {}).get('assigned_bytes', '')} bytes",
        f"Disk 2 assigned size: {summary_by_disk.get('disk2', {}).get('assigned_bytes', '')} bytes",
        f"Disk 1 actual downloaded size: {disk1_actual} bytes",
        f"Disk 2 actual downloaded size: {disk2_actual} bytes",
        f"Remote object count: {EXPECTED_OBJECTS}",
        f"Verified local object count: {counts.get('verified', 0)}",
        f"Missing object count: {counts.get('missing', 0)}",
        f"Failed archive count: {failed_archive_count}",
        f"Duplicate SD1.5 resolution: {'resolved' if duplicate_rows and all(row['canonical_path'] for row in duplicate_rows) else 'pending'}",
        f"Available space remaining on /: {df_root.free} bytes",
        f"Available space remaining on /home/llm/disk2: {df_disk2.free} bytes",
        "Exact next extraction command: pending until all archive families are verified; use the manifest one family at a time.",
        "",
        "Status counts:",
    ]
    for status, count in sorted(counts.items()):
        report.append(f"- {status}: {count}")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"verification_csv={args.verification_csv}")
    print(f"manifest_csv={args.manifest_csv}")
    print(f"status_counts={dict(counts)}")
    print(f"failed_archive_count={failed_archive_count}")
    return 0 if acquisition_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
