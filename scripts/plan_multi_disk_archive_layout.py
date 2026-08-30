#!/usr/bin/env python3
"""Plan a two-disk GenImage archive layout from authenticated rclone output."""

from __future__ import annotations

import argparse
import csv
import posixpath
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


EXPECTED_OBJECTS = 208
EXPECTED_BYTES = 656_414_841_547
GIB = 1024**3

DEFAULT_DISK1_PHYSICAL_ROOT = Path("/home/llm/RiskGuard-AIGI-data/genimage_disk1")
DEFAULT_DISK1_PROJECT_ROOT = Path(
    "/home/llm/AnhNT/RiskGuard-AIGI/datasets/_downloads/genimage_disk1"
)
DEFAULT_DISK2_PHYSICAL_ROOT = Path(
    "/home/llm/disk2/AnhNT/RiskGuard-AIGI-data/genimage_disk2"
)
DEFAULT_DISK2_PROJECT_ROOT = Path(
    "/home/llm/AnhNT/RiskGuard-AIGI/datasets/_downloads/genimage_disk2"
)

ARCHIVE_PART_RE = re.compile(r"^(?P<base>.+)\.(?P<suffix>zip|z\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class RemoteFile:
    generator: str
    remote_path: str
    filename: str
    size_bytes: int
    archive_family: str
    multipart_suffix: str
    remote_object_id: str
    remote_md5: str
    duplicate_remote_path: bool

    @property
    def remote_parent(self) -> str:
        parent = posixpath.dirname(self.remote_path)
        return parent or self.generator


@dataclass
class ArchiveFamily:
    generator: str
    archive_family: str
    remote_parent: str
    files: list[RemoteFile] = field(default_factory=list)
    assigned_disk: str = ""

    @property
    def size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


def archive_family_for(remote_path: str) -> tuple[str, str]:
    generator, filename = remote_path.split("/", 1)
    match = ARCHIVE_PART_RE.match(posixpath.basename(filename))
    if not match:
        stem = posixpath.splitext(posixpath.basename(filename))[0]
        return f"{generator}/{stem}", ""
    return f"{generator}/{match.group('base')}", match.group("suffix").lower()


def read_inventory(path: Path) -> list[RemoteFile]:
    raw_rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                raise ValueError(f"Expected 4 columns in {path}, got {len(row)}: {row!r}")
            remote_path, size_text, remote_object_id, remote_md5 = row
            raw_rows.append(
                {
                    "remote_path": remote_path,
                    "size_bytes": int(size_text),
                    "remote_object_id": remote_object_id,
                    "remote_md5": remote_md5,
                }
            )

    path_counts = Counter(row["remote_path"] for row in raw_rows)
    files: list[RemoteFile] = []
    for row in raw_rows:
        remote_path = row["remote_path"]
        generator = remote_path.split("/", 1)[0]
        archive_family, multipart_suffix = archive_family_for(remote_path)
        files.append(
            RemoteFile(
                generator=generator,
                remote_path=remote_path,
                filename=posixpath.basename(remote_path),
                size_bytes=row["size_bytes"],
                archive_family=archive_family,
                multipart_suffix=multipart_suffix,
                remote_object_id=row["remote_object_id"],
                remote_md5=row["remote_md5"],
                duplicate_remote_path=path_counts[remote_path] > 1,
            )
        )

    if len(files) != EXPECTED_OBJECTS:
        raise ValueError(f"Remote object count {len(files)} != expected {EXPECTED_OBJECTS}")
    total = sum(item.size_bytes for item in files)
    if total != EXPECTED_BYTES:
        raise ValueError(f"Remote byte total {total} != expected {EXPECTED_BYTES}")
    return files


def group_families(files: list[RemoteFile]) -> list[ArchiveFamily]:
    by_family: dict[str, ArchiveFamily] = {}
    for item in files:
        family = by_family.setdefault(
            item.archive_family,
            ArchiveFamily(
                generator=item.generator,
                archive_family=item.archive_family,
                remote_parent=item.remote_parent,
            ),
        )
        family.files.append(item)

    for family in by_family.values():
        family.files.sort(key=lambda item: (item.filename.lower(), item.remote_object_id))
    return sorted(by_family.values(), key=lambda item: item.archive_family)


def best_subset(units: list[tuple[str, int]], target: float, cap1: int, cap2: int) -> set[str]:
    total = sum(size for _, size in units)
    best_names: set[str] | None = None
    best_score: tuple[float, int, int] | None = None
    for mask in range(1 << len(units)):
        names: set[str] = set()
        size1 = 0
        for idx, (name, size) in enumerate(units):
            if mask & (1 << idx):
                names.add(name)
                size1 += size
        size2 = total - size1
        if size1 > cap1 or size2 > cap2:
            continue
        score = (abs(size1 - target), -size1, len(names))
        if best_score is None or score < best_score:
            best_names = names
            best_score = score
    if best_names is None:
        raise ValueError("No capacity-safe assignment exists with the available archive units")
    return best_names


def assign_families(
    families: list[ArchiveFamily],
    available_disk1: int,
    available_disk2: int,
    reserve_bytes: int,
) -> None:
    cap1 = available_disk1 - reserve_bytes
    cap2 = available_disk2 - reserve_bytes
    if cap1 <= 0 or cap2 <= 0:
        raise ValueError("Reserve exceeds available space on at least one disk")

    total = sum(family.size_bytes for family in families)
    target = total / 2

    generator_sizes = defaultdict(int)
    generator_family_counts = defaultdict(int)
    for family in families:
        generator_sizes[family.generator] += family.size_bytes
        generator_family_counts[family.generator] += 1

    generator_units = sorted(generator_sizes.items())
    try:
        disk1_generators = best_subset(generator_units, target, cap1, cap2)
        for family in families:
            family.assigned_disk = "disk1" if family.generator in disk1_generators else "disk2"
        return
    except ValueError:
        pass

    # Fallback for future inventories where a whole-generator split cannot fit.
    family_units = [(family.archive_family, family.size_bytes) for family in families]
    disk1_families = best_subset(family_units, target, cap1, cap2)
    for family in families:
        family.assigned_disk = "disk1" if family.archive_family in disk1_families else "disk2"


def local_name(item: RemoteFile) -> str:
    if not item.duplicate_remote_path:
        return item.filename
    return f"{item.filename}.candidate_{item.remote_object_id}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-with-ids",
        default="artifacts/genimage_rclone_authenticated_inventory_with_ids.csv",
    )
    parser.add_argument("--remote-files-csv", default="artifacts/genimage_remote_files.csv")
    parser.add_argument("--families-csv", default="artifacts/genimage_archive_families.csv")
    parser.add_argument(
        "--assignment-csv", default="artifacts/genimage_two_disk_assignment.csv"
    )
    parser.add_argument(
        "--summary-csv", default="artifacts/genimage_two_disk_assignment_summary.csv"
    )
    parser.add_argument("--reserve-gib", type=float, default=150.0)
    parser.add_argument("--disk1-physical-root", default=str(DEFAULT_DISK1_PHYSICAL_ROOT))
    parser.add_argument("--disk1-project-root", default=str(DEFAULT_DISK1_PROJECT_ROOT))
    parser.add_argument("--disk2-physical-root", default=str(DEFAULT_DISK2_PHYSICAL_ROOT))
    parser.add_argument("--disk2-project-root", default=str(DEFAULT_DISK2_PROJECT_ROOT))
    args = parser.parse_args()

    files = read_inventory(Path(args.inventory_with_ids))
    families = group_families(files)

    disk1_physical_root = Path(args.disk1_physical_root)
    disk1_project_root = Path(args.disk1_project_root)
    disk2_physical_root = Path(args.disk2_physical_root)
    disk2_project_root = Path(args.disk2_project_root)
    for root in [disk1_physical_root, disk1_project_root, disk2_physical_root, disk2_project_root]:
        root.mkdir(parents=True, exist_ok=True)

    available_disk1 = shutil.disk_usage(disk1_physical_root).free
    available_disk2 = shutil.disk_usage(disk2_physical_root).free
    reserve_bytes = int(args.reserve_gib * GIB)
    assign_families(families, available_disk1, available_disk2, reserve_bytes)

    family_by_name = {family.archive_family: family for family in families}
    remote_rows = []
    for item in sorted(files, key=lambda row: (row.remote_path, row.remote_object_id)):
        remote_rows.append(
            {
                "generator": item.generator,
                "remote_path": item.remote_path,
                "filename": item.filename,
                "size_bytes": str(item.size_bytes),
                "archive_family": item.archive_family,
                "multipart_suffix": item.multipart_suffix,
                "remote_object_id": item.remote_object_id,
                "duplicate_remote_path": str(item.duplicate_remote_path).lower(),
                "remote_md5": item.remote_md5,
            }
        )

    family_rows = []
    for family in families:
        suffixes = [item.multipart_suffix for item in family.files if item.multipart_suffix]
        family_rows.append(
            {
                "generator": family.generator,
                "archive_family": family.archive_family,
                "remote_parent": family.remote_parent,
                "file_count": str(len(family.files)),
                "total_size_bytes": str(family.size_bytes),
                "multipart_parts": str(
                    sum(1 for suffix in suffixes if suffix != "zip" and suffix.startswith("z"))
                ),
                "zip_members": str(sum(1 for suffix in suffixes if suffix == "zip")),
                "duplicate_remote_path_count": str(
                    sum(1 for item in family.files if item.duplicate_remote_path)
                ),
                "assigned_disk": family.assigned_disk,
            }
        )

    assignment_rows = []
    disk_family_counts = Counter(family.assigned_disk for family in families)
    assigned_bytes = Counter()
    for family in families:
        assigned_bytes[family.assigned_disk] += family.size_bytes
        physical_root = disk1_physical_root if family.assigned_disk == "disk1" else disk2_physical_root
        project_root = disk1_project_root if family.assigned_disk == "disk1" else disk2_project_root
        for item in family.files:
            name = local_name(item)
            physical_destination = physical_root / item.generator / name
            project_visible_destination = project_root / item.generator / name
            assignment_rows.append(
                {
                    "generator": item.generator,
                    "archive_family": item.archive_family,
                    "remote_parent": item.remote_parent,
                    "remote_path": item.remote_path,
                    "size_bytes": str(item.size_bytes),
                    "assigned_disk": family.assigned_disk,
                    "physical_destination": str(physical_destination),
                    "project_visible_destination": str(project_visible_destination),
                    "remote_object_id": item.remote_object_id,
                    "remote_md5": item.remote_md5,
                    "duplicate_remote_path": str(item.duplicate_remote_path).lower(),
                    "local_filename": name,
                }
            )

    assignment_rows.sort(key=lambda row: (row["assigned_disk"], row["generator"], row["remote_path"], row["remote_object_id"]))

    if assigned_bytes["disk1"] > available_disk1 - reserve_bytes:
        raise ValueError("Disk 1 assignment violates the reserve")
    if assigned_bytes["disk2"] > available_disk2 - reserve_bytes:
        raise ValueError("Disk 2 assignment violates the reserve")
    if assigned_bytes["disk1"] + assigned_bytes["disk2"] != EXPECTED_BYTES:
        raise ValueError("Assigned bytes do not match the authenticated remote total")

    summary_rows = [
        {
            "disk": "disk1",
            "mount_point": "/",
            "available_before_bytes": str(available_disk1),
            "assigned_bytes": str(assigned_bytes["disk1"]),
            "reserve_bytes": str(reserve_bytes),
            "estimated_available_after_bytes": str(available_disk1 - assigned_bytes["disk1"]),
            "archive_family_count": str(disk_family_counts["disk1"]),
            "physical_root": str(disk1_physical_root),
            "project_root": str(disk1_project_root),
        },
        {
            "disk": "disk2",
            "mount_point": "/home/llm/disk2",
            "available_before_bytes": str(available_disk2),
            "assigned_bytes": str(assigned_bytes["disk2"]),
            "reserve_bytes": str(reserve_bytes),
            "estimated_available_after_bytes": str(available_disk2 - assigned_bytes["disk2"]),
            "archive_family_count": str(disk_family_counts["disk2"]),
            "physical_root": str(disk2_physical_root),
            "project_root": str(disk2_project_root),
        },
    ]

    write_csv(
        Path(args.remote_files_csv),
        [
            "generator",
            "remote_path",
            "filename",
            "size_bytes",
            "archive_family",
            "multipart_suffix",
            "remote_object_id",
            "duplicate_remote_path",
            "remote_md5",
        ],
        remote_rows,
    )
    write_csv(
        Path(args.families_csv),
        [
            "generator",
            "archive_family",
            "remote_parent",
            "file_count",
            "total_size_bytes",
            "multipart_parts",
            "zip_members",
            "duplicate_remote_path_count",
            "assigned_disk",
        ],
        family_rows,
    )
    write_csv(
        Path(args.assignment_csv),
        [
            "generator",
            "archive_family",
            "remote_parent",
            "remote_path",
            "size_bytes",
            "assigned_disk",
            "physical_destination",
            "project_visible_destination",
            "remote_object_id",
            "remote_md5",
            "duplicate_remote_path",
            "local_filename",
        ],
        assignment_rows,
    )
    write_csv(
        Path(args.summary_csv),
        [
            "disk",
            "mount_point",
            "available_before_bytes",
            "assigned_bytes",
            "reserve_bytes",
            "estimated_available_after_bytes",
            "archive_family_count",
            "physical_root",
            "project_root",
        ],
        summary_rows,
    )

    print(f"remote_objects={len(files)}")
    print(f"remote_bytes={sum(item.size_bytes for item in files)}")
    print(f"archive_families={len(families)}")
    print(f"disk1_assigned_bytes={assigned_bytes['disk1']}")
    print(f"disk2_assigned_bytes={assigned_bytes['disk2']}")
    print(f"disk1_available_after_bytes={available_disk1 - assigned_bytes['disk1']}")
    print(f"disk2_available_after_bytes={available_disk2 - assigned_bytes['disk2']}")
    print(f"target_bytes={EXPECTED_BYTES / 2:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
