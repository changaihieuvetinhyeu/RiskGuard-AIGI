"""Manifest schema and leakage checks for RiskGuard-AIGI."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = [
    "sample_id",
    "absolute_path",
    "relative_path",
    "dataset",
    "source_id",
    "label",
    "generator",
    "content_class",
    "original_split",
    "riskguard_split",
    "protocol_direction",
    "is_external_test",
    "transformation_chain",
    "severity_tuple",
    "sha256",
    "phash",
    "width",
    "height",
    "file_format",
    "license_id",
    "acquisition_url",
    "archive_sha256",
]

TRAINLIKE_SPLITS = {"train", "calibration"}
TESTLIKE_SPLITS = {"seen_test", "unseen_test", "external_test"}


@dataclass(frozen=True)
class ManifestValidationResult:
    row_count: int
    missing_columns: tuple[str, ...]
    external_trainlike_rows: tuple[str, ...]
    overlapping_transformations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_columns
            or self.external_trainlike_rows
            or self.overlapping_transformations
        )


def transformation_key(row: dict[str, str]) -> str:
    """Return the chain/severity key used to prevent calibration-test overlap."""
    return f"{row.get('transformation_chain', '')}::{row.get('severity_tuple', '')}"


def validate_manifest_rows(rows: list[dict[str, str]]) -> ManifestValidationResult:
    columns = set(rows[0].keys()) if rows else set()
    missing = tuple(col for col in REQUIRED_COLUMNS if col not in columns)

    external_bad = []
    calibration_keys = set()
    test_keys = set()

    for idx, row in enumerate(rows, start=2):
        split = row.get("riskguard_split", "")
        is_external = row.get("is_external_test", "").strip().lower() in {"1", "true", "yes"}
        if is_external and split in TRAINLIKE_SPLITS:
            external_bad.append(row.get("sample_id") or f"line_{idx}")

        key = transformation_key(row)
        if split == "calibration":
            calibration_keys.add(key)
        elif split in TESTLIKE_SPLITS:
            test_keys.add(key)

    overlaps = tuple(sorted(calibration_keys & test_keys))
    return ManifestValidationResult(
        row_count=len(rows),
        missing_columns=missing,
        external_trainlike_rows=tuple(external_bad),
        overlapping_transformations=overlaps,
    )


def validate_manifest_csv(path: str | Path) -> ManifestValidationResult:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return validate_manifest_rows(rows)
