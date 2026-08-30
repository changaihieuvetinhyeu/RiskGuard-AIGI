"""Explicit CSV schemas for Phase 2 manifests and audits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MANIFEST_DTYPES: dict[str, str] = {
    "sample_id": "string",
    "generator": "string",
    "canonical_generator": "string",
    "label": "Int64",
    "official_partition": "string",
    "imagenet_class": "string",
    "archive_path": "string",
    "member_path": "string",
    "physical_output_path": "string",
    "is_sd15": "boolean",
    "selection_level": "string",
    "selection_seed": "Int64",
    "source_generator": "string",
    "source_canonical_generator": "string",
    "filename": "string",
    "member_size": "Int64",
    "archive_family": "string",
    "is_sd15_corrupt_excluded": "boolean",
    "real_canonical_key": "string",
    "sha256": "string",
    "width": "Int64",
    "height": "Int64",
    "file_format": "string",
    "decode_ok": "boolean",
    "is_master_manifest_immutable": "boolean",
    "riskguard_split": "string",
    "riskguard_partition": "string",
    "relative_path": "string",
    "source_id": "string",
    "date": "string",
    "url": "string",
    "evaluation_manifest_id": "string",
    "evaluation_split": "string",
    "evaluation_role": "string",
    "source_riskguard_partition": "string",
    "root_sample_id": "string",
    "transform_parent_sample_id": "string",
    "transformation_lineage_id": "string",
    "transformation_chain": "string",
    "transform_name": "string",
    "transform_parameters_json": "string",
    "severity_tuple": "string",
    "transform_depth": "Int64",
}


def read_manifest_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a manifest CSV with the explicit schema columns present in that file."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    dtypes = {name: dtype for name, dtype in MANIFEST_DTYPES.items() if name in header.columns}
    return pd.read_csv(path, dtype=dtypes, low_memory=False, **kwargs)
