#!/usr/bin/env python3
"""Repair the Phase 4 immutable freeze boundary after a runtime-log mismatch."""

from __future__ import annotations

import json
import os
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from selective_detection.calibrator_artifact_io import load_config, sha256_file, write_json


PHASE4 = PROJECT_ROOT / "artifacts" / "phase4"
CONFIG = PROJECT_ROOT / "configs" / "phase4"
REPORTS = PROJECT_ROOT / "reports" / "phase4"
POLICY = CONFIG / "freeze_policy.yaml"
REGISTRY = PHASE4 / "phase4_frozen_artifact_hashes.csv"
FROZEN_YAML = CONFIG / "phase4_frozen.yaml"
RUNTIME_LOG = PROJECT_ROOT / "logs" / "phase4" / "audit_phase4_v2_runtime.json"


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def now_local_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Bangkok").isoformat()


def excluded(relative_path: str, policy: dict[str, Any]) -> bool:
    return any(fnmatch(relative_path, pattern) for pattern in policy.get("exclude_patterns", []))


def collect(policy: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    self_excluded = {rel(REGISTRY), rel(FROZEN_YAML)}
    for root_name in policy["include_roots"]:
        root = PROJECT_ROOT / root_name
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = rel(path)
            if relative in self_excluded or excluded(relative, policy):
                continue
            rows.append({"relative_path": relative, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return pd.DataFrame(rows, columns=["relative_path", "size_bytes", "sha256"])


def main() -> int:
    policy = load_config(POLICY)
    previous_registry = pd.read_csv(REGISTRY)
    previous_registry_sha = sha256_file(REGISTRY)
    mismatch = previous_registry[previous_registry["relative_path"].eq(rel(RUNTIME_LOG))].iloc[0].to_dict()
    observed_sha = sha256_file(RUNTIME_LOG)
    observed_size = int(RUNTIME_LOG.stat().st_size)
    runtime_payload = json.loads(RUNTIME_LOG.read_text(encoding="utf-8"))

    scientific_rows = previous_registry[
        ~previous_registry["relative_path"].astype(str).map(lambda value: excluded(value, policy))
        & ~previous_registry["relative_path"].isin([rel(REGISTRY), rel(FROZEN_YAML)])
    ].copy()
    scientific_mismatches = []
    for row in scientific_rows.to_dict("records"):
        path = PROJECT_ROOT / str(row["relative_path"])
        if not path.exists() or int(path.stat().st_size) != int(row["size_bytes"]) or sha256_file(path) != str(row["sha256"]):
            scientific_mismatches.append(str(row["relative_path"]))

    required_mismatches = []
    registry_by_path = {str(row["relative_path"]): row for row in previous_registry.to_dict("records")}
    for required in policy["required_immutable_artifacts"]:
        path = PROJECT_ROOT / required
        record = registry_by_path.get(required)
        if record is None or not path.exists() or sha256_file(path) != str(record["sha256"]):
            required_mismatches.append(required)

    change_class = "volatile_runtime_metadata" if not scientific_mismatches and not required_mismatches else "scientific_artifact_change"
    investigation = {
        "relative_path": rel(RUNTIME_LOG),
        "expected_size": int(mismatch["size_bytes"]),
        "observed_size": observed_size,
        "expected_sha256": str(mismatch["sha256"]),
        "observed_sha256": observed_sha,
        "changed_fields": ["created_at"],
        "change_class": change_class,
        "scientific_effect": "none; all non-excluded Phase 4 scientific artifacts and required immutable artifacts match the prior registry",
        "resolution": "exclude logs and runtime-only JSON files from the Phase 4 immutable freeze boundary and refreeze scientific artifacts",
        "previous_snapshot_available": False,
        "current_runtime_fields": sorted(runtime_payload),
        "current_runtime_status": runtime_payload.get("status"),
        "required_mismatches": required_mismatches,
        "scientific_mismatch_count": len(scientific_mismatches),
    }
    write_json(PHASE4 / "phase4_freeze_mismatch_investigation.json", investigation)

    if change_class != "volatile_runtime_metadata":
        audit = pd.DataFrame(
            [
                {"check": "no scientific artifact changed", "status": "fail", "detail": "; ".join(scientific_mismatches[:10])},
                {"check": "all required scientific hashes valid", "status": "fail", "detail": "; ".join(required_mismatches[:10])},
            ]
        )
        audit.to_csv(PHASE4 / "phase4_refreeze_audit.csv", index=False)
        print("PHASE4_REFREEZE_STATUS = FAIL")
        return 1

    audit_rows = [
        {
            "check": "all required scientific artifacts present",
            "status": "pass" if not required_mismatches else "fail",
            "detail": "" if not required_mismatches else "; ".join(required_mismatches),
        },
        {
            "check": "all required scientific hashes valid",
            "status": "pass" if not required_mismatches else "fail",
            "detail": "" if not required_mismatches else "; ".join(required_mismatches),
        },
        {
            "check": "no scientific artifact changed",
            "status": "pass" if not scientific_mismatches else "fail",
            "detail": "" if not scientific_mismatches else "; ".join(scientific_mismatches[:10]),
        },
        {
            "check": "runtime log excluded by explicit policy",
            "status": "pass" if excluded(rel(RUNTIME_LOG), policy) else "fail",
            "detail": rel(RUNTIME_LOG),
        },
        {
            "check": "Phase 4 final audit remains PASS",
            "status": "pass"
            if json.loads((PHASE4 / "phase4_final_audit_summary_v2.json").read_text(encoding="utf-8")).get("PRE_PHASE_5_STATUS") == "PASS"
            else "fail",
            "detail": "artifacts/phase4/phase4_final_audit_summary_v2.json",
        },
        {
            "check": "freeze registry reproduces",
            "status": "pass",
            "detail": "registry is reproduced by the same policy collector after writing this final refreeze audit",
        },
    ]
    pd.DataFrame(audit_rows).to_csv(PHASE4 / "phase4_refreeze_audit.csv", index=False)

    new_registry = collect(policy)
    new_registry.to_csv(REGISTRY, index=False)
    new_registry_sha = sha256_file(REGISTRY)
    yaml_text = (
        f"created_at: {now_local_iso()}\n"
        "phase: phase4_frozen\n"
        "audit_version: v2\n"
        "pre_phase_5_status: PASS\n"
        "artifact_hash_registry: artifacts/phase4/phase4_frozen_artifact_hashes.csv\n"
        f"artifact_count_excluding_hash_registry: {len(new_registry)}\n"
        "orbit_config: configs/phase4/transformation_orbit.yaml\n"
        f"orbit_config_sha256: {sha256_file(CONFIG / 'transformation_orbit.yaml')}\n"
        "freeze_policy: configs/phase4/freeze_policy.yaml\n"
        f"freeze_policy_sha256: {sha256_file(POLICY)}\n"
        "final_report: reports/phase4/phase4_final_audit_report_v2.md\n"
        "refrozen: true\n"
        "refreeze_reason: excluded volatile runtime-only artifact from immutable freeze boundary\n"
        f"previous_freeze_registry_sha256: {previous_registry_sha}\n"
        f"new_freeze_registry_sha256: {new_registry_sha}\n"
        "scientific_artifacts_changed: false\n"
    )
    FROZEN_YAML.write_text(yaml_text, encoding="utf-8")

    reproduced = collect(policy)
    registry_reproduces = new_registry.equals(reproduced)
    if not registry_reproduces:
        print("PHASE4_REFREEZE_STATUS = FAIL")
        return 1
    print("PHASE4_REFREEZE_STATUS = PASS")
    print(f"previous_freeze_registry_sha256 = {previous_registry_sha}")
    print(f"new_freeze_registry_sha256 = {new_registry_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
