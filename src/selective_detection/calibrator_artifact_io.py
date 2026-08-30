"""Phase 5 model and artifact I/O helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTORS = ("univfd", "safe")
SPLITS = ("split_a", "split_b")
PARTITIONS = ("risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out", "bfree_snapshot")


def now_local_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Bangkok").isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def payload_sha256(payload: dict[str, Any], *, ignore_fields: tuple[str, ...] = ("model_hash", "model_set_hash")) -> str:
    clone = copy.deepcopy(payload)
    for field in ignore_fields:
        clone.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(clone)).hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def combo_slug(detector: str, split: str) -> str:
    return f"{detector}_{split}"


def phase4_feature_path(root: Path, detector: str, split: str, partition: str) -> Path:
    return root / "artifacts" / "phase4" / "features" / detector / split / f"{partition}.parquet"


def phase5_root(root: Path) -> Path:
    return root / "artifacts" / "phase5"


def relative_to_root(root: Path, path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def default_config_payload() -> dict[str, Any]:
    return {
        "seed": 20260916,
        "model": {
            "type": "logistic_regression",
            "penalty": "l2",
            "solver": "lbfgs",
            "fit_intercept": True,
            "class_weight": None,
            "max_iter": 5000,
            "tolerance": 1.0e-10,
        },
        "regularization": {
            "candidate_C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        },
        "cross_validation": {
            "folds": 5,
            "group_key": "sha256",
            "seed": 20260916,
        },
        "selection": {
            "primary_metric": "binary_nll",
            "secondary_metric": "brier_score",
            "tertiary_metric": "aurc",
            "final_tie_breaker": "smaller_C",
            "tie_tolerance": 1.0e-12,
        },
        "calibration": {
            "ece_bins": 15,
            "ece_binning": "equal_width",
        },
    }


def write_default_config(root: Path) -> Path:
    path = root / "configs" / "phase5" / "riskguard_calibrator.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(default_config_payload(), sort_keys=False)
    except Exception:
        text = json.dumps(default_config_payload(), indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def load_phase4_freeze_policy(root: Path) -> dict[str, Any]:
    policy_path = root / "configs" / "phase4" / "freeze_policy.yaml"
    if not policy_path.exists():
        return {
            "include_roots": ["configs/phase4", "artifacts/phase4", "reports/phase4"],
            "exclude_patterns": ["logs/**", "**/*.tmp", "**/*.lock", "**/*runtime*.json"],
            "required_immutable_artifacts": [
                "artifacts/phase4/phase4_final_audit_checklist_v2.csv",
                "artifacts/phase4/phase4_final_audit_summary_v2.json",
                "reports/phase4/phase4_final_audit_report_v2.md",
                "reports/phase4/phase4_transformation_orbit_report.md",
                "configs/phase4/transformation_orbit.yaml",
            ],
        }
    return load_config(policy_path)


def phase4_policy_excludes(relative_path: str, policy: dict[str, Any]) -> bool:
    return any(fnmatch(relative_path, pattern) for pattern in policy.get("exclude_patterns", []))


def verify_frozen_inputs(
    root: Path,
    output_root: Path | None = None,
    *,
    raise_on_fail: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recompute frozen Phase 2/3/4 artifact registries."""

    registries = [
        ("phase2", root / "artifacts" / "phase2_frozen_artifact_hashes.csv"),
        ("phase3", root / "artifacts" / "phase3" / "phase3_frozen_artifact_hashes.csv"),
        ("phase4", root / "artifacts" / "phase4" / "phase4_frozen_artifact_hashes.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for phase, registry in registries:
        frozen = pd.read_csv(registry)
        for record in frozen.to_dict("records"):
            rel = str(record["relative_path"])
            path = root / rel
            expected_exists = bool(record.get("exists", True))
            observed_exists = path.exists()
            observed_size = int(path.stat().st_size) if observed_exists else 0
            observed_sha = sha256_file(path) if observed_exists else ""
            expected_size = int(record.get("size_bytes", 0))
            expected_sha = str(record.get("sha256", ""))
            ok = expected_exists == observed_exists and (not observed_exists or (observed_size == expected_size and observed_sha == expected_sha))
            rows.append(
                {
                    "phase": phase,
                    "registry": relative_to_root(root, registry),
                    "relative_path": rel,
                    "expected_exists": expected_exists,
                    "observed_exists": observed_exists,
                    "expected_size_bytes": expected_size,
                    "observed_size_bytes": observed_size,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "status": "pass" if ok else "fail",
                }
            )
        if phase == "phase4":
            phase4_yaml = load_config(root / "configs" / "phase4" / "phase4_frozen.yaml")
            policy = load_phase4_freeze_policy(root)
            policy_rel = str(phase4_yaml.get("freeze_policy", "configs/phase4/freeze_policy.yaml"))
            policy_path = root / policy_rel
            expected_policy_sha = str(phase4_yaml.get("freeze_policy_sha256", ""))
            observed_policy_sha = sha256_file(policy_path) if policy_path.exists() else ""
            rows.append(
                {
                    "phase": "phase4_policy",
                    "registry": relative_to_root(root, registry),
                    "relative_path": policy_rel,
                    "expected_exists": True,
                    "observed_exists": policy_path.exists(),
                    "expected_size_bytes": int(policy_path.stat().st_size) if policy_path.exists() else 0,
                    "observed_size_bytes": int(policy_path.stat().st_size) if policy_path.exists() else 0,
                    "expected_sha256": expected_policy_sha,
                    "observed_sha256": observed_policy_sha,
                    "status": "pass" if policy_path.exists() and expected_policy_sha == observed_policy_sha else "fail",
                }
            )
            registry_by_path = {str(row["relative_path"]): row for row in frozen.to_dict("records")}
            for required_rel in policy.get("required_immutable_artifacts", []):
                required_path = root / str(required_rel)
                observed_exists = required_path.exists()
                observed_size = int(required_path.stat().st_size) if observed_exists else 0
                observed_sha = sha256_file(required_path) if observed_exists else ""
                registry_record = registry_by_path.get(str(required_rel), {})
                expected_sha = str(registry_record.get("sha256", ""))
                expected_size = int(registry_record.get("size_bytes", 0)) if registry_record else 0
                excluded = phase4_policy_excludes(str(required_rel), policy)
                ok = bool(registry_record) and not excluded and observed_exists and observed_size == expected_size and observed_sha == expected_sha
                rows.append(
                    {
                        "phase": "phase4_required",
                        "registry": relative_to_root(root, registry),
                        "relative_path": str(required_rel),
                        "expected_exists": True,
                        "observed_exists": observed_exists,
                        "expected_size_bytes": expected_size,
                        "observed_size_bytes": observed_size,
                        "expected_sha256": expected_sha,
                        "observed_sha256": observed_sha,
                        "status": "pass" if ok else "fail",
                    }
                )
    audit = pd.DataFrame(rows)
    summary_path = root / "artifacts" / "phase4" / "phase4_final_audit_summary_v2.json"
    phase4_summary = read_json(summary_path)
    phase4_status = phase4_summary.get("PRE_PHASE_5_STATUS")
    summary = {
        "created_at": now_local_iso(),
        "row_count": int(len(audit)),
        "failed_count": int((audit["status"] != "pass").sum()),
        "phase4_pre_phase_5_status": phase4_status,
        "phase4_status_pass": phase4_status == "PASS",
        "status": "pass" if (audit["status"].eq("pass").all() and phase4_status == "PASS") else "fail",
    }
    if output_root is not None:
        out = Path(output_root)
        out.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out / "frozen_input_audit.csv", index=False)
        write_json(out / "frozen_input_audit.json", summary)
    if raise_on_fail and summary["status"] != "pass":
        failed = audit[audit["status"] != "pass"]["relative_path"].head(10).tolist()
        raise RuntimeError(f"Frozen upstream input audit failed: {failed}")
    return audit, summary


def environment_provenance(root: Path, commands: list[str], started_at: float) -> dict[str, Any]:
    import pyarrow
    import scipy
    import sklearn

    try:
        import psutil

        peak_rss_mb = float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        peak_rss_mb = float("nan")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count())
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    except Exception:
        cuda_available = False
        cuda_device_count = 0
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        git_available = True
    except Exception:
        git_commit = None
        git_status = "not a git repository"
        git_available = False
    return {
        "created_at": now_local_iso(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "operating_system": platform.platform(),
        "gpu_cuda_available": cuda_available,
        "gpu_cuda_device_count": cuda_device_count,
        "cuda_visible_devices": cuda_visible_devices,
        "git_available": git_available,
        "git_commit": git_commit,
        "git_working_tree_status": git_status,
        "commands": commands,
        "runtime_seconds": float(time.time() - started_at),
        "peak_rss_mb": peak_rss_mb,
    }
