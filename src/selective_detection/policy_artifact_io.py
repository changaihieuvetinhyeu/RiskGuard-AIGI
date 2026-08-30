"""Policy and freeze I/O helpers for Phase 6."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def payload_sha256(payload: dict[str, Any], ignore_fields: tuple[str, ...] = ("policy_sha256",)) -> str:
    clone = copy.deepcopy(payload)
    for field in ignore_fields:
        clone.pop(field, None)
    return hashlib.sha256(canonical_json_bytes(clone)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        text = json.dumps(payload, indent=2, sort_keys=True)
    target.write_text(text, encoding="utf-8")


def read_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml

        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_to(root: Path, path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def freeze_paths(root: Path, paths: list[Path], output_csv: Path) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for path in sorted(paths, key=lambda item: relative_to(root, item)):
        rel = relative_to(root, path)
        if rel in seen:
            continue
        seen.add(rel)
        exists = path.exists()
        rows.append(
            {
                "relative_path": rel,
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256_file(path) if exists else "",
            }
        )
    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def verify_freeze_registry(root: Path, registry_csv: Path) -> pd.DataFrame:
    frozen = pd.read_csv(registry_csv)
    rows = []
    for record in frozen.to_dict("records"):
        rel = str(record["relative_path"])
        path = root / rel
        expected_exists = bool(record.get("exists", True))
        observed_exists = path.exists()
        observed_size = int(path.stat().st_size) if observed_exists else 0
        observed_sha = sha256_file(path) if observed_exists else ""
        expected_size = int(record.get("size_bytes", 0))
        expected_sha = str(record.get("sha256", ""))
        ok = expected_exists == observed_exists and (
            not observed_exists or (observed_size == expected_size and observed_sha == expected_sha)
        )
        rows.append(
            {
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
    return pd.DataFrame(rows)

