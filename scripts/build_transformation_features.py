#!/usr/bin/env python3
"""Run Phase 4 transformation-orbit inference, features, and audits.

The script is intentionally resumable. Use `--scope pilot` for the mandated
8192-parent pilot and `--scope full` for the complete Phase 4 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from fnmatch import fnmatch
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from selective_detection.adapters.wavelet_detector import SAFEDetector
from selective_detection.adapters.vision_transformer_detector import UnivFDDetector
from selective_detection.tabular_input_schema import read_manifest_csv
from selective_detection.transformation_orbit import (
    ORBIT_SEED,
    ORBIT_VERSION,
    VIEW_ORDER,
    apply_view,
    default_orbit_views,
    make_view_id,
    orbit_config_payload,
    transform_chain_id,
    transformed_pixel_sha256,
)
from selective_detection.reliability_features import (
    PRIMARY_FEATURES,
    embedding_drift_mean,
    margin_distance,
    orbit_logit_variance,
    orbit_support_distance_max,
)

try:
    import yaml
except ModuleNotFoundError:  # UnivFD env intentionally contains only detector dependencies.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
PHASE3 = ARTIFACTS / "phase3"
PHASE4 = ARTIFACTS / "phase4"
REPORTS = PROJECT_ROOT / "reports" / "phase4"
LOGS = PROJECT_ROOT / "logs" / "phase4"
CONFIG_DIR = PROJECT_ROOT / "configs" / "phase4"
FREEZE_POLICY = CONFIG_DIR / "freeze_policy.yaml"
ORBIT_CONFIG = CONFIG_DIR / "transformation_orbit.yaml"
MANIFESTS = PROJECT_ROOT / "datasets" / "manifests"

DETECTORS = ("univfd", "safe")
SPLITS = ("split_a", "split_b")
SCOPES = ("risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out")
PHASE4_PARTITION = {
    "risk_fit": "risk_fit",
    "threshold_cal": "threshold_cal",
    "protocol_seen": "clean_seen_test",
    "protocol_held_out": "clean_unseen_test",
    "bfree_snapshot": "bfree_snapshot",
}
CHECKPOINTS = {
    "univfd": PROJECT_ROOT / "third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
    "safe": PROJECT_ROOT / "third_party/SAFE/checkpoint/checkpoint-best.pth",
}


def now_local_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Bangkok").isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase2Cache:
    """Minimal frozen Phase 2 cache loader without Phase 3 YAML dependencies."""

    def __init__(self, project_root: str | Path, detector: str):
        self.project_root = Path(project_root)
        self.detector = detector
        self.cache_dir = self.project_root / "artifacts" / "cache" / detector / "clean"
        self.index = pd.read_parquet(self.cache_dir / "index.parquet")
        self.index["sample_id"] = self.index["sample_id"].astype(str)
        self.predictions = self._load_predictions()

    def _load_predictions(self) -> pd.DataFrame:
        frames = []
        for shard in sorted(self.index["prediction_shard"].unique()):
            frame = pd.read_parquet(shard)
            frame["sample_id"] = frame["sample_id"].astype(str)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def prediction_rows(self, sample_ids: list[str] | pd.Series | np.ndarray) -> pd.DataFrame:
        requested = pd.DataFrame({"sample_id": pd.Series(sample_ids, dtype="string"), "_order": range(len(sample_ids))})
        merged = requested.merge(self.predictions, on="sample_id", how="left")
        if merged["raw_logit"].isna().any():
            missing = merged.loc[merged["raw_logit"].isna(), "sample_id"].head(5).tolist()
            raise RuntimeError(f"missing predictions for {self.detector}: {missing}")
        return merged.sort_values("_order", kind="mergesort").drop(columns=["_order"]).reset_index(drop=True)

    def embeddings_for(self, sample_ids: list[str] | pd.Series | np.ndarray) -> np.ndarray:
        requested = pd.DataFrame({"sample_id": pd.Series(sample_ids, dtype="string"), "_order": range(len(sample_ids))})
        coords = requested.merge(self.index, on="sample_id", how="left")
        if coords["embedding_shard"].isna().any():
            missing = coords.loc[coords["embedding_shard"].isna(), "sample_id"].head(5).tolist()
            raise RuntimeError(f"missing embeddings for {self.detector}: {missing}")
        first = np.load(coords["embedding_shard"].iloc[0], mmap_mode="r")
        out = np.empty((len(coords), int(first.shape[1])), dtype=np.float32)
        for shard, group in coords.groupby("embedding_shard", sort=False):
            arr = np.load(shard, mmap_mode="r")
            out[group["_order"].to_numpy(dtype=np.int64)] = arr[group["row_offset"].to_numpy(dtype=np.int64)]
        return out


def base_thresholds(project_root: str | Path) -> pd.DataFrame:
    thresholds = pd.read_csv(Path(project_root) / "artifacts" / "phase2_clean_thresholds.csv")
    return thresholds[thresholds["threshold_source"].eq("threshold_cal")].copy()


def verify_phase2_frozen_hashes(project_root: str | Path) -> pd.DataFrame:
    root = Path(project_root)
    frozen = pd.read_csv(root / "artifacts" / "phase2_frozen_artifact_hashes.csv")
    rows = []
    for row in frozen.to_dict("records"):
        path = root / str(row["relative_path"])
        exists = path.exists()
        observed_sha = sha256_file(path) if exists else ""
        ok = bool(row["exists"]) == exists and str(row["sha256"]) == observed_sha
        rows.append({**row, "observed_sha256": observed_sha, "status": "pass" if ok else "fail"})
    audit = pd.DataFrame(rows)
    failed = audit[audit["status"] == "fail"]
    if len(failed):
        raise RuntimeError(f"Frozen Phase 2 input changed unexpectedly: {failed['relative_path'].head(10).tolist()}")
    return audit


def scope_dir(scope: str) -> Path:
    return PHASE4 if scope == "full" else PHASE4 / scope


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def phase4_freeze_policy() -> dict[str, Any]:
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


def ensure_freeze_policy() -> dict[str, Any]:
    policy = phase4_freeze_policy()
    text = (
        "include_roots:\n"
        + "".join(f"  - {root}\n" for root in policy["include_roots"])
        + "\nexclude_patterns:\n"
        + "".join(f'  - "{pattern}"\n' for pattern in policy["exclude_patterns"])
        + "\nrequired_immutable_artifacts:\n"
        + "".join(f"  - {artifact}\n" for artifact in policy["required_immutable_artifacts"])
    )
    if not FREEZE_POLICY.exists() or FREEZE_POLICY.read_text(encoding="utf-8") != text:
        FREEZE_POLICY.write_text(text, encoding="utf-8")
    return policy


def freeze_excluded(relative_path: str, policy: dict[str, Any]) -> bool:
    return any(fnmatch(relative_path, pattern) for pattern in policy["exclude_patterns"])


def upsert_csv(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if len(old) and all(key in old.columns for key in keys) and all(key in new_rows.columns for key in keys):
            marker = old[keys].astype(str).agg("\x1f".join, axis=1)
            incoming = set(new_rows[keys].astype(str).agg("\x1f".join, axis=1))
            old = old[~marker.isin(incoming)]
        combined = pd.concat([old, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined.to_csv(path, index=False)


def deterministic_uint(text: str) -> int:
    digest = hashlib.sha256(f"{ORBIT_SEED}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def logit_threshold(probability_threshold: float) -> float:
    p = min(max(float(probability_threshold), 1e-12), 1.0 - 1e-12)
    return float(math.log(p / (1.0 - p)))


def verify_phase3_frozen_hashes() -> pd.DataFrame:
    registry = PHASE3 / "phase3_frozen_artifact_hashes.csv"
    frozen = pd.read_csv(registry)
    rows = []
    for row in frozen.to_dict("records"):
        path = PROJECT_ROOT / str(row["relative_path"])
        exists = path.exists()
        observed_size = int(path.stat().st_size) if exists else 0
        observed_sha = sha256_file(path) if exists else ""
        ok = exists and observed_size == int(row["size_bytes"]) and observed_sha == str(row["sha256"])
        rows.append({**row, "observed_size_bytes": observed_size, "observed_sha256": observed_sha, "status": "pass" if ok else "fail"})
    audit = pd.DataFrame(rows)
    failed = audit[audit["status"] == "fail"]
    if len(failed):
        raise RuntimeError(f"Frozen Phase 3 input changed unexpectedly: {failed['relative_path'].head(10).tolist()}")
    return audit


def verify_frozen_inputs() -> dict[str, int]:
    phase2 = verify_phase2_frozen_hashes(PROJECT_ROOT)
    phase3 = verify_phase3_frozen_hashes()
    return {"phase2_checked": int(len(phase2)), "phase3_checked": int(len(phase3))}


def write_orbit_config() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        text = yaml.safe_dump(orbit_config_payload(), sort_keys=False)
    else:
        text = """orbit_version: phase4_orbit_v1
seed: 20260916
views:
  - view_name: identity
    parameters:
      operation: identity
  - view_name: jpeg_q75
    parameters:
      operation: jpeg
      quality: 75
      subsampling: "4:2:0"
  - view_name: resize_075_restore
    parameters:
      operation: resize_restore
      scale: 0.75
      resample: bicubic
      antialias: true
  - view_name: gaussian_blur_sigma_05
    parameters:
      operation: gaussian_blur
      sigma: 0.5
  - view_name: center_crop_090_restore
    parameters:
      operation: center_crop_restore
      crop_fraction: 0.9
      resample: bicubic
      antialias: true
"""
    ORBIT_CONFIG.write_text(text, encoding="utf-8")


def phase2_sha_table() -> pd.DataFrame:
    cache = Phase2Cache(PROJECT_ROOT, "univfd")
    return cache.predictions[["sample_id", "sha256"]].drop_duplicates("sample_id").copy()


def load_contexts(include_bfree: bool) -> pd.DataFrame:
    sha_table = phase2_sha_table()
    frames = []
    for split in SPLITS:
        for scope in SCOPES:
            if scope in {"risk_fit", "threshold_cal"}:
                path = MANIFESTS / f"{split}_{scope}.csv"
                role = scope
            else:
                path = MANIFESTS / f"verified_v2_{split}_{scope}_eval.csv"
                role = scope
            df = read_manifest_csv(path)
            df["source_sample_id"] = df["sample_id"].astype(str)
            if "sha256" not in df.columns:
                df = df.merge(sha_table, on="sample_id", how="left", validate="one_to_one")
            out = pd.DataFrame(
                {
                    "parent_sample_id": "phase4::" + split + "::" + scope + "::" + df["source_sample_id"].astype(str),
                    "source_sample_id": df["source_sample_id"].astype(str),
                    "sha256": df["sha256"].astype(str),
                    "source_path": df["physical_output_path"].astype(str),
                    "split": split,
                    "partition": PHASE4_PARTITION[scope],
                    "evaluation_role": role,
                    "generator": df["canonical_generator"].astype(str),
                    "label": df["label"].astype(int),
                    "source_id": "",
                    "near_duplicate_group": "",
                    "near_duplicate_group_source": "",
                    "scope": scope,
                }
            )
            frames.append(out)
    if include_bfree:
        bfree = read_manifest_csv(MANIFESTS / "bfree_viral_verified_snapshot.csv")
        bfree["source_sample_id"] = bfree["sample_id"].astype(str)
        bfree["near_duplicate_group"] = bfree_near_duplicate_groups(bfree)
        for split in SPLITS:
            out = pd.DataFrame(
                {
                    "parent_sample_id": "phase4::" + split + "::bfree_snapshot::" + bfree["source_sample_id"].astype(str),
                    "source_sample_id": bfree["source_sample_id"].astype(str),
                    "sha256": bfree["sha256"].astype(str),
                    "source_path": bfree["physical_output_path"].astype(str),
                    "split": split,
                    "partition": PHASE4_PARTITION["bfree_snapshot"],
                    "evaluation_role": "B-Free Viral Verified Snapshot",
                    "generator": bfree["canonical_generator"].astype(str),
                    "label": bfree["label"].astype(int),
                    "source_id": bfree["source_id"].astype(str),
                    "near_duplicate_group": bfree["near_duplicate_group"].astype(str),
                    "near_duplicate_group_source": "derived_from_phash_audit",
                    "scope": "bfree_snapshot",
                }
            )
            frames.append(out)
    contexts = pd.concat(frames, ignore_index=True)
    if contexts["sha256"].isna().any() or contexts["sha256"].eq("").any():
        raise RuntimeError("missing SHA-256 values in Phase 4 parent contexts")
    return contexts.sort_values(["split", "scope", "source_sample_id"], kind="mergesort").reset_index(drop=True)


def bfree_near_duplicate_groups(bfree: pd.DataFrame) -> pd.Series:
    audit_path = ARTIFACTS / "bfree_viral_pass1_audit" / "image_audit.parquet"
    if not audit_path.exists():
        return pd.Series([""] * len(bfree), index=bfree.index, dtype="string")
    audit = pd.read_parquet(audit_path)
    counts = audit["phash"].astype(str).value_counts()
    duplicate_phashes = set(counts[counts > 1].index)
    mapping = audit.set_index("sha256")["phash"].astype(str).to_dict()
    return bfree["sha256"].map(lambda sha: f"phash:{mapping.get(str(sha), '')}" if mapping.get(str(sha), "") in duplicate_phashes else "")


def select_pilot(contexts: pd.DataFrame, pilot_size: int) -> pd.DataFrame:
    base = contexts[contexts["scope"].isin(SCOPES)].copy()
    base["stratum_generator"] = np.where(base["label"].astype(int) == 0, "real", base["generator"].astype(str))
    strata = sorted(base.groupby(["split", "scope", "label", "stratum_generator"]).groups)
    if len(strata) == 0:
        raise RuntimeError("no strata available for pilot")
    quota = pilot_size // len(strata)
    remainder = pilot_size % len(strata)
    selected_parts = []
    used_sources: set[str] = set()
    for index, key in enumerate(strata):
        need = quota + (1 if index < remainder else 0)
        group = base[
            (base["split"] == key[0])
            & (base["scope"] == key[1])
            & (base["label"].astype(int) == int(key[2]))
            & (base["stratum_generator"] == key[3])
        ].copy()
        group["_order"] = group["parent_sample_id"].map(deterministic_uint)
        group = group.sort_values(["_order", "source_sample_id"], kind="mergesort")
        take = []
        for row in group.to_dict("records"):
            if row["source_sample_id"] in used_sources:
                continue
            take.append(row)
            used_sources.add(row["source_sample_id"])
            if len(take) == need:
                break
        selected_parts.extend(take)
    selected = pd.DataFrame(selected_parts).drop(columns=["stratum_generator", "_order"], errors="ignore")
    if len(selected) < pilot_size:
        remaining = base[~base["source_sample_id"].isin(used_sources)].copy()
        remaining["_order"] = remaining["parent_sample_id"].map(deterministic_uint)
        remaining = remaining.sort_values(["_order", "source_sample_id"], kind="mergesort")
        fill = remaining.head(pilot_size - len(selected)).drop(columns=["stratum_generator", "_order"], errors="ignore")
        selected = pd.concat([selected, fill], ignore_index=True)
    selected = selected.head(pilot_size).sort_values(["split", "scope", "source_sample_id"], kind="mergesort").reset_index(drop=True)
    validate_pilot(selected, pilot_size)
    return selected


def validate_pilot(pilot: pd.DataFrame, pilot_size: int) -> None:
    if len(pilot) != pilot_size:
        raise RuntimeError(f"pilot has {len(pilot)} rows, expected {pilot_size}")
    if pilot["source_sample_id"].nunique() != len(pilot):
        raise RuntimeError("pilot source_sample_id values are not unique")
    if set(pilot["split"]) != set(SPLITS):
        raise RuntimeError("pilot does not include both splits")
    if set(pilot["scope"]) != set(SCOPES):
        raise RuntimeError("pilot does not include all four GenImage scopes")
    if set(pilot["label"].astype(int)) != {0, 1}:
        raise RuntimeError("pilot does not include both labels")
    fake_generators = set(pilot.loc[pilot["label"].astype(int) == 1, "generator"].astype(str))
    expected = {"adm", "biggan", "glide", "midjourney", "sd14", "sd15", "vqdm", "wukong"}
    if fake_generators != expected:
        raise RuntimeError(f"pilot fake generators mismatch: {sorted(fake_generators)}")


def prepare(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    PHASE4.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_inputs()
    write_orbit_config()
    contexts = load_contexts(include_bfree=args.scope == "full")
    parents = select_pilot(contexts, args.pilot_size) if args.scope == "pilot" else contexts
    out_dir = scope_dir(args.scope)
    out_dir.mkdir(parents=True, exist_ok=True)
    parent_path = out_dir / "parent_context_manifest.parquet"
    parents.to_parquet(parent_path, index=False)
    orbit_path = out_dir / "transformation_orbit_manifest.parquet"
    materialize_orbit_manifest(parents, orbit_path, args.scope, args.audit_parents, args.manifest_workers)
    summary = {
        "created_at": now_local_iso(),
        "scope": args.scope,
        "parent_rows": int(len(parents)),
        "expected_view_rows": int(len(parents) * len(VIEW_ORDER)),
        "parent_manifest": rel(parent_path),
        "orbit_manifest": rel(orbit_path),
        "orbit_config": rel(ORBIT_CONFIG),
        "orbit_config_sha256": sha256_file(ORBIT_CONFIG),
        "frozen_inputs": frozen,
        "seconds": time.perf_counter() - started,
    }
    write_json(out_dir / "prepare_status.json", summary)
    print(json.dumps(summary, indent=2))


def save_audit_subset(parents: pd.DataFrame, audit_dir: Path, audit_parent_count: int) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    for parent_counter, parent in enumerate(parents.head(audit_parent_count).to_dict("records")):
        with Image.open(parent["source_path"]) as handle:
            source = handle.convert("RGB")
            for view_index, view in enumerate(default_orbit_views()):
                transformed = apply_view(source, view.view_name)
                stem = hashlib.sha256(str(parent["parent_sample_id"]).encode("utf-8")).hexdigest()[:16]
                transformed.save(audit_dir / f"{stem}_{view_index}_{view.view_name}.png")


def orbit_rows_for_parent(parent: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    with Image.open(parent["source_path"]) as handle:
        source = handle.convert("RGB")
        for view_index, view in enumerate(default_orbit_views()):
            transformed = apply_view(source, view.view_name)
            chain_id = transform_chain_id(view)
            view_id = make_view_id(str(parent["parent_sample_id"]), str(parent["sha256"]), chain_id)
            rows.append(
                {
                    "parent_sample_id": parent["parent_sample_id"],
                    "source_sample_id": parent["source_sample_id"],
                    "parent_sha256": parent["sha256"],
                    "view_id": view_id,
                    "view_name": view.view_name,
                    "view_index": view_index,
                    "transform_chain_id": chain_id,
                    "transform_parameters": view.parameter_json,
                    "split": parent["split"],
                    "partition": parent["partition"],
                    "evaluation_role": parent["evaluation_role"],
                    "generator": parent["generator"],
                    "label": int(parent["label"]),
                    "source_path": parent["source_path"],
                    "transformed_pixel_sha256": transformed_pixel_sha256(transformed),
                    "source_id": parent.get("source_id", ""),
                    "near_duplicate_group": parent.get("near_duplicate_group", ""),
                    "near_duplicate_group_source": parent.get("near_duplicate_group_source", ""),
                    "orbit_version": ORBIT_VERSION,
                }
            )
    return rows


def materialize_orbit_manifest(parents: pd.DataFrame, orbit_path: Path, scope: str, audit_parent_count: int, workers: int) -> None:
    orbit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = orbit_path.parent / "audit_subset_pixels"
    save_audit_subset(parents, audit_dir, audit_parent_count)
    if orbit_path.exists():
        orbit_path.unlink()
    writer: pq.ParquetWriter | None = None
    batch_rows: list[dict[str, Any]] = []
    parent_records = parents.to_dict("records")
    parent_counter = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            row_iter = pool.map(orbit_rows_for_parent, parent_records, chunksize=8)
            for row_group in row_iter:
                batch_rows.extend(row_group)
                parent_counter += 1
                if len(batch_rows) >= 25000:
                    writer = write_manifest_batch(orbit_path, batch_rows, writer)
                    batch_rows = []
                    print(f"materialized orbit rows for {parent_counter:,} parents ({scope})", flush=True)
    else:
        for parent in parent_records:
            batch_rows.extend(orbit_rows_for_parent(parent))
            parent_counter += 1
            if len(batch_rows) >= 25000:
                writer = write_manifest_batch(orbit_path, batch_rows, writer)
                batch_rows = []
                print(f"materialized orbit rows for {parent_counter:,} parents ({scope})", flush=True)
    if batch_rows:
        writer = write_manifest_batch(orbit_path, batch_rows, writer)
    if writer is not None:
        writer.close()


def write_manifest_batch(path: Path, rows: list[dict[str, Any]], writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    return writer


def load_detector(detector: str, device: str):
    if detector == "univfd":
        return UnivFDDetector(device=device)
    if detector == "safe":
        return SAFEDetector(device=device)
    raise ValueError(detector)


class IdentityStore:
    def __init__(self, detector: str):
        self.detector = detector
        self.phase2 = Phase2Cache(PROJECT_ROOT, detector)
        self.bfree_pred = pd.read_parquet(ARTIFACTS / f"bfree_viral_verified_{detector}_predictions.parquet")
        self.bfree_pred["sample_id"] = self.bfree_pred["sample_id"].astype(str)
        self.bfree_emb = np.load(ARTIFACTS / f"bfree_viral_verified_{detector}_embeddings.npy", mmap_mode="r")
        self.bfree_index = {sid: idx for idx, sid in enumerate(self.bfree_pred["sample_id"].astype(str))}

    def rows_and_embeddings(self, sample_ids: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        rows: list[pd.DataFrame] = []
        embeddings: list[np.ndarray] = []
        output_rows: list[pd.Series] = []
        output_embeddings: list[np.ndarray] = []
        genimage_positions = [i for i, sid in enumerate(sample_ids) if not sid.startswith("bfree_viral_")]
        if genimage_positions:
            ids = [sample_ids[i] for i in genimage_positions]
            pred = self.phase2.prediction_rows(ids)
            emb = self.phase2.embeddings_for(ids)
            rows.append(pred)
            embeddings.append(emb)
        gen_lookup = {}
        if rows:
            pred = rows[0].reset_index(drop=True)
            emb = embeddings[0]
            for local_idx, source_id in enumerate(pred["sample_id"].astype(str)):
                gen_lookup[source_id] = (pred.iloc[local_idx], emb[local_idx])
        bfree_lookup = self.bfree_pred.set_index("sample_id", drop=False)
        for sid in sample_ids:
            if sid.startswith("bfree_viral_"):
                idx = self.bfree_index[sid]
                output_rows.append(bfree_lookup.loc[sid])
                output_embeddings.append(np.asarray(self.bfree_emb[idx], dtype=np.float32))
            else:
                row, emb = gen_lookup[sid]
                output_rows.append(row)
                output_embeddings.append(np.asarray(emb, dtype=np.float32))
        return pd.DataFrame(output_rows).reset_index(drop=True), np.vstack(output_embeddings).astype(np.float32)


def infer(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    verify_frozen_inputs()
    parent_path = scope_dir(args.scope) / "parent_context_manifest.parquet"
    parents = pd.read_parquet(parent_path)
    out_dir = scope_dir(args.scope) / "orbit_cache" / args.detector
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(CHECKPOINTS[args.detector])
    detector = load_detector(args.detector, args.device)
    identity = IdentityStore(args.detector)
    views = default_orbit_views()
    embedding_dim = int(detector.embedding_dimension)
    status = {
        "detector": args.detector,
        "scope": args.scope,
        "device": args.device,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "batch_size": int(args.batch_size),
        "image_workers": int(args.image_workers),
        "shard_size": int(args.shard_size),
        "checkpoint_sha256": checkpoint_sha,
        "preprocessing_id": detector.preprocessing_id,
        "shards": [],
        "started_at": now_local_iso(),
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for shard_id, start in enumerate(range(0, len(parents), args.shard_size)):
        shard = parents.iloc[start : start + args.shard_size].reset_index(drop=True)
        pred_path = out_dir / f"predictions_{shard_id:05d}.parquet"
        emb_path = out_dir / f"embeddings_{shard_id:05d}.npy"
        expected_rows = len(shard) * len(VIEW_ORDER)
        if validate_cache_shard(pred_path, emb_path, expected_rows):
            print(f"skipping verified Phase 4 shard {args.detector}/{args.scope}/{shard_id:05d}", flush=True)
        else:
            print(f"running Phase 4 shard {args.detector}/{args.scope}/{shard_id:05d} parents={len(shard):,}", flush=True)
            shard_start = time.perf_counter()
            rows, embeddings = infer_shard(
                shard,
                detector,
                identity,
                views,
                checkpoint_sha,
                embedding_dim,
                args.batch_size,
                args.image_workers,
                args.device,
            )
            if len(rows) != expected_rows or embeddings.shape != (expected_rows, embedding_dim):
                raise RuntimeError(f"cache shard shape mismatch: rows={len(rows)} embeddings={embeddings.shape}")
            assert_finite_outputs(rows, embeddings)
            rows["runtime_ms"] = np.float32(((time.perf_counter() - shard_start) * 1000.0) / expected_rows)
            rows.to_parquet(pred_path, index=False)
            np.save(emb_path, embeddings.astype(np.float32))
        status["shards"].append({"shard_id": shard_id, "rows": expected_rows, "prediction_shard": rel(pred_path), "embedding_shard": rel(emb_path)})
    write_cache_index(out_dir)
    status["rows"] = int(len(parents) * len(VIEW_ORDER))
    status["parents"] = int(len(parents))
    status["seconds"] = float(time.perf_counter() - started)
    status["peak_cuda_memory_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if args.device.startswith("cuda") and torch.cuda.is_available() else 0.0
    )
    write_json(out_dir / "status.json", status)
    print(json.dumps({k: status[k] for k in ["detector", "scope", "parents", "rows", "seconds", "peak_cuda_memory_mb"]}, indent=2))


def validate_cache_shard(pred_path: Path, emb_path: Path, expected_rows: int) -> bool:
    if not pred_path.exists() or not emb_path.exists():
        return False
    try:
        pred = pd.read_parquet(pred_path, columns=["view_id"])
        emb = np.load(emb_path, mmap_mode="r")
        return len(pred) == expected_rows and emb.shape[0] == expected_rows
    except Exception:
        return False


def infer_shard(
    shard: pd.DataFrame,
    detector: Any,
    identity: IdentityStore,
    views: tuple[Any, ...],
    checkpoint_sha: str,
    embedding_dim: int,
    batch_size: int,
    image_workers: int,
    device: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    total_rows = len(shard) * len(VIEW_ORDER)
    row_slots: list[dict[str, Any] | None] = [None] * total_rows
    embeddings = np.empty((total_rows, embedding_dim), dtype=np.float32)
    parent_records = shard.to_dict("records")
    identity_rows, identity_emb = identity.rows_and_embeddings(shard["source_sample_id"].astype(str).tolist())
    for parent_index, parent in enumerate(parent_records):
        base_slot = parent_index * len(VIEW_ORDER)
        identity_view = views[0]
        id_pred = identity_rows.iloc[parent_index]
        row_slots[base_slot] = prediction_row(
            parent,
            identity_view,
            detector.name,
            float(id_pred["raw_logit"]),
            float(id_pred["fake_probability"]),
            str(id_pred["checkpoint_sha256"]),
            str(id_pred["preprocessing_id"]),
            embedding_dim,
            predicted_label=int(id_pred["predicted_label"]),
        )
        embeddings[base_slot] = identity_emb[parent_index]
    if image_workers > 0:
        infer_shard_with_tensor_loader(
            parent_records,
            row_slots,
            embeddings,
            detector,
            views,
            checkpoint_sha,
            embedding_dim,
            batch_size,
            image_workers,
            device,
        )
        if any(row is None for row in row_slots):
            raise RuntimeError("internal error: incomplete prediction row slots")
        return pd.DataFrame(row_slots), embeddings

    pending_images: list[Image.Image] = []
    pending_meta: list[tuple[int, dict[str, Any], Any]] = []

    def flush() -> None:
        nonlocal pending_images, pending_meta
        if not pending_images:
            return
        logits, emb = detector.predict(pending_images)
        probs = torch.sigmoid(logits)
        for offset, (slot, parent, view) in enumerate(pending_meta):
            probability = float(probs[offset].item())
            logit = float(logits[offset].item())
            row_slots[slot] = prediction_row(parent, view, detector.name, logit, probability, checkpoint_sha, detector.preprocessing_id, embedding_dim)
            embeddings[slot] = emb[offset].numpy().astype(np.float32)
        pending_images = []
        pending_meta = []

    for parent_index, parent in enumerate(parent_records):
        base_slot = parent_index * len(VIEW_ORDER)
        with Image.open(parent["source_path"]) as handle:
            source = handle.convert("RGB")
            for view_offset, view in enumerate(views[1:], start=1):
                pending_images.append(apply_view(source, view.view_name))
                pending_meta.append((base_slot + view_offset, parent, view))
                if len(pending_images) >= batch_size:
                    flush()
    flush()
    if any(row is None for row in row_slots):
        raise RuntimeError("internal error: incomplete prediction row slots")
    return pd.DataFrame(row_slots), embeddings


class NonIdentityOrbitTensorDataset(torch.utils.data.Dataset):
    def __init__(self, parent_records: list[dict[str, Any]], views: tuple[Any, ...], transform: Any):
        self.parent_records = parent_records
        self.views = views
        self.transform = transform

    def __len__(self) -> int:
        return len(self.parent_records)

    def __getitem__(self, parent_index: int) -> dict[str, Any]:
        parent = self.parent_records[parent_index]
        tensors = []
        with Image.open(parent["source_path"]) as handle:
            source = handle.convert("RGB")
            for view in self.views[1:]:
                tensors.append(self.transform(apply_view(source, view.view_name)))
        return {
            "parent_index": int(parent_index),
            "tensors": torch.stack(tensors, dim=0),
            "view_offsets": torch.arange(1, len(self.views), dtype=torch.int64),
        }


@torch.inference_mode()
def detector_predict_tensor_batch(detector: Any, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = batch.to(detector.device, non_blocking=True)
    if detector.name == "safe":
        logits, embeddings = detector._features_and_logits(tensor)
    elif detector.name == "univfd":
        embeddings = detector.model.model.encode_image(tensor).float()
        logits = detector.model.fc(embeddings).flatten().float()
    else:
        raise ValueError(detector.name)
    return logits.detach().cpu(), embeddings.detach().cpu()


def infer_shard_with_tensor_loader(
    parent_records: list[dict[str, Any]],
    row_slots: list[dict[str, Any] | None],
    embeddings: np.ndarray,
    detector: Any,
    views: tuple[Any, ...],
    checkpoint_sha: str,
    embedding_dim: int,
    batch_size: int,
    image_workers: int,
    device: str,
) -> None:
    dataset = NonIdentityOrbitTensorDataset(parent_records, views, detector.transform)
    parent_batch_size = max(1, int(batch_size) // max(1, len(views) - 1))
    loader_kwargs: dict[str, Any] = {
        "batch_size": parent_batch_size,
        "shuffle": False,
        "num_workers": int(image_workers),
        "pin_memory": bool(device.startswith("cuda") and torch.cuda.is_available()),
    }
    if image_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    non_identity_views = len(views) - 1
    for batch in loader:
        tensor_batch = batch["tensors"].reshape(-1, *batch["tensors"].shape[2:])
        parent_indexes = batch["parent_index"].repeat_interleave(non_identity_views).numpy()
        view_offsets = batch["view_offsets"].reshape(-1).numpy()
        logits, emb = detector_predict_tensor_batch(detector, tensor_batch)
        probs = torch.sigmoid(logits)
        for offset, (parent_index, view_offset) in enumerate(zip(parent_indexes, view_offsets, strict=True)):
            parent = parent_records[int(parent_index)]
            view = views[int(view_offset)]
            slot = int(parent_index) * len(VIEW_ORDER) + int(view_offset)
            probability = float(probs[offset].item())
            logit = float(logits[offset].item())
            row_slots[slot] = prediction_row(
                parent,
                view,
                detector.name,
                logit,
                probability,
                checkpoint_sha,
                detector.preprocessing_id,
                embedding_dim,
            )
            embeddings[slot] = emb[offset].numpy().astype(np.float32)


def prediction_row(
    parent: dict[str, Any],
    view: Any,
    detector: str,
    raw_logit: float,
    fake_probability: float,
    checkpoint_sha: str,
    preprocessing_id: str,
    embedding_dim: int,
    predicted_label: int | None = None,
) -> dict[str, Any]:
    chain_id = transform_chain_id(view)
    view_id = make_view_id(str(parent["parent_sample_id"]), str(parent["sha256"]), chain_id)
    return {
        "parent_sample_id": parent["parent_sample_id"],
        "source_sample_id": parent["source_sample_id"],
        "view_id": view_id,
        "view_name": view.view_name,
        "view_index": VIEW_ORDER.index(view.view_name),
        "detector": detector,
        "raw_logit": raw_logit,
        "fake_probability": fake_probability,
        "predicted_label": int(fake_probability >= 0.5) if predicted_label is None else int(predicted_label),
        "embedding_dimension": int(embedding_dim),
        "checkpoint_sha256": checkpoint_sha,
        "preprocessing_id": preprocessing_id,
        "split": parent["split"],
        "partition": parent["partition"],
        "evaluation_role": parent["evaluation_role"],
        "generator": parent["generator"],
        "label": int(parent["label"]),
        "sha256": parent["sha256"],
        "source_id": parent.get("source_id", ""),
        "near_duplicate_group": parent.get("near_duplicate_group", ""),
        "near_duplicate_group_source": parent.get("near_duplicate_group_source", ""),
        "orbit_version": ORBIT_VERSION,
    }


def assert_finite_outputs(rows: pd.DataFrame, embeddings: np.ndarray) -> None:
    if not np.isfinite(rows["raw_logit"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Phase 4 logits contain NaN/Inf")
    if not np.isfinite(rows["fake_probability"].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Phase 4 probabilities contain NaN/Inf")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Phase 4 embeddings contain NaN/Inf")


def write_cache_index(out_dir: Path) -> None:
    rows = []
    for pred_path in sorted(out_dir.glob("predictions_*.parquet")):
        shard_id = int(pred_path.stem.split("_")[-1])
        emb_path = out_dir / f"embeddings_{shard_id:05d}.npy"
        pred = pd.read_parquet(pred_path, columns=["parent_sample_id", "view_id"])
        for offset, row in enumerate(pred.to_dict("records")):
            rows.append(
                {
                    "parent_sample_id": row["parent_sample_id"],
                    "view_id": row["view_id"],
                    "prediction_shard": str(pred_path),
                    "embedding_shard": str(emb_path),
                    "row_offset": offset,
                }
            )
    pd.DataFrame(rows).to_parquet(out_dir / "index.parquet", index=False)


def load_orbit_cache(detector: str, scope: str) -> tuple[pd.DataFrame, np.ndarray]:
    cache_dir = scope_dir(scope) / "orbit_cache" / detector
    frames = []
    arrays = []
    for pred_path in sorted(cache_dir.glob("predictions_*.parquet")):
        shard_id = int(pred_path.stem.split("_")[-1])
        emb_path = cache_dir / f"embeddings_{shard_id:05d}.npy"
        frames.append(pd.read_parquet(pred_path))
        arrays.append(np.load(emb_path))
    if not frames:
        raise RuntimeError(f"no Phase 4 cache shards found for {detector}/{scope}")
    return pd.concat(frames, ignore_index=True), np.vstack(arrays).astype(np.float32)


def features(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    verify_frozen_inputs()
    out_dir = scope_dir(args.scope)
    parent = pd.read_parquet(out_dir / "parent_context_manifest.parquet")
    orbit_manifest = pd.read_parquet(out_dir / "transformation_orbit_manifest.parquet")
    pred, embeddings = load_orbit_cache(args.detector, args.scope)
    if len(pred) != embeddings.shape[0]:
        raise RuntimeError("prediction/embedding row mismatch")
    pred = pred.merge(
        orbit_manifest[["parent_sample_id", "view_id", "transformed_pixel_sha256"]],
        on=["parent_sample_id", "view_id"],
        how="left",
        validate="one_to_one",
    )
    if pred["transformed_pixel_sha256"].isna().any():
        raise RuntimeError("orbit cache to manifest join is incomplete")
    pred["_row"] = np.arange(len(pred))
    support = compute_support_distances(args.detector, pred, embeddings, args.device, args.support_batch_size)
    pred["view_support_distance"] = support["distance"]
    upsert_csv(out_dir / "support_distance_crossfit_audit.csv", support["audit"], ["detector", "split"])
    upsert_csv(out_dir / "support_distance_fit_registry.csv", support["registry"], ["detector", "split"])
    feature_rows = build_feature_rows(args.detector, pred, embeddings, support["bank_sha"])
    feature_root = out_dir / "features" / args.detector
    write_feature_artifacts(feature_rows, feature_root)
    write_feature_diagnostics(args.detector, feature_rows, pred, out_dir)
    identity_parity(args.detector, args.scope)
    status = {
        "detector": args.detector,
        "scope": args.scope,
        "parent_rows": int(len(parent)),
        "feature_rows": int(len(feature_rows)),
        "primary_features": list(PRIMARY_FEATURES),
        "seconds": time.perf_counter() - started,
        "created_at": now_local_iso(),
    }
    write_json(out_dir / f"feature_status_{args.detector}.json", status)
    print(json.dumps(status, indent=2))


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x.astype(np.float32), axis=1)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise RuntimeError("zero-norm or non-finite embedding in support distance")
    return x.astype(np.float32) / norms[:, None]


def compute_support_distances(
    detector: str,
    pred: pd.DataFrame,
    embeddings: np.ndarray,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    distances = np.empty(len(pred), dtype=np.float32)
    audit_rows = []
    registry_rows = []
    bank_sha: dict[str, str] = {}
    for split in SPLITS:
        selected_path = PHASE3 / "fits" / detector / split / "knn_selected_k.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_k = int(selected["selected_k"])
        bank_manifest = pd.read_parquet(PHASE3 / "fits" / detector / split / "knn_reference_bank.parquet")
        bank_manifest["sample_id"] = bank_manifest["sample_id"].astype(str)
        phase2 = Phase2Cache(PROJECT_ROOT, detector)
        bank_embeddings = phase2.embeddings_for(bank_manifest["sample_id"].tolist())
        bank_norm = normalize_matrix(bank_embeddings)
        bank_ids = bank_manifest["sample_id"].astype(str).to_numpy()
        bank_shas = bank_manifest["sha256"].astype(str).to_numpy()
        bank_sha[split] = selected.get("reference_bank_sha256", sha256_file(PHASE3 / "fits" / detector / split / "knn_reference_bank.parquet"))
        split_mask = pred["split"].astype(str).eq(split).to_numpy()
        query_idx = np.flatnonzero(split_mask)
        query_norm = normalize_matrix(embeddings[query_idx])
        query_ids = pred.iloc[query_idx]["source_sample_id"].astype(str).to_numpy()
        query_shas = pred.iloc[query_idx]["sha256"].astype(str).to_numpy()
        query_parent = pred.iloc[query_idx]["parent_sample_id"].astype(str).to_numpy()
        bank_folds = np.asarray([deterministic_uint(f"{split}:risk_fit:{sample_id}") % 5 for sample_id in bank_ids], dtype=np.int16)
        query_partitions = pred.iloc[query_idx]["evaluation_role"].astype(str).to_numpy()
        query_folds = np.asarray(
            [
                deterministic_uint(f"{split}:risk_fit:{sample_id}") % 5 if role == "risk_fit" else -1
                for sample_id, role in zip(query_ids, query_partitions, strict=True)
            ],
            dtype=np.int16,
        )
        result, audit = support_query(
            bank_norm,
            bank_ids,
            bank_shas,
            bank_folds,
            query_norm,
            query_ids,
            query_shas,
            query_parent,
            query_folds,
            selected_k,
            device,
            batch_size,
        )
        distances[query_idx] = result
        audit.update({"detector": detector, "split": split, "selected_k": selected_k, "query_rows": int(len(query_idx))})
        audit["crossfit_folds"] = 5
        audit_rows.append(audit)
        registry_rows.append(
            {
                "detector": detector,
                "split": split,
                "source_partition": "risk_fit",
                "reference_bank": selected["reference_bank"],
                "reference_bank_sha256": bank_sha[split],
                "selected_k": selected_k,
                "phase3_selected_k_artifact": rel(selected_path),
                "phase3_selected_k_sha256": sha256_file(selected_path),
                "bfree_in_bank": False,
                "risk_fit_crossfit_folds": 5,
            }
        )
    if not np.isfinite(distances).all():
        raise RuntimeError("non-finite support distances")
    return {"distance": distances, "audit": pd.DataFrame(audit_rows), "registry": pd.DataFrame(registry_rows), "bank_sha": bank_sha}


def support_query(
    bank_norm: np.ndarray,
    bank_ids: np.ndarray,
    bank_shas: np.ndarray,
    bank_folds: np.ndarray,
    query_norm: np.ndarray,
    query_ids: np.ndarray,
    query_shas: np.ndarray,
    query_parent_ids: np.ndarray,
    query_folds: np.ndarray,
    k: int,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    id_lookup: dict[str, list[int]] = {}
    sha_lookup: dict[str, list[int]] = {}
    for index, value in enumerate(bank_ids):
        id_lookup.setdefault(str(value), []).append(index)
    for index, value in enumerate(bank_shas):
        sha_lookup.setdefault(str(value), []).append(index)
    bank_all = torch.as_tensor(bank_norm, dtype=torch.float32, device=torch_device)
    allowed_indices_by_fold: dict[int, np.ndarray] = {-1: np.arange(len(bank_ids), dtype=np.int64)}
    bank_t_by_fold: dict[int, torch.Tensor] = {-1: bank_all.T.contiguous()}
    original_to_allowed_by_fold: dict[int, np.ndarray] = {}
    for fold in range(5):
        allowed = np.flatnonzero(bank_folds != fold).astype(np.int64)
        allowed_indices_by_fold[fold] = allowed
        bank_t_by_fold[fold] = bank_all[torch.as_tensor(allowed, dtype=torch.long, device=torch_device)].T.contiguous()
    for fold, allowed in allowed_indices_by_fold.items():
        lookup = np.full(len(bank_ids), -1, dtype=np.int64)
        lookup[allowed] = np.arange(len(allowed), dtype=np.int64)
        original_to_allowed_by_fold[fold] = lookup
    out = np.empty(query_norm.shape[0], dtype=np.float32)
    self_neighbor_count = 0
    same_sha_neighbor_count = 0
    same_fold_neighbor_count = 0
    invalid_neighbor_count = 0
    neighbor_digest = hashlib.sha256()
    for start in range(0, query_norm.shape[0], batch_size):
        end = min(start + batch_size, query_norm.shape[0])
        batch_len = end - start
        values_np = np.empty((batch_len, int(k)), dtype=np.float32)
        indexes_np = np.empty((batch_len, int(k)), dtype=np.int64)
        batch_folds = query_folds[start:end]
        for fold in sorted(int(value) for value in np.unique(batch_folds)):
            local_rows = np.flatnonzero(batch_folds == fold)
            allowed = allowed_indices_by_fold[fold]
            if len(allowed) < k:
                values_np[local_rows] = -float("inf")
                indexes_np[local_rows] = 0
                continue
            q = torch.as_tensor(query_norm[start:end][local_rows], dtype=torch.float32, device=torch_device)
            sims = q @ bank_t_by_fold[fold]
            original_to_allowed = original_to_allowed_by_fold[fold]
            for sub_local, local in enumerate(local_rows):
                global_idx = start + int(local)
                blocked = id_lookup.get(str(query_ids[global_idx]), []) + sha_lookup.get(str(query_shas[global_idx]), [])
                if blocked:
                    blocked_original = np.asarray(sorted(set(blocked)), dtype=np.int64)
                    blocked_allowed = original_to_allowed[blocked_original]
                    blocked_allowed = blocked_allowed[blocked_allowed >= 0]
                    if len(blocked_allowed):
                        sims[sub_local, torch.as_tensor(blocked_allowed, dtype=torch.long, device=torch_device)] = -float("inf")
            values, indexes = torch.topk(sims, k=int(k), dim=1, largest=True, sorted=True)
            values_np[local_rows] = values.detach().cpu().numpy()
            indexes_np[local_rows] = allowed[indexes.detach().cpu().numpy()]
            del q, sims, values, indexes
        for local, global_idx in enumerate(range(start, end)):
            idx = indexes_np[local]
            if len(idx) < k or np.isneginf(values_np[local]).any():
                invalid_neighbor_count += 1
            self_neighbor_count += int(np.sum(bank_ids[idx] == query_ids[global_idx]))
            same_sha_neighbor_count += int(np.sum(bank_shas[idx] == query_shas[global_idx]))
            if query_folds[global_idx] >= 0:
                same_fold_neighbor_count += int(np.sum(bank_folds[idx] == query_folds[global_idx]))
            neighbor_digest.update(str(query_parent_ids[global_idx]).encode("utf-8"))
            neighbor_digest.update(",".join(bank_ids[idx]).encode("utf-8"))
        out[start:end] = (1.0 - values_np).mean(axis=1).astype(np.float32)
    return out, {
        "self_neighbor_count": int(self_neighbor_count),
        "same_sha_neighbor_count": int(same_sha_neighbor_count),
        "same_fold_neighbor_count": int(same_fold_neighbor_count),
        "invalid_neighbor_count": int(invalid_neighbor_count),
        "support_neighbor_ids_sha256": neighbor_digest.hexdigest(),
    }


def build_feature_rows(detector: str, pred: pd.DataFrame, embeddings: np.ndarray, support_bank_sha: dict[str, str]) -> pd.DataFrame:
    thresholds = base_thresholds(PROJECT_ROOT)
    orbit_config_sha = sha256_file(ORBIT_CONFIG)
    rows = []
    pred = pred.sort_values(["parent_sample_id", "view_index"], kind="mergesort").reset_index(drop=True)
    for parent_id, group in pred.groupby("parent_sample_id", sort=False):
        if list(group["view_name"]) != list(VIEW_ORDER):
            raise RuntimeError(f"missing or out-of-order views for {parent_id}")
        row_indexes = group["_row"].to_numpy(dtype=np.int64)
        emb = embeddings[row_indexes]
        logits = group["raw_logit"].to_numpy(dtype=np.float64)
        support_dist = group["view_support_distance"].to_numpy(dtype=np.float64)
        identity = group.iloc[0]
        threshold_probability = float(
            thresholds[(thresholds["detector"] == detector) & (thresholds["split"] == identity["split"])]["decision_threshold"].iloc[0]
        )
        threshold_logit = logit_threshold(threshold_probability)
        base_probability = float(identity["fake_probability"])
        base_prediction = int(base_probability >= threshold_probability)
        feature_payload = {
            "margin_distance": margin_distance(float(identity["raw_logit"]), threshold_logit),
            "orbit_logit_variance": orbit_logit_variance(logits),
            "embedding_drift_mean": embedding_drift_mean(emb),
            "orbit_support_distance_max": orbit_support_distance_max(support_dist),
        }
        if tuple(feature_payload) != PRIMARY_FEATURES:
            raise RuntimeError("primary feature schema drift")
        rows.append(
            {
                "sample_id": identity["source_sample_id"],
                "sha256": identity["sha256"],
                "detector": detector,
                "split": identity["split"],
                "partition": identity["partition"],
                "evaluation_role": identity["evaluation_role"],
                "generator": identity["generator"],
                "label": int(identity["label"]),
                "base_logit": float(identity["raw_logit"]),
                "base_probability": base_probability,
                "base_prediction": base_prediction,
                "base_error": int(base_prediction != int(identity["label"])),
                **feature_payload,
                "orbit_version": ORBIT_VERSION,
                "orbit_config_sha256": orbit_config_sha,
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "support_bank_sha256": support_bank_sha[str(identity["split"])],
                "selected_k": int(json.loads((PHASE3 / "fits" / detector / str(identity["split"]) / "knn_selected_k.json").read_text())["selected_k"]),
                "parent_sample_id": parent_id,
                "source_id": identity.get("source_id", ""),
                "near_duplicate_group": identity.get("near_duplicate_group", ""),
                "near_duplicate_group_source": identity.get("near_duplicate_group_source", ""),
            }
        )
    features = pd.DataFrame(rows)
    if not np.isfinite(features[list(PRIMARY_FEATURES)].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("non-finite Phase 4 feature value")
    return features


def write_feature_artifacts(features: pd.DataFrame, feature_root: Path) -> None:
    feature_root.mkdir(parents=True, exist_ok=True)
    name_map = {
        "risk_fit": "risk_fit",
        "threshold_cal": "threshold_cal",
        "protocol_seen": "protocol_seen",
        "protocol_held_out": "protocol_held_out",
        "B-Free Viral Verified Snapshot": "bfree_snapshot",
    }
    required = [
        "sample_id",
        "sha256",
        "detector",
        "split",
        "partition",
        "evaluation_role",
        "generator",
        "label",
        "base_logit",
        "base_probability",
        "base_prediction",
        "base_error",
        *PRIMARY_FEATURES,
        "orbit_version",
        "orbit_config_sha256",
        "checkpoint_sha256",
        "support_bank_sha256",
        "selected_k",
    ]
    extra = [col for col in ["parent_sample_id", "source_id", "near_duplicate_group", "near_duplicate_group_source"] if col in features.columns]
    for split in SPLITS:
        split_dir = feature_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_df = features[features["split"].astype(str).eq(split)]
        for role, name in name_map.items():
            part = split_df[split_df["evaluation_role"].astype(str).eq(role)].copy()
            if len(part):
                part[required + extra].to_parquet(split_dir / f"{name}.parquet", index=False)


def summarize_feature_frame(df: pd.DataFrame, detector: str, scope_name: str) -> pd.DataFrame:
    rows = []
    for feature in PRIMARY_FEATURES:
        values = df[feature].to_numpy(dtype=np.float64)
        q75, q25 = np.percentile(values, [75, 25]) if len(values) else (np.nan, np.nan)
        auroc = np.nan
        aupr = np.nan
        if len(df) and df["base_error"].nunique() == 2:
            auroc = float(roc_auc_score(df["base_error"].astype(int), values))
            aupr = float(average_precision_score(df["base_error"].astype(int), values))
        rows.append(
            {
                "detector": detector,
                "scope": scope_name,
                "feature": feature,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=0)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "IQR": float(q75 - q25) if len(values) else np.nan,
                "minimum": float(np.min(values)) if len(values) else np.nan,
                "maximum": float(np.max(values)) if len(values) else np.nan,
                "error_detection_AUROC": auroc,
                "error_detection_AUPR": aupr,
            }
        )
    return pd.DataFrame(rows)


def write_feature_diagnostics(detector: str, features: pd.DataFrame, pred: pd.DataFrame, out_dir: Path) -> None:
    risk = features[features["evaluation_role"].eq("risk_fit")]
    cal = features[features["evaluation_role"].eq("threshold_cal")]
    upsert_csv(
        out_dir / "feature_diagnostics_risk_fit_oof.csv",
        summarize_feature_frame(risk, detector, "risk_fit_oof"),
        ["detector", "scope", "feature"],
    )
    upsert_csv(
        out_dir / "feature_diagnostics_threshold_cal.csv",
        summarize_feature_frame(cal, detector, "threshold_cal"),
        ["detector", "scope", "feature"],
    )
    test = features[features["evaluation_role"].isin(["protocol_seen", "protocol_held_out", "B-Free Viral Verified Snapshot"])]
    dist_rows = []
    for role, role_df in test.groupby("evaluation_role"):
        for feature in PRIMARY_FEATURES:
            vals = role_df[feature].to_numpy(dtype=np.float64)
            dist_rows.append(
                {
                    "detector": detector,
                    "evaluation_role": role,
                    "feature": feature,
                    "rows": int(len(role_df)),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "std": float(np.std(vals, ddof=0)) if len(vals) else np.nan,
                    "minimum": float(np.min(vals)) if len(vals) else np.nan,
                    "maximum": float(np.max(vals)) if len(vals) else np.nan,
                }
            )
    upsert_csv(out_dir / "test_feature_distribution_summary.csv", pd.DataFrame(dist_rows), ["detector", "evaluation_role", "feature"])
    corr_rows = []
    for left in PRIMARY_FEATURES:
        for right in PRIMARY_FEATURES:
            if left >= right:
                continue
            corr = spearmanr(features[left], features[right], nan_policy="omit")
            corr_rows.append({"detector": detector, "left_feature": left, "right_feature": right, "spearman": float(corr.statistic)})
    upsert_csv(out_dir / "feature_rank_correlations.csv", pd.DataFrame(corr_rows), ["detector", "left_feature", "right_feature"])
    flips = (
        pred.assign(identity_prediction=pred.groupby("parent_sample_id")["predicted_label"].transform("first"))
        .assign(prediction_flipped=lambda x: x["predicted_label"].astype(int) != x["identity_prediction"].astype(int))
        .groupby(["detector", "split", "evaluation_role", "view_name"], as_index=False)
        .agg(rows=("view_id", "count"), flip_count=("prediction_flipped", "sum"))
    )
    flips["flip_rate"] = flips["flip_count"] / flips["rows"]
    upsert_csv(out_dir / "orbit_prediction_flip_diagnostics.csv", flips, ["detector", "split", "evaluation_role", "view_name"])
    runtime = pd.DataFrame(
        [
            {
                "detector": detector,
                "stage": "features",
                "scope": out_dir.name if out_dir.name != "phase4" else "full",
                "rows": int(len(features)),
                "created_at": now_local_iso(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        ]
    )
    upsert_csv(out_dir / "runtime_resource_audit.csv", runtime, ["detector", "stage", "scope"])


def identity_parity(detector: str, scope: str) -> pd.DataFrame:
    out_dir = scope_dir(scope)
    pred, embeddings = load_orbit_cache(detector, scope)
    identity_rows = pred[pred["view_name"].eq("identity")].reset_index(drop=True)
    identity_emb = embeddings[pred["view_name"].eq("identity").to_numpy()]
    store = IdentityStore(detector)
    base_rows, base_emb = store.rows_and_embeddings(identity_rows["source_sample_id"].astype(str).tolist())
    rows = []
    for idx, row in identity_rows.iterrows():
        raw_diff = abs(float(row["raw_logit"]) - float(base_rows.iloc[idx]["raw_logit"]))
        prob_diff = abs(float(row["fake_probability"]) - float(base_rows.iloc[idx]["fake_probability"]))
        emb_diff = float(np.max(np.abs(identity_emb[idx].astype(np.float32) - base_emb[idx].astype(np.float32))))
        pred_match = int(row["predicted_label"]) == int(base_rows.iloc[idx]["predicted_label"])
        rows.append(
            {
                "detector": detector,
                "scope": scope,
                "parent_sample_id": row["parent_sample_id"],
                "sample_id": row["source_sample_id"],
                "predicted_label_match": pred_match,
                "raw_logit_abs_diff": raw_diff,
                "fake_probability_abs_diff": prob_diff,
                "embedding_max_abs_diff": emb_diff,
                "parity_pass": pred_match and raw_diff <= 1e-6 and prob_diff <= 1e-6 and emb_diff <= 1e-5,
            }
        )
    audit = pd.DataFrame(rows)
    upsert_csv(out_dir / "identity_parity_audit.csv", audit, ["detector", "parent_sample_id", "sample_id"])
    return audit


def freeze_phase4_outputs(summary: dict[str, Any]) -> None:
    policy = ensure_freeze_policy()
    rows = []
    excluded = {
        str((PHASE4 / "phase4_frozen_artifact_hashes.csv").resolve()),
        str((CONFIG_DIR / "phase4_frozen.yaml").resolve()),
    }
    for root_name in policy["include_roots"]:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            relative = rel(path)
            if not path.is_file() or str(path.resolve()) in excluded or freeze_excluded(relative, policy):
                continue
            rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    frozen = pd.DataFrame(rows)
    frozen.to_csv(PHASE4 / "phase4_frozen_artifact_hashes.csv", index=False)
    payload = {
        "created_at": now_local_iso(),
        "phase": "phase4_frozen",
        "pre_phase_5_status": summary["pre_phase_5_status"],
        "artifact_hash_registry": "artifacts/phase4/phase4_frozen_artifact_hashes.csv",
        "artifact_count_excluding_hash_registry": int(len(frozen)),
        "orbit_config": "configs/phase4/transformation_orbit.yaml",
        "orbit_config_sha256": sha256_file(ORBIT_CONFIG),
        "freeze_policy": "configs/phase4/freeze_policy.yaml",
        "freeze_policy_sha256": sha256_file(FREEZE_POLICY),
        "final_report": "reports/phase4/phase4_final_audit_report.md",
    }
    if yaml is not None:
        text = yaml.safe_dump(payload, sort_keys=False)
    else:
        text = "\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n"
    (CONFIG_DIR / "phase4_frozen.yaml").write_text(text, encoding="utf-8")


def audit(args: argparse.Namespace) -> None:
    out_dir = scope_dir(args.scope)
    REPORTS.mkdir(parents=True, exist_ok=True)
    checks = []

    def add(check: str, ok: bool, hard: bool, detail: str = "") -> None:
        checks.append({"check": check, "status": "pass" if ok else "fail", "hard_blocker": hard, "detail": detail})

    try:
        frozen = verify_frozen_inputs()
        add("Frozen Phase 2 and Phase 3 hashes unchanged", True, True, json.dumps(frozen))
    except Exception as exc:
        add("Frozen Phase 2 and Phase 3 hashes unchanged", False, True, str(exc))
    config_ok = ORBIT_CONFIG.exists() and sha256_file(ORBIT_CONFIG)
    add("Five-view orbit configuration exists", bool(config_ok), True, rel(ORBIT_CONFIG))
    parent_path = out_dir / "parent_context_manifest.parquet"
    orbit_path = out_dir / "transformation_orbit_manifest.parquet"
    if parent_path.exists() and orbit_path.exists():
        parent = pd.read_parquet(parent_path)
        orbit = pd.read_parquet(orbit_path)
        expected_views = len(parent) * len(VIEW_ORDER)
        complete = orbit.groupby("parent_sample_id")["view_name"].nunique().eq(len(VIEW_ORDER)).all()
        dup = int(orbit.duplicated(["parent_sample_id", "view_name"]).sum())
        crossing = int(orbit.groupby("view_id")[["split", "partition", "evaluation_role"]].nunique().max(axis=1).gt(1).sum())
        add("Every parent has exactly five views", len(orbit) == expected_views and bool(complete), True, f"parents={len(parent)} views={len(orbit)} expected={expected_views}")
        add("No duplicate parent/view rows", dup == 0, True, f"duplicates={dup}")
        add("No view crosses split/partition/evaluation_role", crossing == 0, True, f"crossing_view_ids={crossing}")
    else:
        parent = pd.DataFrame()
        orbit = pd.DataFrame()
        add("Parent and orbit manifests exist", False, True, f"{rel(parent_path)} {rel(orbit_path)}")
    expected_by_split_role = {}
    if len(parent):
        expected_by_split_role = parent.groupby(["split", "evaluation_role"]).size().to_dict()
    for detector in DETECTORS:
        cache_dir = out_dir / "orbit_cache" / detector
        if cache_dir.exists() and list(cache_dir.glob("predictions_*.parquet")):
            pred, emb = load_orbit_cache(detector, args.scope)
            finite = np.isfinite(pred["raw_logit"].to_numpy(float)).all() and np.isfinite(pred["fake_probability"].to_numpy(float)).all() and np.isfinite(emb).all()
            add(f"{detector} orbit cache finite", bool(finite), True, f"rows={len(pred)} emb_shape={emb.shape}")
            parity_path = out_dir / "identity_parity_audit.csv"
            if parity_path.exists():
                parity = pd.read_csv(parity_path)
                pass_rate = float(parity["parity_pass"].mean()) if len(parity) else 0.0
                add(f"{detector} identity parity 100%", pass_rate == 1.0, True, f"pass_rate={pass_rate:.6f}")
        else:
            add(f"{detector} orbit cache exists", False, True, rel(cache_dir))
        feature_root = out_dir / "features" / detector
        feature_files = sorted(feature_root.glob("*/*.parquet"))
        if feature_files:
            frames = [pd.read_parquet(path) for path in feature_files]
            feat = pd.concat(frames, ignore_index=True)
            exact_features = all(col in feat.columns for col in PRIMARY_FEATURES)
            finite_features = np.isfinite(feat[list(PRIMARY_FEATURES)].to_numpy(float)).all()
            add(f"{detector} four primary features materialized", exact_features and finite_features, True, f"rows={len(feat)} files={len(feature_files)}")
            add(f"{detector} feature non-degeneracy", bool((feat[list(PRIMARY_FEATURES)].std(ddof=0) > 0).all()), False)
            expected_total = int(len(parent)) if len(parent) else 0
            add(f"{detector} feature row total matches parent inventory", len(feat) == expected_total, True, f"features={len(feat)} parents={expected_total}")
            file_rows_ok = True
            missing_files = []
            role_to_name = {
                "risk_fit": "risk_fit",
                "threshold_cal": "threshold_cal",
                "protocol_seen": "protocol_seen",
                "protocol_held_out": "protocol_held_out",
                "B-Free Viral Verified Snapshot": "bfree_snapshot",
            }
            for (split, role), expected_rows in expected_by_split_role.items():
                path = feature_root / str(split) / f"{role_to_name[str(role)]}.parquet"
                if not path.exists():
                    file_rows_ok = False
                    missing_files.append(rel(path))
                    continue
                observed_rows = len(pd.read_parquet(path, columns=["sample_id"]))
                if observed_rows != int(expected_rows):
                    file_rows_ok = False
                    missing_files.append(f"{rel(path)} rows={observed_rows} expected={expected_rows}")
            add(f"{detector} feature rows match every split/partition file", file_rows_ok, True, "; ".join(missing_files[:5]))
            bfree = feat[feat["evaluation_role"].astype(str).eq("B-Free Viral Verified Snapshot")]
            if len(bfree):
                bfree_ok = (
                    bfree["source_id"].astype(str).ne("").all()
                    and bfree["near_duplicate_group_source"].astype(str).eq("derived_from_phash_audit").all()
                )
                add(f"{detector} B-Free source_id and near_duplicate_group_source preserved", bfree_ok, True, f"rows={len(bfree)}")
        else:
            add(f"{detector} feature artifacts exist", False, True, rel(feature_root))
    support_path = out_dir / "support_distance_crossfit_audit.csv"
    if support_path.exists():
        support = pd.read_csv(support_path)
        ok = bool(
            (
                support["self_neighbor_count"].astype(int).eq(0)
                & support["same_sha_neighbor_count"].astype(int).eq(0)
                & support.get("same_fold_neighbor_count", pd.Series([0] * len(support))).astype(int).eq(0)
                & support["invalid_neighbor_count"].astype(int).eq(0)
            ).all()
        )
        add("Support distance has no self, same-SHA, same-fold, or invalid neighbors", ok, True, f"rows={len(support)}")
    else:
        add("Support distance audit exists", False, True, rel(support_path))
    determinism_path = out_dir / "determinism_audit.csv"
    if determinism_path.exists():
        det = pd.read_csv(determinism_path)
        det_ok = bool(det[["view_id_match", "pixel_hash_match"]].all().all()) if len(det) else False
        add("Determinism audit passed", det_ok, True, f"rows={len(det)}")
    else:
        add("Determinism audit exists", False, True, rel(determinism_path))
    tests_path = out_dir / "test_suite_status.json"
    if tests_path.exists():
        tests = json.loads(tests_path.read_text(encoding="utf-8"))
        add("Full Phase 4 test suite passed", bool(tests.get("passed")), True, tests.get("summary", ""))
    else:
        add("Full Phase 4 test suite status exists", args.scope != "full", True, rel(tests_path))
    status = "PASS" if args.scope == "full" and all(row["status"] == "pass" or not row["hard_blocker"] for row in checks) else "FAIL"
    checklist = pd.DataFrame(checks)
    checklist_path = out_dir / "phase4_final_audit_checklist.csv"
    checklist.to_csv(checklist_path, index=False)
    summary = {
        "created_at": now_local_iso(),
        "scope": args.scope,
        "pre_phase_5_status": status,
        "checks": int(len(checklist)),
        "failed_hard_blockers": int(((checklist["status"] == "fail") & checklist["hard_blocker"]).sum()),
        "note": "Pilot audits intentionally report FAIL for PRE_PHASE_5_STATUS until the full orbit and both detectors are complete.",
    }
    write_json(out_dir / "phase4_final_audit_summary.json", summary)
    write_reports(args.scope, checklist, summary)
    if status == "PASS":
        freeze_phase4_outputs(summary)
    print(json.dumps(summary, indent=2))


def write_reports(scope: str, checklist: pd.DataFrame, summary: dict[str, Any]) -> None:
    out_dir = scope_dir(scope)
    report_dir = REPORTS if scope == "full" else REPORTS / scope
    report_dir.mkdir(parents=True, exist_ok=True)
    status = summary["pre_phase_5_status"]
    failed = checklist[checklist["status"].eq("fail")]
    process_lines = [
        "# Phase 4 Transformation Orbit Report",
        "",
        f"Created at: {now_local_iso()}",
        f"Scope: `{scope}`",
        "",
        "## Process",
        "",
        "- Verified frozen Phase 2 and Phase 3 hash registries before Phase 4 execution.",
        "- Wrote the frozen five-view orbit configuration under `configs/phase4/`.",
        "- Materialized split/partition-aware parent contexts to prevent cross-partition lineage leakage.",
        "- Used `CUDA_VISIBLE_DEVICES=1` for GPU-bound inference/support-distance work.",
        "- Reused frozen Phase 2 identity-view outputs for exact identity parity and inferred the four non-identity orbit views.",
        "",
        "## Artifacts",
        "",
        f"- Parent contexts: `{rel(out_dir / 'parent_context_manifest.parquet')}`",
        f"- Orbit manifest: `{rel(out_dir / 'transformation_orbit_manifest.parquet')}`",
        f"- Orbit cache root: `{rel(out_dir / 'orbit_cache')}`",
        f"- Feature root: `{rel(out_dir / 'features')}`",
        "",
        "## Anomalies",
        "",
        "- B-Free has `source_id` in the verified manifest but no original row-level `near_duplicate_group`; Phase 4 derives `near_duplicate_group` from the available pHash audit when possible.",
    ]
    if scope != "full":
        process_lines.append("- Full-orbit detector caches/features are not complete in this pilot scope, so the final Phase 5 gate remains FAIL.")
    process_lines.extend(["", f"PRE_PHASE_5_STATUS = {status}"])
    (report_dir / "phase4_transformation_orbit_report.md").write_text("\n".join(process_lines) + "\n", encoding="utf-8")

    audit_lines = [
        "# Phase 4 Final Audit Report",
        "",
        f"Scope: `{scope}`",
        "",
        "## Failed Checks",
        "",
    ]
    if len(failed):
        for row in failed.to_dict("records"):
            audit_lines.append(f"- {row['check']}: {row['detail']}")
    else:
        audit_lines.append("- None.")
    audit_lines.extend(["", "## Summary", "", checklist.to_markdown(index=False), "", f"PRE_PHASE_5_STATUS = {status}"])
    (report_dir / "phase4_final_audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")


def determinism(args: argparse.Namespace) -> None:
    out_dir = scope_dir(args.scope)
    orbit = pd.read_parquet(out_dir / "transformation_orbit_manifest.parquet").head(args.parents * len(VIEW_ORDER))
    rerun_path = out_dir / "determinism_audit.csv"
    rows = []
    for parent_id, group in orbit.groupby("parent_sample_id", sort=False):
        source_path = group["source_path"].iloc[0]
        parent_sha = group["parent_sha256"].iloc[0]
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
            for _, row in group.iterrows():
                view = next(view for view in default_orbit_views() if view.view_name == row["view_name"])
                transformed = apply_view(source, view.view_name)
                chain = transform_chain_id(view)
                rows.append(
                    {
                        "parent_sample_id": parent_id,
                        "view_name": row["view_name"],
                        "view_id_match": make_view_id(parent_id, parent_sha, chain) == row["view_id"],
                        "pixel_hash_match": transformed_pixel_sha256(transformed) == row["transformed_pixel_sha256"],
                    }
                )
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(rerun_path, index=False)
    print(json.dumps({"rows": len(audit_df), "determinism_pass": bool(audit_df[["view_id_match", "pixel_hash_match"]].all().all())}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.add_argument("--pilot-size", type=int, default=8192)
    p.add_argument("--audit-parents", type=int, default=256)
    p.add_argument("--manifest-workers", type=int, default=1)
    p.set_defaults(func=prepare)

    p = sub.add_parser("infer")
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.add_argument("--detector", choices=DETECTORS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-workers", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=5000)
    p.set_defaults(func=infer)

    p = sub.add_parser("features")
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.add_argument("--detector", choices=DETECTORS, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--support-batch-size", type=int, default=512)
    p.set_defaults(func=features)

    p = sub.add_parser("determinism")
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.add_argument("--parents", type=int, default=2048)
    p.set_defaults(func=determinism)

    p = sub.add_parser("audit")
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.set_defaults(func=audit)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
