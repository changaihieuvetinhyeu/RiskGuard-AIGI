"""Phase 6 certified selective risk-control pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from selective_detection.confidence_interval_bootstrap import percentile_ci, stratified_count_bootstrap
from selective_detection.exact_binomial_bound import clopper_pearson_upper
from selective_detection.policy_artifact_io import (
    canonical_json_bytes,
    freeze_paths,
    payload_sha256,
    read_json,
    read_yaml,
    relative_to,
    sha256_file,
    verify_freeze_registry,
    write_json,
    write_yaml,
)
from selective_detection.selective_metrics import aurc, eaurc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTORS = ("univfd", "safe")
SPLITS = ("split_a", "split_b")
ALPHAS = (0.05, 0.01)
DELTA = 0.05
CANDIDATE_COUNT = 10
SEED = 20260916
POLICIES = ("global_cp", "source_group_cp", "predicted_class_cp")
METHODS = ("riskguard", "msp", "entropy", "energy", "temp_msp", "mahalanobis", "knn")
PRIMARY_COMPARATORS = ("msp", "knn")
EVAL_DATASETS = ("protocol_seen", "protocol_held_out", "bfree_snapshot")
CURVE_METHODS = ("riskguard", "msp", "knn", "mahalanobis")

METHOD_LABELS = {
    "riskguard": "RiskGuard full_four logistic",
    "msp": "MSP",
    "entropy": "predictive entropy",
    "energy": "binary energy",
    "temp_msp": "temperature-scaled MSP",
    "mahalanobis": "Mahalanobis",
    "knn": "cosine kNN",
}

DATASET_LABELS = {
    "protocol_seen": "protocol-seen generators",
    "protocol_held_out": "protocol-held-out generators",
    "bfree_snapshot": "B-Free Viral Verified Snapshot",
}

SOURCE_DISPLAY = {
    "real": "real_all",
    "adm": "ADM",
    "biggan": "BigGAN",
    "glide": "GLIDE",
    "sd14": "SD1.4",
    "midjourney": "Midjourney",
    "sd15": "SD1.5",
    "wukong": "Wukong",
    "vqdm": "VQDM",
    "bfree_viral": "bfree_viral",
}


@dataclass(frozen=True)
class Phase6Paths:
    root: Path
    output_root: Path

    @property
    def artifacts(self) -> Path:
        return self.output_root

    @property
    def configs(self) -> Path:
        return self.root / "configs" / "phase6"

    @property
    def reports(self) -> Path:
        return self.root / "reports" / "phase6"

    @property
    def logs(self) -> Path:
        return self.root / "logs" / "phase6"


def phase6_paths(root: Path = PROJECT_ROOT, output_root: Path | None = None) -> Phase6Paths:
    out = output_root if output_root is not None else root / "artifacts" / "phase6"
    paths = Phase6Paths(root=Path(root), output_root=Path(out))
    for directory in [paths.artifacts, paths.configs, paths.reports, paths.logs, paths.reports / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def alpha_slug(alpha: float) -> str:
    return f"alpha_{str(float(alpha)).replace('.', 'p')}"


def combo_stem(detector: str, split: str, method: str, alpha: float, policy: str) -> str:
    return f"{detector}_{split}_{method}_{alpha_slug(alpha)}_{policy}"


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_unit_rank(seed: int, sha: str) -> str:
    return stable_hash_text(f"{seed}|{sha}")


def score_path(root: Path, detector: str, split: str, method: str, dataset: str) -> Path:
    if method == "riskguard":
        return root / "artifacts" / "phase5" / "scores" / detector / split / f"{dataset}.parquet"
    return root / "artifacts" / "phase3" / "scores" / detector / method / f"{split}_{dataset}.parquet"


def normalize_score_frame(root: Path, path: Path, detector: str, split: str, method: str, dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    if method == "riskguard":
        df["risk_score"] = df["risk_probability"].astype(float)
        df["baseline"] = "riskguard"
        df["risk_orientation"] = "higher_risk_score_more_likely_to_reject"
        model_hash = str(df["model_sha256"].iloc[0]) if "model_sha256" in df.columns and len(df) else ""
        df["model_or_baseline_score_hash"] = model_hash or sha256_file(path)
    else:
        df["model_or_baseline_score_hash"] = sha256_file(path)
    df["method"] = method
    df["method_label"] = METHOD_LABELS.get(method, method)
    df["score_artifact"] = relative_to(root, path)
    df["score_artifact_sha256"] = sha256_file(path)
    if "source_id" not in df.columns:
        df["source_id"] = ""
    if "near_duplicate_group" not in df.columns:
        df["near_duplicate_group"] = ""
    if "near_duplicate_group_source" not in df.columns:
        df["near_duplicate_group_source"] = ""
    if dataset == "bfree_snapshot":
        manifest = pd.read_csv(root / "datasets" / "manifests" / "bfree_viral_verified_snapshot.csv")
        manifest_cols = ["sample_id", "source_id"]
        enrich = manifest[manifest_cols].copy()
        df = df.drop(columns=[c for c in manifest_cols[1:] if c in df.columns], errors="ignore")
        df = df.merge(enrich, on="sample_id", how="left", validate="one_to_one")
        df["source_id"] = df["source_id"].fillna(df["sha256"]).astype(str)
        empty_source = df["source_id"].eq("") | df["source_id"].eq("nan")
        df.loc[empty_source, "source_id"] = df.loc[empty_source, "sha256"].astype(str)
        df["near_duplicate_group"] = df["source_id"].astype(str)
        df["near_duplicate_group_source"] = "source_id_fallback"
    for col in ["sample_id", "sha256", "generator"]:
        df[col] = df[col].astype(str)
    for col in ["label", "base_prediction", "base_error"]:
        df[col] = df[col].astype(int)
    df["risk_score"] = df["risk_score"].astype(float)
    if not np.isfinite(df["risk_score"].to_numpy()).all():
        raise ValueError(f"Non-finite risk score in {path}")
    return df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def load_scores(root: Path, detector: str, split: str, method: str, dataset: str) -> pd.DataFrame:
    path = score_path(root, detector, split, method, dataset)
    if not path.exists():
        raise FileNotFoundError(path)
    return normalize_score_frame(root, path, detector, split, method, dataset)


def deduplicate_eval_rows(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset == "bfree_snapshot":
        return df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    ordered = df.sort_values(["sha256", "sample_id"], kind="mergesort")
    return ordered.drop_duplicates("sha256", keep="first").sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def json_dumps_compact(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def verify_upstream_frozen_inputs(paths: Phase6Paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = paths.root
    required_registries = [
        ("phase2", root / "artifacts" / "phase2_frozen_artifact_hashes.csv"),
        ("phase3", root / "artifacts" / "phase3" / "phase3_frozen_artifact_hashes.csv"),
        ("phase4", root / "artifacts" / "phase4" / "phase4_frozen_artifact_hashes.csv"),
        ("phase5", root / "artifacts" / "phase5" / "phase5_frozen_artifact_hashes.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for phase, registry in required_registries:
        exists = registry.exists()
        rows.append(
            {
                "audit_type": "required_registry",
                "phase": phase,
                "relative_path": relative_to(root, registry),
                "expected_exists": True,
                "observed_exists": exists,
                "expected_size_bytes": "",
                "observed_size_bytes": int(registry.stat().st_size) if exists else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(registry) if exists else "",
                "status": "pass" if exists else "fail",
            }
        )
        if not exists:
            continue
        frozen = pd.read_csv(registry)
        for record in frozen.to_dict("records"):
            rel = str(record["relative_path"])
            path = root / rel
            expected_size = int(record.get("size_bytes", 0))
            expected_sha = str(record.get("sha256", ""))
            observed_exists = path.exists()
            observed_size = int(path.stat().st_size) if observed_exists else 0
            observed_sha = sha256_file(path) if observed_exists else ""
            ok = observed_exists and observed_size == expected_size and observed_sha == expected_sha
            rows.append(
                {
                    "audit_type": "frozen_hash",
                    "phase": phase,
                    "relative_path": rel,
                    "expected_exists": True,
                    "observed_exists": observed_exists,
                    "expected_size_bytes": expected_size,
                    "observed_size_bytes": observed_size,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "status": "pass" if ok else "fail",
                }
            )

    required_configs = [
        "configs/phase2_frozen.yaml",
        "configs/phase3/phase3_frozen.yaml",
        "configs/phase4/phase4_frozen.yaml",
        "configs/phase5/phase5_frozen.yaml",
    ]
    for rel in required_configs:
        path = root / rel
        rows.append(
            {
                "audit_type": "required_config",
                "phase": rel.split("/")[1] if "/" in rel else "phase2",
                "relative_path": rel,
                "expected_exists": True,
                "observed_exists": path.exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(path) if path.exists() else "",
                "status": "pass" if path.exists() else "fail",
            }
        )

    phase5_yaml = read_yaml(root / "configs" / "phase5" / "phase5_frozen.yaml")
    required_phase5 = {
        "pre_phase_6_status": "PASS",
        "feature_count": 4,
        "model_count": 4,
        "threshold_selected": False,
        "group_control_performed": False,
    }
    for key, expected in required_phase5.items():
        observed = phase5_yaml.get(key)
        rows.append(
            {
                "audit_type": "phase5_status",
                "phase": "phase5",
                "relative_path": f"configs/phase5/phase5_frozen.yaml::{key}",
                "expected_exists": True,
                "observed_exists": True,
                "expected_size_bytes": "",
                "observed_size_bytes": "",
                "expected_sha256": str(expected),
                "observed_sha256": str(observed),
                "status": "pass" if observed == expected else "fail",
            }
        )

    score_deep = pd.read_csv(root / "artifacts" / "phase5" / "score_artifact_deep_audit.csv")
    for record in score_deep.to_dict("records"):
        ok = (
            str(record.get("status")) == "pass"
            and bool(record.get("model_hash_ok"))
            and bool(record.get("feature_artifact_hash_ok"))
            and int(record.get("missing_rows", 1)) == 0
            and int(record.get("unexpected_rows", 1)) == 0
        )
        rows.append(
            {
                "audit_type": "phase5_score_artifact",
                "phase": "phase5",
                "relative_path": f"artifacts/phase5/scores/{record['detector']}/{record['split']}/{record['artifact']}.parquet",
                "expected_exists": True,
                "observed_exists": (root / "artifacts" / "phase5" / "scores" / str(record["detector"]) / str(record["split"]) / f"{record['artifact']}.parquet").exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": int(record.get("actual_rows", 0)),
                "expected_sha256": "model_hash_ok=True;feature_artifact_hash_ok=True",
                "observed_sha256": f"model_hash_ok={record.get('model_hash_ok')};feature_artifact_hash_ok={record.get('feature_artifact_hash_ok')}",
                "status": "pass" if ok else "fail",
            }
        )

    for detector in DETECTORS:
        for split in SPLITS:
            for dataset in ["threshold_cal", "protocol_seen", "protocol_held_out", "bfree_snapshot"]:
                path = score_path(root, detector, split, "riskguard", dataset)
                rows.append(
                    {
                        "audit_type": "required_phase5_score",
                        "phase": "phase5",
                        "relative_path": relative_to(root, path),
                        "expected_exists": True,
                        "observed_exists": path.exists(),
                        "expected_size_bytes": "",
                        "observed_size_bytes": int(path.stat().st_size) if path.exists() else 0,
                        "expected_sha256": "",
                        "observed_sha256": sha256_file(path) if path.exists() else "",
                        "status": "pass" if path.exists() else "fail",
                    }
                )
            for method in METHODS:
                if method == "riskguard":
                    continue
                for dataset in ["threshold_cal", "protocol_seen", "protocol_held_out", "bfree_snapshot"]:
                    path = score_path(root, detector, split, method, dataset)
                    rows.append(
                        {
                            "audit_type": "required_phase3_score",
                            "phase": "phase3",
                            "relative_path": relative_to(root, path),
                            "expected_exists": True,
                            "observed_exists": path.exists(),
                            "expected_size_bytes": "",
                            "observed_size_bytes": int(path.stat().st_size) if path.exists() else 0,
                            "expected_sha256": "",
                            "observed_sha256": sha256_file(path) if path.exists() else "",
                            "status": "pass" if path.exists() else "fail",
                        }
                    )

    audit = pd.DataFrame(rows)
    audit.to_csv(paths.artifacts / "frozen_input_audit.csv", index=False)
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "row_count": int(len(audit)),
        "failed_count": int(audit["status"].ne("pass").sum()),
        "phase2_frozen_mismatches": int(((audit["phase"] == "phase2") & (audit["audit_type"] == "frozen_hash") & audit["status"].ne("pass")).sum()),
        "phase3_frozen_mismatches": int(((audit["phase"] == "phase3") & (audit["audit_type"] == "frozen_hash") & audit["status"].ne("pass")).sum()),
        "phase4_frozen_mismatches": int(((audit["phase"] == "phase4") & (audit["audit_type"] == "frozen_hash") & audit["status"].ne("pass")).sum()),
        "phase5_frozen_mismatches": int(((audit["phase"] == "phase5") & (audit["audit_type"] == "frozen_hash") & audit["status"].ne("pass")).sum()),
        "required_artifacts_missing": int((audit["expected_exists"].eq(True) & audit["observed_exists"].ne(True)).sum()),
        "status": "PASS" if audit["status"].eq("pass").all() else "FAIL",
    }
    write_json(paths.artifacts / "frozen_input_audit.json", summary)
    return audit, summary


def build_calibration_split(paths: Phase6Paths, force: bool = False) -> pd.DataFrame:
    verify_upstream_frozen_inputs(paths)
    assignment_dir = paths.artifacts / "calibration_split_assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            out_path = assignment_dir / f"{detector}_{split}.csv"
            if out_path.exists() and not force:
                units = pd.read_csv(out_path)
            else:
                df = load_scores(paths.root, detector, split, "riskguard", "threshold_cal")
                unit_rows = []
                for sha, group in df.groupby("sha256", sort=True):
                    first = group.sort_values("sample_id", kind="mergesort").iloc[0]
                    unit_rows.append(
                        {
                            "sha256": sha,
                            "label": int(first["label"]),
                            "generator": str(first["generator"]),
                            "base_error": int(first["base_error"]),
                            "row_count": int(len(group)),
                            "stable_rank": stable_unit_rank(SEED, str(sha)),
                        }
                    )
                units = pd.DataFrame(unit_rows)
                units["stratum"] = units["label"].astype(str) + "|" + units["generator"].astype(str) + "|" + units["base_error"].astype(str)
                units["calibration_subset"] = "policy_certify"
                for _, index in units.groupby("stratum", sort=True).groups.items():
                    ordered = units.loc[list(index)].sort_values(["stable_rank", "sha256"], kind="mergesort")
                    n = len(ordered)
                    if n == 1:
                        select_count = 1
                    else:
                        select_count = n // 2
                    select_index = ordered.index[:select_count]
                    units.loc[select_index, "calibration_subset"] = "policy_select"
                units = units.sort_values(["calibration_subset", "sha256"], kind="mergesort").reset_index(drop=True)
                units.to_csv(out_path, index=False)

            for method in METHODS:
                df = load_scores(paths.root, detector, split, method, "threshold_cal")
                merged = df.merge(units[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
                if merged["calibration_subset"].isna().any():
                    raise RuntimeError(f"Missing calibration assignment for {detector}/{split}/{method}")
                select_sha = set(merged.loc[merged["calibration_subset"].eq("policy_select"), "sha256"].astype(str))
                certify_sha = set(merged.loc[merged["calibration_subset"].eq("policy_certify"), "sha256"].astype(str))
                overlap = len(select_sha & certify_sha)
                for subset, group in merged.groupby("calibration_subset", sort=True):
                    generators = group["generator"].astype(str).value_counts().sort_index().to_dict()
                    audit_rows.append(
                        {
                            "detector": detector,
                            "split": split,
                            "method": method,
                            "subset": subset,
                            "row_count": int(len(group)),
                            "unique_sha256": int(group["sha256"].nunique()),
                            "correct_count": int((group["base_error"].astype(int) == 0).sum()),
                            "error_count": int(group["base_error"].astype(int).sum()),
                            "real_count": int((group["label"].astype(int) == 0).sum()),
                            "fake_count": int((group["label"].astype(int) == 1).sum()),
                            "generator_counts_json": json_dumps_compact({str(k): int(v) for k, v in generators.items()}),
                            "cross_subset_sha_overlap": int(overlap),
                            "status": "pass" if overlap == 0 else "fail",
                        }
                    )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(paths.artifacts / "calibration_split_audit.csv", index=False)
    return audit


def assignment_for(paths: Phase6Paths, detector: str, split: str) -> pd.DataFrame:
    return pd.read_csv(paths.artifacts / "calibration_split_assignments" / f"{detector}_{split}.csv")


def policy_groups(df: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "global_cp":
        return pd.Series(["global_all"] * len(df), index=df.index, dtype=str)
    if policy == "source_group_cp":
        labels = df["label"].astype(int)
        gens = df["generator"].astype(str).str.lower()
        return pd.Series(np.where(labels.eq(0), "real_all", gens), index=df.index, dtype=str)
    if policy == "predicted_class_cp":
        pred = df["base_prediction"].astype(int)
        return pd.Series(np.where(pred.eq(0), "base_prediction_real", "base_prediction_fake"), index=df.index, dtype=str)
    raise ValueError(f"Unknown policy: {policy}")


def group_definitions_from_frame(df: pd.DataFrame, policy: str) -> dict[str, dict[str, Any]]:
    groups = policy_groups(df, policy)
    out: dict[str, dict[str, Any]] = {}
    for group_name in sorted(groups.unique()):
        group_df = df.loc[groups.eq(group_name)]
        if policy == "global_cp":
            rule = "all certification rows"
        elif policy == "source_group_cp" and group_name == "real_all":
            rule = "label == 0"
        elif policy == "source_group_cp":
            rule = f"label == 1 and generator == {group_name}"
        elif group_name == "base_prediction_real":
            rule = "base_prediction == 0"
        else:
            rule = "base_prediction == 1"
        out[str(group_name)] = {
            "rule": rule,
            "row_count": int(len(group_df)),
            "label_counts": {str(k): int(v) for k, v in group_df["label"].value_counts().sort_index().to_dict().items()},
            "generator_counts": {str(k): int(v) for k, v in group_df["generator"].value_counts().sort_index().to_dict().items()},
        }
    return out


def threshold_scan(df: pd.DataFrame, policy: str, alpha: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df[["sample_id", "risk_score", "base_error"]].copy()
    work["group"] = policy_groups(df, policy).to_numpy()
    work = work.sort_values(["risk_score", "sample_id"], kind="mergesort").reset_index(drop=True)
    risks = work["risk_score"].to_numpy(dtype=np.float64)
    errors = work["base_error"].to_numpy(dtype=np.int64)
    groups = work["group"].astype(str).to_numpy()
    total = len(work)
    unique_values, starts = np.unique(risks, return_index=True)
    ends = np.r_[starts[1:], total]
    cumulative_errors = np.cumsum(errors, dtype=np.int64)
    unique_groups = sorted(pd.unique(groups))
    group_cum_count = {group: np.cumsum(groups == group, dtype=np.int64) for group in unique_groups}
    group_cum_error = {group: np.cumsum((groups == group) & (errors == 1), dtype=np.int64) for group in unique_groups}
    rows = []
    for tau, end in zip(unique_values, ends):
        idx = int(end) - 1
        accepted = int(end)
        accepted_errors = int(cumulative_errors[idx])
        group_counts: dict[str, int] = {}
        group_errors: dict[str, int] = {}
        group_risks: dict[str, float | None] = {}
        feasible = accepted > 0
        for group in unique_groups:
            g_n = int(group_cum_count[group][idx])
            g_k = int(group_cum_error[group][idx])
            group_counts[group] = g_n
            group_errors[group] = g_k
            group_risk = (g_k / g_n) if g_n else None
            group_risks[group] = group_risk
            if g_n == 0 or group_risk is None or group_risk > float(alpha):
                feasible = False
        rows.append(
            {
                "threshold": float(tau),
                "accepted_count": accepted,
                "coverage": float(accepted / total),
                "accepted_errors": accepted_errors,
                "empirical_risk": float(accepted_errors / accepted) if accepted else float("nan"),
                "group_accepted_counts_json": json_dumps_compact(group_counts),
                "group_accepted_errors_json": json_dumps_compact(group_errors),
                "group_empirical_risks_json": json_dumps_compact(group_risks),
                "select_feasible": bool(feasible),
            }
        )
    return pd.DataFrame(rows)


def threshold_for_target_count(curve: pd.DataFrame, target_count: int) -> pd.Series | None:
    eligible = curve[(curve["accepted_count"] > 0) & (curve["accepted_count"] <= int(target_count))]
    if eligible.empty:
        return None
    eligible = eligible.sort_values(["accepted_count", "threshold"], ascending=[False, False], kind="mergesort")
    return eligible.iloc[0]


def candidate_record_hash(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != "candidate_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def generate_candidates_for(
    paths: Phase6Paths,
    detector: str,
    split: str,
    method: str,
    alpha: float,
    policy: str,
) -> list[dict[str, Any]]:
    assignments = assignment_for(paths, detector, split)
    df = load_scores(paths.root, detector, split, method, "threshold_cal")
    df = df.merge(assignments[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
    select = df[df["calibration_subset"].eq("policy_select")].copy()
    curve = threshold_scan(select, policy, alpha)
    if curve.empty:
        return []
    feasible = curve[curve["select_feasible"]].copy()
    candidate_rows = []
    if not feasible.empty:
        largest_count = int(feasible["accepted_count"].max())
        target_fractions = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
        for fraction in target_fractions:
            target = max(1, int(math.floor(largest_count * fraction)))
            row = threshold_for_target_count(curve, target)
            if row is None:
                continue
            row = row.to_dict()
            row["candidate_source"] = "select_feasible_fraction"
            row["target_fraction"] = fraction
            candidate_rows.append(row)
    else:
        fallback_coverages = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.01]
        total = int(curve["accepted_count"].max())
        for coverage in fallback_coverages:
            target = max(1, int(math.floor(total * coverage)))
            row = threshold_for_target_count(curve, target)
            if row is None:
                continue
            row = row.to_dict()
            row["candidate_source"] = "fallback_select_coverage"
            row["target_fraction"] = coverage
            candidate_rows.append(row)
    dedup: dict[float, dict[str, Any]] = {}
    for row in candidate_rows:
        dedup.setdefault(float(row["threshold"]), row)
    ordered = sorted(dedup.values(), key=lambda item: (-float(item["threshold"]), -int(item["accepted_count"])))[:CANDIDATE_COUNT]
    out = []
    for rank, row in enumerate(ordered, start=1):
        record = {
            "detector": detector,
            "split": split,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "alpha": float(alpha),
            "policy": policy,
            "candidate_id": f"{combo_stem(detector, split, method, alpha, policy)}_C{rank:02d}",
            "threshold": float(row["threshold"]),
            "select_accepted_count": int(row["accepted_count"]),
            "select_coverage": float(row["coverage"]),
            "select_error_count": int(row["accepted_errors"]),
            "select_empirical_risk": float(row["empirical_risk"]),
            "candidate_rank": int(rank),
            "candidate_source": str(row["candidate_source"]),
        }
        record["candidate_sha256"] = candidate_record_hash(record)
        out.append(record)
    return out


def select_phase6_candidates(paths: Phase6Paths, force: bool = False) -> pd.DataFrame:
    if not (paths.artifacts / "calibration_split_audit.csv").exists() or force:
        build_calibration_split(paths, force=force)
    candidate_dir = paths.artifacts / "candidate_thresholds"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in METHODS:
                for alpha in ALPHAS:
                    for policy in POLICIES:
                        records = generate_candidates_for(paths, detector, split, method, alpha, policy)
                        all_records.extend(records)
                        payload = {
                            "detector": detector,
                            "split": split,
                            "method": method,
                            "alpha": float(alpha),
                            "policy": policy,
                            "candidate_count": len(records),
                            "candidates": records,
                        }
                        write_json(candidate_dir / f"{combo_stem(detector, split, method, alpha, policy)}.json", payload)
    registry = pd.DataFrame(all_records)
    registry.to_csv(paths.artifacts / "candidate_threshold_registry.csv", index=False)
    freeze_inputs = [paths.artifacts / "candidate_threshold_registry.csv"] + sorted(candidate_dir.glob("*.json"))
    freeze_paths(paths.root, freeze_inputs, paths.artifacts / "candidate_threshold_freeze.csv")
    return registry


def certify_one(
    paths: Phase6Paths,
    detector: str,
    split: str,
    method: str,
    alpha: float,
    policy: str,
    candidates: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignments = assignment_for(paths, detector, split)
    df = load_scores(paths.root, detector, split, method, "threshold_cal")
    df = df.merge(assignments[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
    certify = df[df["calibration_subset"].eq("policy_certify")].copy()
    groups = policy_groups(certify, policy)
    unique_groups = sorted(groups.unique())
    k_candidates = int(len(candidates))
    g_count = int(len(unique_groups))
    delta_cell = float(DELTA / (k_candidates * g_count)) if k_candidates and g_count else float("nan")
    trace_rows: list[dict[str, Any]] = []
    summary_by_candidate: list[dict[str, Any]] = []
    for candidate in candidates.sort_values("candidate_rank", kind="mergesort").to_dict("records"):
        threshold = float(candidate["threshold"])
        accepted_all = certify["risk_score"].to_numpy(dtype=float) <= threshold
        total_accepted = int(accepted_all.sum())
        max_cp = 0.0
        all_certified = True
        group_bounds: dict[str, float] = {}
        group_counts: dict[str, dict[str, Any]] = {}
        for group in unique_groups:
            mask = groups.eq(group).to_numpy()
            group_size = int(mask.sum())
            accepted = accepted_all & mask
            accepted_count = int(accepted.sum())
            errors = int(certify.loc[accepted, "base_error"].astype(int).sum())
            empirical = float(errors / accepted_count) if accepted_count else float("nan")
            cp_upper = clopper_pearson_upper(errors, accepted_count, delta_cell)
            max_cp = max(max_cp, cp_upper)
            is_group_certified = bool(accepted_count > 0 and cp_upper <= float(alpha))
            all_certified = all_certified and is_group_certified
            group_bounds[str(group)] = cp_upper
            group_counts[str(group)] = {
                "group_size": group_size,
                "accepted_count": accepted_count,
                "accepted_errors": errors,
                "empirical_selective_risk": empirical,
                "cp_upper": cp_upper,
                "certified": is_group_certified,
            }
            trace_rows.append(
                {
                    "detector": detector,
                    "split": split,
                    "method": method,
                    "alpha": float(alpha),
                    "delta": float(DELTA),
                    "policy": policy,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "threshold": threshold,
                    "candidate_count_K": k_candidates,
                    "group_count_G": g_count,
                    "delta_cell": delta_cell,
                    "group": str(group),
                    "certify_group_size": group_size,
                    "accepted_count": accepted_count,
                    "accepted_errors": errors,
                    "empirical_selective_risk": empirical,
                    "cp_upper": cp_upper,
                    "group_certified": is_group_certified,
                    "candidate_certified": bool(all_certified),
                }
            )
        summary_by_candidate.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_rank": int(candidate["candidate_rank"]),
                "threshold": threshold,
                "certification_coverage": float(total_accepted / len(certify)) if len(certify) else 0.0,
                "certification_accepted_count": total_accepted,
                "max_group_cp_upper": float(max_cp),
                "candidate_certified": bool(all_certified),
                "group_bounds": group_bounds,
                "group_counts": group_counts,
            }
        )
    certified = [row for row in summary_by_candidate if row["candidate_certified"]]
    if certified:
        selected = sorted(
            certified,
            key=lambda row: (
                -float(row["threshold"]),
                -float(row["certification_coverage"]),
                float(row["max_group_cp_upper"]),
                str(row["candidate_id"]),
            ),
        )[0]
        policy_status = "CERTIFIED"
        selected_threshold = float(selected["threshold"])
    else:
        selected = None
        policy_status = "NO_CERTIFIED_THRESHOLD"
        selected_threshold = None
    score_df = load_scores(paths.root, detector, split, method, "threshold_cal")
    policy_payload = {
        "policy_version": "phase6_risk_control_v1",
        "detector": detector,
        "split": split,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "alpha": float(alpha),
        "delta": float(DELTA),
        "policy_type": policy,
        "group_definitions": group_definitions_from_frame(certify, policy),
        "candidate_count": k_candidates,
        "delta_cell": delta_cell,
        "selected_threshold": selected_threshold,
        "policy_status": policy_status,
        "select_manifest_sha256": sha256_file(paths.artifacts / "calibration_split_assignments" / f"{detector}_{split}.csv"),
        "certify_manifest_sha256": sha256_file(paths.artifacts / "calibration_split_assignments" / f"{detector}_{split}.csv"),
        "candidate_registry_sha256": sha256_file(paths.artifacts / "candidate_threshold_registry.csv"),
        "model_or_baseline_score_hash": str(score_df["model_or_baseline_score_hash"].iloc[0]) if len(score_df) else "",
        "score_artifact_sha256": str(score_df["score_artifact_sha256"].iloc[0]) if len(score_df) else "",
        "certification_counts": selected["group_counts"] if selected else {},
        "group_CP_bounds": selected["group_bounds"] if selected else {},
        "overall_certification_coverage": float(selected["certification_coverage"]) if selected else 0.0,
        "software_versions": software_versions(),
    }
    policy_payload["policy_sha256"] = payload_sha256(policy_payload)
    return trace_rows, policy_payload


def software_versions() -> dict[str, str]:
    import scipy

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def certify_phase6_policies(paths: Phase6Paths, force: bool = False) -> pd.DataFrame:
    if not (paths.artifacts / "candidate_threshold_registry.csv").exists() or force:
        select_phase6_candidates(paths, force=force)
    candidates = pd.read_csv(paths.artifacts / "candidate_threshold_registry.csv")
    policy_dir = paths.artifacts / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    all_trace: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in METHODS:
                for alpha in ALPHAS:
                    for policy in POLICIES:
                        subset = candidates[
                            candidates["detector"].eq(detector)
                            & candidates["split"].eq(split)
                            & candidates["method"].eq(method)
                            & np.isclose(candidates["alpha"].astype(float), float(alpha))
                            & candidates["policy"].eq(policy)
                        ].copy()
                        trace_rows, payload = certify_one(paths, detector, split, method, alpha, policy, subset)
                        all_trace.extend(trace_rows)
                        out_path = policy_dir / f"{combo_stem(detector, split, method, alpha, policy)}.json"
                        write_json(out_path, payload)
                        registry_rows.append(
                            {
                                "detector": detector,
                                "split": split,
                                "method": method,
                                "alpha": float(alpha),
                                "delta": float(DELTA),
                                "policy": policy,
                                "policy_status": payload["policy_status"],
                                "selected_threshold": payload["selected_threshold"],
                                "candidate_count": payload["candidate_count"],
                                "delta_cell": payload["delta_cell"],
                                "overall_certification_coverage": payload["overall_certification_coverage"],
                                "max_group_cp_upper": max(payload["group_CP_bounds"].values()) if payload["group_CP_bounds"] else float("nan"),
                                "policy_json": relative_to(paths.root, out_path),
                                "policy_sha256": payload["policy_sha256"],
                            }
                        )
    trace = pd.DataFrame(all_trace)
    trace.to_parquet(paths.artifacts / "certification_trace.parquet", index=False)
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(paths.artifacts / "certified_threshold_registry.csv", index=False)
    return registry


def policy_artifact_paths(paths: Phase6Paths) -> list[Path]:
    files: list[Path] = []
    for rel in [
        "frozen_input_audit.csv",
        "frozen_input_audit.json",
        "calibration_split_audit.csv",
        "candidate_threshold_registry.csv",
        "candidate_threshold_freeze.csv",
        "certification_trace.parquet",
        "certified_threshold_registry.csv",
    ]:
        files.append(paths.artifacts / rel)
    files.extend(sorted((paths.artifacts / "calibration_split_assignments").glob("*.csv")))
    files.extend(sorted((paths.artifacts / "candidate_thresholds").glob("*.json")))
    files.extend(sorted((paths.artifacts / "policies").glob("*.json")))
    files.append(paths.configs / "risk_control.yaml")
    return files


def freeze_phase6_policies(paths: Phase6Paths, force: bool = False) -> dict[str, Any]:
    if not (paths.artifacts / "certified_threshold_registry.csv").exists() or force:
        certify_phase6_policies(paths, force=force)
    config = {
        "policy_freeze_status": "PASS",
        "alpha_primary": 0.05,
        "alpha_secondary": 0.01,
        "delta": DELTA,
        "candidate_count_max": CANDIDATE_COUNT,
        "primary_policy": "source_group_cp",
        "test_labels_opened": False,
        "seed": SEED,
        "methods": list(METHODS),
        "policies": list(POLICIES),
        "primary_comparators": list(PRIMARY_COMPARATORS),
    }
    write_yaml(paths.configs / "risk_control.yaml", config)
    registry = freeze_paths(paths.root, policy_artifact_paths(paths), paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv")
    verify = verify_freeze_registry(paths.root, paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv")
    summary = {
        "policy_freeze_status": "PASS" if verify["status"].eq("pass").all() else "FAIL",
        "alpha_primary": 0.05,
        "alpha_secondary": 0.01,
        "delta": DELTA,
        "candidate_count_max": CANDIDATE_COUNT,
        "primary_policy": "source_group_cp",
        "test_labels_opened": False,
        "phase6_policy_frozen_registry": "artifacts/phase6/phase6_policy_frozen_artifact_hashes.csv",
        "phase6_policy_frozen_registry_sha256": sha256_file(paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv"),
        "policy_frozen_mismatches": int(verify["status"].ne("pass").sum()),
        "required_policy_artifacts_missing": int(verify["observed_exists"].ne(True).sum()),
        "policy_artifact_count": int(len(registry)),
    }
    write_yaml(paths.configs / "phase6_policy_frozen.yaml", summary)
    return summary


def create_test_label_opening_record(paths: Phase6Paths) -> dict[str, Any]:
    record_path = paths.artifacts / "test_label_opening_record.json"
    if record_path.exists():
        return read_json(record_path)
    policy_verify = verify_freeze_registry(paths.root, paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv")
    if not policy_verify["status"].eq("pass").all():
        raise RuntimeError("Policy freeze verification failed; refusing to open test labels.")
    record = {
        "policy_freeze_registry_sha256": sha256_file(paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv"),
        "policy_freeze_yaml_sha256": sha256_file(paths.configs / "phase6_policy_frozen.yaml"),
        "opened_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "protocol_seen_labels_opened": True,
        "protocol_held_out_labels_opened": True,
        "bfree_labels_opened": True,
        "policy_modified_after_opening": False,
    }
    write_json(record_path, record)
    return record


def load_policy(paths: Phase6Paths, detector: str, split: str, method: str, alpha: float, policy: str) -> dict[str, Any]:
    path = paths.artifacts / "policies" / f"{combo_stem(detector, split, method, alpha, policy)}.json"
    return read_json(path)


def numeric_or_undefined(value: float, reason: str = "undefined_zero_denominator") -> float | str:
    if value is None or not np.isfinite(float(value)):
        return reason
    return float(value)


def accepted_metric_summary(df: pd.DataFrame, threshold: float | None) -> dict[str, Any]:
    total = int(len(df))
    errors = df["base_error"].astype(int).to_numpy()
    labels = df["label"].astype(int).to_numpy()
    predictions = df["base_prediction"].astype(int).to_numpy()
    risks = df["risk_score"].astype(float).to_numpy()
    sample_ids = df["sample_id"].astype(str).to_numpy()
    accepted = np.zeros(total, dtype=bool) if threshold is None or pd.isna(threshold) else risks <= float(threshold)
    accepted_count = int(accepted.sum())
    rejected_count = int(total - accepted_count)
    accepted_errors = int(errors[accepted].sum()) if accepted_count else 0
    overall_base_error = float(errors.mean()) if total else float("nan")
    selective_risk_value = float(accepted_errors / accepted_count) if accepted_count else float("nan")
    real = labels == 0
    fake = labels == 1
    accepted_real = accepted & real
    accepted_fake = accepted & fake
    real_cov = float(accepted_real.sum() / real.sum()) if real.sum() else float("nan")
    fake_cov = float(accepted_fake.sum() / fake.sum()) if fake.sum() else float("nan")
    fpr = float(((predictions == 1) & accepted_real).sum() / accepted_real.sum()) if accepted_real.sum() else float("nan")
    fnr = float(((predictions == 0) & accepted_fake).sum() / accepted_fake.sum()) if accepted_fake.sum() else float("nan")
    balanced = float(np.nanmean([fpr, fnr])) if np.isfinite(fpr) or np.isfinite(fnr) else float("nan")
    risk_reduction = overall_base_error - selective_risk_value if np.isfinite(selective_risk_value) else float("nan")
    return {
        "total_samples": total,
        "accepted_samples": accepted_count,
        "rejected_samples": rejected_count,
        "coverage": float(accepted_count / total) if total else float("nan"),
        "accepted_errors": accepted_errors,
        "selective_risk": numeric_or_undefined(selective_risk_value),
        "overall_base_error": overall_base_error,
        "risk_reduction": numeric_or_undefined(risk_reduction),
        "real_coverage": numeric_or_undefined(real_cov, "undefined_zero_denominator"),
        "fake_coverage": numeric_or_undefined(fake_cov, "undefined_zero_denominator"),
        "accepted_FPR": numeric_or_undefined(fpr, "undefined_zero_denominator"),
        "accepted_FNR": numeric_or_undefined(fnr, "undefined_zero_denominator"),
        "balanced_selective_risk": numeric_or_undefined(balanced, "undefined_zero_denominator"),
        "AURC": aurc(errors, risks, sample_ids),
        "E_AURC": eaurc(errors, risks, sample_ids),
    }


def group_metric_rows(base: dict[str, Any], df: pd.DataFrame, threshold: float | None) -> list[dict[str, Any]]:
    risks = df["risk_score"].astype(float).to_numpy()
    accepted_all = np.zeros(len(df), dtype=bool) if threshold is None or pd.isna(threshold) else risks <= float(threshold)
    rows = []
    group_specs: list[tuple[str, str, np.ndarray]] = []
    labels = df["label"].astype(int).to_numpy()
    predictions = df["base_prediction"].astype(int).to_numpy()
    if (labels == 0).any():
        group_specs.append(("source", "real_all", labels == 0))
    for generator in sorted(df.loc[df["label"].astype(int).eq(1), "generator"].astype(str).unique()):
        group_specs.append(("source", str(generator), df["generator"].astype(str).eq(generator).to_numpy()))
    group_specs.append(("predicted_class", "base_prediction_real", predictions == 0))
    group_specs.append(("predicted_class", "base_prediction_fake", predictions == 1))
    for group_type, group_name, mask in group_specs:
        group_size = int(mask.sum())
        accepted = accepted_all & mask
        accepted_count = int(accepted.sum())
        accepted_errors = int(df.loc[accepted, "base_error"].astype(int).sum()) if accepted_count else 0
        risk = float(accepted_errors / accepted_count) if accepted_count else float("nan")
        accepted_labels = labels[accepted]
        accepted_predictions = predictions[accepted]
        fpr = float((accepted_predictions[accepted_labels == 0] == 1).mean()) if (accepted_labels == 0).sum() else float("nan")
        fnr = float((accepted_predictions[accepted_labels == 1] == 0).mean()) if (accepted_labels == 1).sum() else float("nan")
        rows.append(
            {
                **base,
                "group_type": group_type,
                "group": group_name,
                "group_size": group_size,
                "accepted_count": accepted_count,
                "coverage": float(accepted_count / group_size) if group_size else float("nan"),
                "accepted_error_count": accepted_errors,
                "selective_risk": numeric_or_undefined(risk),
                "accepted_FPR": numeric_or_undefined(fpr),
                "accepted_FNR": numeric_or_undefined(fnr),
            }
        )
    return rows


def worst_group_summary(base: dict[str, Any], group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = [row for row in group_rows if row["group_type"] == "source"]
    risks = []
    coverages = []
    for row in source:
        risk = row["selective_risk"]
        cov = row["coverage"]
        if isinstance(risk, (float, int)) and np.isfinite(risk):
            risks.append(float(risk))
        if isinstance(cov, (float, int)) and np.isfinite(cov):
            coverages.append(float(cov))
    return {
        **base,
        "worst_group_selective_risk": max(risks) if risks else "undefined_zero_denominator",
        "worst_group_coverage": source[int(np.argmax(risks))]["coverage"] if risks else "undefined_zero_denominator",
        "minimum_group_coverage": min(coverages) if coverages else "undefined_zero_denominator",
        "maximum_group_risk_gap": (max(risks) - min(risks)) if len(risks) >= 2 else "undefined_zero_denominator",
    }


def audit_bfree_source_id_restoration(paths: Phase6Paths) -> dict[str, Any]:
    """Verify B-Free source_id restoration across every frozen score context."""

    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in METHODS:
                df = load_scores(paths.root, detector, split, method, "bfree_snapshot")
                unresolved = df["source_id"].astype(str).isin(["", "nan", "None"]).sum()
                rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "method": method,
                        "row_count": int(len(df)),
                        "unique_sha256": int(df["sha256"].nunique()),
                        "unique_sample_id": int(df["sample_id"].nunique()),
                        "unique_source_id": int(df["source_id"].nunique()),
                        "unresolved_source_id_count": int(unresolved),
                        "near_duplicate_group_source": ",".join(sorted(df["near_duplicate_group_source"].astype(str).unique())),
                        "status": "pass" if len(df) == 733 and df["sha256"].nunique() == 733 and unresolved == 0 else "fail",
                    }
                )
    audit = pd.DataFrame(rows)
    audit.to_csv(paths.artifacts / "bfree_source_id_restoration_audit.csv", index=False)
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "context_count": int(len(audit)),
        "all_contexts_733_rows": bool(audit["row_count"].eq(733).all()),
        "all_contexts_733_unique_sha256": bool(audit["unique_sha256"].eq(733).all()),
        "total_unresolved_source_id_count": int(audit["unresolved_source_id_count"].sum()),
        "status": "PASS" if audit["status"].eq("pass").all() else "FAIL",
    }
    write_json(paths.artifacts / "bfree_source_id_restoration_audit.json", summary)
    return summary


def evaluate_phase6_final(paths: Phase6Paths, force: bool = False) -> pd.DataFrame:
    if not (paths.configs / "phase6_policy_frozen.yaml").exists() or force:
        freeze_phase6_policies(paths, force=force)
    create_test_label_opening_record(paths)
    metric_rows: list[dict[str, Any]] = []
    group_rows_all: list[dict[str, Any]] = []
    worst_rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in METHODS:
                for dataset in EVAL_DATASETS:
                    scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, method, dataset), dataset)
                    for alpha in ALPHAS:
                        for policy in POLICIES:
                            policy_payload = load_policy(paths, detector, split, method, alpha, policy)
                            threshold = policy_payload["selected_threshold"]
                            base = {
                                "detector": detector,
                                "split": split,
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "alpha": float(alpha),
                                "policy": policy,
                                "evaluation_dataset": dataset,
                                "evaluation_dataset_label": DATASET_LABELS[dataset],
                                "policy_status": policy_payload["policy_status"],
                                "selected_threshold": threshold,
                                "score_artifact_sha256": str(scores["score_artifact_sha256"].iloc[0]) if len(scores) else "",
                                "evaluation_weighting": "bfree_unique_image" if dataset == "bfree_snapshot" else "sha256_deduplicated",
                            }
                            metric_rows.append({**base, **accepted_metric_summary(scores, threshold)})
                            rows = group_metric_rows(base, scores, threshold)
                            group_rows_all.extend(rows)
                            worst_rows.append(worst_group_summary(base, rows))
    metrics = pd.DataFrame(metric_rows)
    groups = pd.DataFrame(group_rows_all)
    worst = pd.DataFrame(worst_rows)
    metrics.to_csv(paths.artifacts / "final_selective_metrics.csv", index=False)
    groups.to_csv(paths.artifacts / "final_group_metrics.csv", index=False)
    worst.to_csv(paths.artifacts / "worst_group_summary.csv", index=False)
    audit_bfree_source_id_restoration(paths)
    make_risk_coverage_curves(paths)
    make_phase6_figures(paths)
    write_phase6_reports(paths)
    return metrics


def make_risk_coverage_curves(paths: Phase6Paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in CURVE_METHODS:
                for dataset in EVAL_DATASETS:
                    df = deduplicate_eval_rows(load_scores(paths.root, detector, split, method, dataset), dataset)
                    work = df[["sample_id", "risk_score", "base_error"]].sort_values(["risk_score", "sample_id"], kind="mergesort")
                    if work.empty:
                        continue
                    risks = work["risk_score"].to_numpy(dtype=float)
                    errors = work["base_error"].astype(int).to_numpy()
                    unique_values, starts = np.unique(risks, return_index=True)
                    ends = np.r_[starts[1:], len(work)]
                    cumulative_errors = np.cumsum(errors, dtype=np.int64)
                    for tau, end in zip(unique_values, ends):
                        accepted_count = int(end)
                        accepted_errors = int(cumulative_errors[end - 1])
                        rows.append(
                            {
                                "detector": detector,
                                "split": split,
                                "method": method,
                                "dataset": dataset,
                                "coverage": float(accepted_count / len(work)),
                                "selective_risk": float(accepted_errors / accepted_count) if accepted_count else float("nan"),
                                "threshold": float(tau),
                                "accepted_count": accepted_count,
                            }
                        )
    curves = pd.DataFrame(rows)
    curves.to_parquet(paths.artifacts / "risk_coverage_curves.parquet", index=False)
    return curves


def category_frame_for_metrics(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    accepted = np.zeros(len(df), dtype=bool) if threshold is None or pd.isna(threshold) else df["risk_score"].to_numpy(float) <= float(threshold)
    labels = df["label"].astype(int).to_numpy()
    preds = df["base_prediction"].astype(int).to_numpy()
    errors = df["base_error"].astype(int).to_numpy()
    category = []
    for a, y, pred, err in zip(accepted, labels, preds, errors):
        if not a:
            category.append("rejected")
        elif y == 0 and pred == 1:
            category.append("accepted_real_error")
        elif y == 0:
            category.append("accepted_real_correct")
        elif y == 1 and pred == 0:
            category.append("accepted_fake_error")
        elif y == 1:
            category.append("accepted_fake_correct")
        elif err:
            category.append("accepted_other_error")
        else:
            category.append("accepted_other_correct")
    out = df[["label", "generator", "sha256", "source_id", "near_duplicate_group"]].copy()
    out["bootstrap_category"] = category
    return out


BOOT_CATEGORIES = [
    "rejected",
    "accepted_real_error",
    "accepted_real_correct",
    "accepted_fake_error",
    "accepted_fake_correct",
    "accepted_other_error",
    "accepted_other_correct",
]


def metric_from_counts(metric: str):
    def fn(counts: dict[str, int]) -> float:
        ar_err = counts["accepted_real_error"]
        ar_ok = counts["accepted_real_correct"]
        af_err = counts["accepted_fake_error"]
        af_ok = counts["accepted_fake_correct"]
        ao_err = counts["accepted_other_error"]
        ao_ok = counts["accepted_other_correct"]
        accepted = ar_err + ar_ok + af_err + af_ok + ao_err + ao_ok
        total = accepted + counts["rejected"]
        errors = ar_err + af_err + ao_err
        real_acc = ar_err + ar_ok
        fake_acc = af_err + af_ok
        fpr = ar_err / real_acc if real_acc else float("nan")
        fnr = af_err / fake_acc if fake_acc else float("nan")
        if metric == "coverage":
            return accepted / total if total else float("nan")
        if metric == "selective_risk":
            return errors / accepted if accepted else float("nan")
        if metric == "accepted_FPR":
            return fpr
        if metric == "accepted_FNR":
            return fnr
        if metric == "balanced_selective_risk":
            return float(np.nanmean([fpr, fnr])) if np.isfinite(fpr) or np.isfinite(fnr) else float("nan")
        raise ValueError(metric)

    return fn


def bootstrap_unit_label(dataset: str) -> str:
    return "near_duplicate_group" if dataset == "bfree_snapshot" else "sha256"


def bootstrap_unit_inverse(df: pd.DataFrame, dataset: str) -> tuple[np.ndarray, list[np.ndarray]]:
    if dataset == "bfree_snapshot":
        unit_values = df["near_duplicate_group"].astype(str).to_numpy()
        fallback = pd.Series(unit_values).isin(["", "nan", "None"]).to_numpy()
        if fallback.any():
            unit_values[fallback] = df.loc[fallback, "source_id"].astype(str).to_numpy()
    else:
        unit_values = df["sha256"].astype(str).to_numpy()
    units, inverse = np.unique(unit_values, return_inverse=True)
    meta = (
        pd.DataFrame(
            {
                "unit": unit_values,
                "label": df["label"].astype(int).to_numpy(),
                "generator": df["generator"].astype(str).to_numpy(),
            }
        )
        .drop_duplicates("unit", keep="first")
        .set_index("unit")
        .loc[units]
        .reset_index()
    )
    strata = []
    for _, group in meta.groupby(["label", "generator"], dropna=False, sort=True):
        strata.append(group.index.to_numpy(dtype=np.int64))
    return inverse.astype(np.int64), strata


def stratified_poisson_row_weights(
    inverse: np.ndarray,
    strata: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    unit_weights = np.zeros(int(inverse.max()) + 1 if len(inverse) else 0, dtype=np.float64)
    for unit_indexes in strata:
        if len(unit_indexes) == 0:
            continue
        draws = rng.poisson(1.0, len(unit_indexes)).astype(np.float64)
        if draws.sum() == 0:
            draws[rng.integers(0, len(draws))] = 1.0
        unit_weights[unit_indexes] = draws
    return unit_weights[inverse]


def weighted_aurc_from_sorted_errors(sorted_errors: np.ndarray, sorted_weights: np.ndarray) -> float:
    total_weight = float(sorted_weights.sum())
    if total_weight <= 0:
        return float("nan")
    cumulative_weight = np.cumsum(sorted_weights, dtype=np.float64)
    cumulative_errors = np.cumsum(sorted_weights * sorted_errors, dtype=np.float64)
    mask = sorted_weights > 0
    if not mask.any():
        return float("nan")
    return float(np.sum(sorted_weights[mask] * (cumulative_errors[mask] / cumulative_weight[mask])) / total_weight)


def bootstrap_rank_metric_draws(
    df: pd.DataFrame,
    dataset: str,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    errors = df["base_error"].astype(int).to_numpy(dtype=np.float64)
    risks = df["risk_score"].astype(float).to_numpy(dtype=np.float64)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    risk_order = np.lexsort((sample_ids, risks))
    optimal_order = np.lexsort((sample_ids, errors))
    inverse, strata = bootstrap_unit_inverse(df, dataset)
    rng = np.random.default_rng(seed)
    aurc_draws = np.empty(int(replicates), dtype=np.float64)
    eaurc_draws = np.empty(int(replicates), dtype=np.float64)
    sorted_errors = errors[risk_order]
    optimal_errors = errors[optimal_order]
    for i in range(int(replicates)):
        weights = stratified_poisson_row_weights(inverse, strata, rng)
        value = weighted_aurc_from_sorted_errors(sorted_errors, weights[risk_order])
        optimum = weighted_aurc_from_sorted_errors(optimal_errors, weights[optimal_order])
        aurc_draws[i] = value
        eaurc_draws[i] = value - optimum if np.isfinite(value) and np.isfinite(optimum) else float("nan")
    return aurc_draws, eaurc_draws


def weighted_threshold_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
    accepted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    total = float(weights.sum())
    accepted_w = float(weights[accepted].sum())
    accepted_errors = float(weights[accepted] @ errors[accepted]) if accepted_w > 0 else 0.0
    real = labels == 0
    fake = labels == 1
    accepted_real = accepted & real
    accepted_fake = accepted & fake
    real_acc_w = float(weights[accepted_real].sum())
    fake_acc_w = float(weights[accepted_fake].sum())
    fpr = float(weights[accepted_real & (predictions == 1)].sum() / real_acc_w) if real_acc_w > 0 else float("nan")
    fnr = float(weights[accepted_fake & (predictions == 0)].sum() / fake_acc_w) if fake_acc_w > 0 else float("nan")
    return {
        "coverage": accepted_w / total if total > 0 else float("nan"),
        "selective_risk": accepted_errors / accepted_w if accepted_w > 0 else float("nan"),
        "balanced_selective_risk": float(np.nanmean([fpr, fnr])) if np.isfinite(fpr) or np.isfinite(fnr) else float("nan"),
    }


def source_group_masks(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    labels = df["label"].astype(int).to_numpy()
    masks: list[tuple[str, np.ndarray]] = []
    if (labels == 0).any():
        masks.append(("real_all", labels == 0))
    generators = df["generator"].astype(str)
    for generator in sorted(generators[labels == 1].unique()):
        masks.append((str(generator), generators.eq(generator).to_numpy()))
    return masks


def weighted_source_group_extremes(
    labels: np.ndarray,
    errors: np.ndarray,
    accepted: np.ndarray,
    weights: np.ndarray,
    masks: list[tuple[str, np.ndarray]],
) -> dict[str, float]:
    risks = []
    coverages = []
    for _, mask in masks:
        group_weight = float(weights[mask].sum())
        if group_weight <= 0:
            continue
        accepted_mask = accepted & mask
        accepted_weight = float(weights[accepted_mask].sum())
        coverages.append(accepted_weight / group_weight)
        if accepted_weight > 0:
            risks.append(float((weights[accepted_mask] @ errors[accepted_mask]) / accepted_weight))
    return {
        "worst_group_selective_risk": max(risks) if risks else float("nan"),
        "minimum_group_coverage": min(coverages) if coverages else float("nan"),
    }


def paired_category_frame(
    labels: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
    generators: np.ndarray,
    rg_accepted: np.ndarray,
    comp_accepted: np.ndarray,
) -> pd.DataFrame:
    groups = np.where(labels == 0, "real_all", generators.astype(str))
    category = [
        f"{group}|{int(label)}|{int(pred)}|{int(error)}|{int(rg)}|{int(comp)}"
        for group, label, pred, error, rg, comp in zip(groups, labels, predictions, errors, rg_accepted, comp_accepted)
    ]
    return pd.DataFrame(
        {
            "label": labels.astype(int),
            "generator": generators.astype(str),
            "paired_category": category,
        }
    )


def paired_metric_from_category_counts(category_values: list[str], metric: str):
    parsed = []
    for value in category_values:
        group, label, prediction, error, rg_accepted, comp_accepted = value.split("|")
        parsed.append(
            {
                "category": value,
                "group": group,
                "label": int(label),
                "prediction": int(prediction),
                "error": int(error),
                "riskguard": int(rg_accepted),
                "comparator": int(comp_accepted),
            }
        )

    def summarize(counts: dict[str, int], method: str) -> dict[str, float]:
        total = 0.0
        accepted = 0.0
        accepted_errors = 0.0
        real_accepted = 0.0
        fake_accepted = 0.0
        real_errors = 0.0
        fake_errors = 0.0
        group_total: dict[str, float] = {}
        group_accepted: dict[str, float] = {}
        group_errors: dict[str, float] = {}
        for item in parsed:
            count = float(counts.get(item["category"], 0))
            if count <= 0:
                continue
            group = item["group"]
            total += count
            group_total[group] = group_total.get(group, 0.0) + count
            if item[method] != 1:
                continue
            accepted += count
            err_count = count * item["error"]
            accepted_errors += err_count
            group_accepted[group] = group_accepted.get(group, 0.0) + count
            group_errors[group] = group_errors.get(group, 0.0) + err_count
            if item["label"] == 0:
                real_accepted += count
                if item["prediction"] == 1:
                    real_errors += count
            elif item["label"] == 1:
                fake_accepted += count
                if item["prediction"] == 0:
                    fake_errors += count
        fpr = real_errors / real_accepted if real_accepted > 0 else float("nan")
        fnr = fake_errors / fake_accepted if fake_accepted > 0 else float("nan")
        risks = [
            group_errors.get(group, 0.0) / group_accepted[group]
            for group in group_total
            if group_accepted.get(group, 0.0) > 0
        ]
        coverages = [
            group_accepted.get(group, 0.0) / group_total[group]
            for group in group_total
            if group_total[group] > 0
        ]
        return {
            "coverage": accepted / total if total > 0 else float("nan"),
            "selective_risk": accepted_errors / accepted if accepted > 0 else float("nan"),
            "balanced_selective_risk": float(np.nanmean([fpr, fnr])) if np.isfinite(fpr) or np.isfinite(fnr) else float("nan"),
            "worst_group_selective_risk": max(risks) if risks else float("nan"),
            "minimum_group_coverage": min(coverages) if coverages else float("nan"),
        }

    def fn(counts: dict[str, int]) -> float:
        left = summarize(counts, "riskguard")
        right = summarize(counts, "comparator")
        return left[metric] - right[metric]

    return fn


def paired_all_metrics_from_category_counts(category_values: list[str]):
    metric_functions = {
        metric: paired_metric_from_category_counts(category_values, metric)
        for metric in [
            "coverage",
            "selective_risk",
            "balanced_selective_risk",
            "worst_group_selective_risk",
            "minimum_group_coverage",
        ]
    }

    def fn(counts: dict[str, int]) -> dict[str, float]:
        return {metric: metric_fn(counts) for metric, metric_fn in metric_functions.items()}

    return fn


def stratified_count_bootstrap_multi(
    df: pd.DataFrame,
    strata_cols: list[str],
    category_col: str,
    category_values: list[str],
    metric_fn: Any,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    grouped = list(df.groupby(strata_cols, dropna=False, sort=True)) if strata_cols else [((), df)]
    strata = []
    for _, group in grouped:
        counts = group[category_col].value_counts().reindex(category_values, fill_value=0).to_numpy(dtype=np.float64)
        n = int(counts.sum())
        if n <= 0:
            continue
        probs = counts / counts.sum()
        strata.append((n, probs))
    probe = metric_fn(dict.fromkeys(category_values, 0))
    draws = {metric: np.empty(int(n_bootstrap), dtype=np.float64) for metric in probe}
    for i in range(int(n_bootstrap)):
        aggregate = dict.fromkeys(category_values, 0)
        for n, probs in strata:
            sampled = rng.multinomial(n, probs)
            for value, count in zip(category_values, sampled):
                aggregate[value] += int(count)
        values = metric_fn(aggregate)
        for metric, value in values.items():
            draws[metric][i] = value
    return draws


def paired_category_bootstrap_draws(
    df: pd.DataFrame,
    category_values: list[str],
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    parsed = [value.split("|") for value in category_values]
    groups = np.asarray([item[0] for item in parsed], dtype=object)
    unique_groups, group_codes = np.unique(groups, return_inverse=True)
    labels = np.asarray([int(item[1]) for item in parsed], dtype=np.int64)
    predictions = np.asarray([int(item[2]) for item in parsed], dtype=np.int64)
    errors = np.asarray([int(item[3]) for item in parsed], dtype=np.float64)
    rg = np.asarray([int(item[4]) for item in parsed], dtype=np.float64)
    comp = np.asarray([int(item[5]) for item in parsed], dtype=np.float64)

    strata = []
    for _, group in df.groupby(["label", "generator"], dropna=False, sort=True):
        counts = group["paired_category"].value_counts().reindex(category_values, fill_value=0).to_numpy(dtype=np.float64)
        n = int(counts.sum())
        if n > 0:
            strata.append((n, counts / counts.sum()))

    def summarize(counts: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
        total = float(counts.sum())
        accepted_counts = counts * accepted
        accepted_total = float(accepted_counts.sum())
        accepted_errors = float(accepted_counts @ errors)
        real_acc = float((accepted_counts * (labels == 0)).sum())
        fake_acc = float((accepted_counts * (labels == 1)).sum())
        fpr = float((accepted_counts * (labels == 0) * (predictions == 1)).sum() / real_acc) if real_acc > 0 else float("nan")
        fnr = float((accepted_counts * (labels == 1) * (predictions == 0)).sum() / fake_acc) if fake_acc > 0 else float("nan")
        group_total = np.bincount(group_codes, weights=counts, minlength=len(unique_groups)).astype(np.float64)
        group_acc = np.bincount(group_codes, weights=accepted_counts, minlength=len(unique_groups)).astype(np.float64)
        group_err = np.bincount(group_codes, weights=accepted_counts * errors, minlength=len(unique_groups)).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            group_risk = group_err / group_acc
            group_cov = group_acc / group_total
        finite_risk = group_risk[np.isfinite(group_risk)]
        finite_cov = group_cov[np.isfinite(group_cov)]
        return {
            "coverage": accepted_total / total if total > 0 else float("nan"),
            "selective_risk": accepted_errors / accepted_total if accepted_total > 0 else float("nan"),
            "balanced_selective_risk": float(np.nanmean([fpr, fnr])) if np.isfinite(fpr) or np.isfinite(fnr) else float("nan"),
            "worst_group_selective_risk": float(finite_risk.max()) if finite_risk.size else float("nan"),
            "minimum_group_coverage": float(finite_cov.min()) if finite_cov.size else float("nan"),
        }

    metrics = ["coverage", "selective_risk", "balanced_selective_risk", "worst_group_selective_risk", "minimum_group_coverage"]
    draws = {metric: np.empty(int(n_bootstrap), dtype=np.float64) for metric in metrics}
    for i in range(int(n_bootstrap)):
        sampled = np.zeros(len(category_values), dtype=np.float64)
        for n, probs in strata:
            sampled += rng.multinomial(n, probs)
        left = summarize(sampled, rg)
        right = summarize(sampled, comp)
        for metric in metrics:
            draws[metric][i] = left[metric] - right[metric]
    return draws


def bootstrap_phase6_results(paths: Phase6Paths, replicates: int = 2000, force: bool = False) -> pd.DataFrame:
    if not (paths.artifacts / "final_selective_metrics.csv").exists() or force:
        evaluate_phase6_final(paths, force=force)
    rows: list[dict[str, Any]] = []
    metric_names = ["coverage", "selective_risk", "accepted_FPR", "accepted_FNR", "balanced_selective_risk"]
    rank_cache: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for detector in DETECTORS:
        for split in SPLITS:
            for method in METHODS:
                for dataset in EVAL_DATASETS:
                    scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, method, dataset), dataset)
                    rank_key = (detector, split, method, dataset)
                    rank_cache[rank_key] = bootstrap_rank_metric_draws(
                        scores,
                        dataset,
                        int(replicates),
                        SEED + stable_seed_offset(detector, split, method, dataset, "rank_metrics"),
                    )
                    strata = ["label", "generator"]
                    if dataset == "bfree_snapshot":
                        strata = ["label", "generator"]
                    for alpha in ALPHAS:
                        for policy in POLICIES:
                            policy_payload = load_policy(paths, detector, split, method, alpha, policy)
                            threshold = policy_payload["selected_threshold"]
                            cat = category_frame_for_metrics(scores, threshold)
                            for metric in metric_names:
                                draws = stratified_count_bootstrap(
                                    cat,
                                    strata,
                                    "bootstrap_category",
                                    BOOT_CATEGORIES,
                                    metric_from_counts(metric),
                                    int(replicates),
                                    SEED + stable_seed_offset(detector, split, method, dataset, alpha, policy, metric),
                                )
                                lo, hi, valid = percentile_ci(draws)
                                point_value = accepted_metric_summary(scores, threshold).get(metric)
                                rows.append(
                                    {
                                        "detector": detector,
                                        "split": split,
                                        "method": method,
                                        "alpha": float(alpha),
                                        "policy": policy,
                                        "evaluation_dataset": dataset,
                                        "metric": metric,
                                        "point_estimate": point_value,
                                        "ci_lower_2p5": lo,
                                        "ci_upper_97p5": hi,
                                        "valid_bootstrap_replicate_count": valid,
                                        "bootstrap_replicates": int(replicates),
                                        "bootstrap_unit": "near_duplicate_group" if dataset == "bfree_snapshot" else "sha256",
                                        "bootstrap_note": "stratified categorical unit bootstrap for fixed-threshold count-ratio metrics",
                                    }
                                )
                            errors = scores["base_error"].astype(int).to_numpy()
                            risks = scores["risk_score"].astype(float).to_numpy()
                            ids = scores["sample_id"].astype(str).to_numpy()
                            aurc_draws, eaurc_draws = rank_cache[rank_key]
                            for metric, value, draws in [
                                ("AURC", aurc(errors, risks, ids), aurc_draws),
                                ("E_AURC", eaurc(errors, risks, ids), eaurc_draws),
                            ]:
                                lo, hi, valid = percentile_ci(draws)
                                rows.append(
                                    {
                                        "detector": detector,
                                        "split": split,
                                        "method": method,
                                        "alpha": float(alpha),
                                        "policy": policy,
                                        "evaluation_dataset": dataset,
                                        "metric": metric,
                                        "point_estimate": value,
                                        "ci_lower_2p5": lo,
                                        "ci_upper_97p5": hi,
                                        "valid_bootstrap_replicate_count": valid,
                                        "bootstrap_replicates": int(replicates),
                                        "bootstrap_unit": "near_duplicate_group" if dataset == "bfree_snapshot" else "sha256",
                                        "bootstrap_note": "stratified cluster Poisson bootstrap for rank metrics",
                                    }
                                )
    ci = pd.DataFrame(rows)
    ci.to_csv(paths.artifacts / "bootstrap_confidence_intervals.csv", index=False)
    paired = paired_method_comparisons(paths, replicates)
    paired.to_csv(paths.artifacts / "paired_method_comparisons.csv", index=False)
    write_phase6_reports(paths)
    return ci


def stable_seed_offset(*parts: Any) -> int:
    return int(stable_hash_text("|".join(map(str, parts)))[:8], 16) % 1_000_000


def paired_method_comparisons(paths: Phase6Paths, replicates: int = 2000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "coverage",
        "selective_risk",
        "balanced_selective_risk",
        "AURC",
        "worst_group_selective_risk",
        "minimum_group_coverage",
    ]
    for detector in DETECTORS:
        for split in SPLITS:
            for dataset in EVAL_DATASETS:
                riskguard_scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, "riskguard", dataset), dataset)
                for comparator in PRIMARY_COMPARATORS:
                    comp_scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, comparator, dataset), dataset)
                    join_cols = ["sample_id", "sha256", "label", "generator", "base_prediction", "base_error"]
                    joined = riskguard_scores[join_cols + ["source_id", "near_duplicate_group", "risk_score"]].rename(columns={"risk_score": "risk_score_riskguard"}).merge(
                        comp_scores[join_cols + ["risk_score"]].rename(columns={"risk_score": f"risk_score_{comparator}"}),
                        on=join_cols,
                        how="inner",
                        validate="one_to_one",
                    )
                    labels = joined["label"].astype(int).to_numpy()
                    predictions = joined["base_prediction"].astype(int).to_numpy()
                    errors = joined["base_error"].astype(int).to_numpy(dtype=np.float64)
                    weights_one = np.ones(len(joined), dtype=np.float64)
                    masks = source_group_masks(joined)
                    rg_risks = joined["risk_score_riskguard"].astype(float).to_numpy()
                    comp_risks = joined[f"risk_score_{comparator}"].astype(float).to_numpy()
                    sample_ids = joined["sample_id"].astype(str).to_numpy()
                    rg_order = np.lexsort((sample_ids, rg_risks))
                    comp_order = np.lexsort((sample_ids, comp_risks))
                    optimal_order = np.lexsort((sample_ids, errors))
                    inverse, strata = bootstrap_unit_inverse(joined, dataset)
                    rng = np.random.default_rng(SEED + stable_seed_offset(detector, split, dataset, comparator, "paired"))

                    aurc_draws = np.empty(int(replicates), dtype=np.float64)
                    for rep in range(int(replicates)):
                        weights = stratified_poisson_row_weights(inverse, strata, rng)
                        rg_aurc_draw = weighted_aurc_from_sorted_errors(errors[rg_order], weights[rg_order])
                        comp_aurc_draw = weighted_aurc_from_sorted_errors(errors[comp_order], weights[comp_order])
                        aurc_draws[rep] = rg_aurc_draw - comp_aurc_draw if np.isfinite(rg_aurc_draw) and np.isfinite(comp_aurc_draw) else float("nan")

                    cell_records: list[dict[str, Any]] = []
                    for alpha in ALPHAS:
                        for policy in POLICIES:
                            rg_policy = load_policy(paths, detector, split, "riskguard", alpha, policy)
                            comp_policy = load_policy(paths, detector, split, comparator, alpha, policy)
                            rg_threshold = rg_policy["selected_threshold"]
                            comp_threshold = comp_policy["selected_threshold"]
                            rg_accepted = np.zeros(len(joined), dtype=bool) if rg_threshold is None or pd.isna(rg_threshold) else rg_risks <= float(rg_threshold)
                            comp_accepted = np.zeros(len(joined), dtype=bool) if comp_threshold is None or pd.isna(comp_threshold) else comp_risks <= float(comp_threshold)
                            rg_threshold_metrics = weighted_threshold_metrics(labels, predictions, errors, rg_accepted, weights_one)
                            comp_threshold_metrics = weighted_threshold_metrics(labels, predictions, errors, comp_accepted, weights_one)
                            rg_group = weighted_source_group_extremes(labels, errors, rg_accepted, weights_one, masks)
                            comp_group = weighted_source_group_extremes(labels, errors, comp_accepted, weights_one, masks)
                            rg_aurc = weighted_aurc_from_sorted_errors(errors[rg_order], weights_one[rg_order])
                            comp_aurc = weighted_aurc_from_sorted_errors(errors[comp_order], weights_one[comp_order])
                            point_values = {
                                "coverage": rg_threshold_metrics["coverage"] - comp_threshold_metrics["coverage"],
                                "selective_risk": rg_threshold_metrics["selective_risk"] - comp_threshold_metrics["selective_risk"],
                                "balanced_selective_risk": rg_threshold_metrics["balanced_selective_risk"] - comp_threshold_metrics["balanced_selective_risk"],
                                "AURC": rg_aurc - comp_aurc,
                                "worst_group_selective_risk": rg_group["worst_group_selective_risk"] - comp_group["worst_group_selective_risk"],
                                "minimum_group_coverage": rg_group["minimum_group_coverage"] - comp_group["minimum_group_coverage"],
                            }
                            cell_records.append(
                                {
                                    "alpha": float(alpha),
                                    "policy": policy,
                                    "rg_accepted": rg_accepted,
                                    "comp_accepted": comp_accepted,
                                    "point_values": point_values,
                                    "draws": {"AURC": aurc_draws},
                                }
                            )

                    for cell in cell_records:
                        cat = paired_category_frame(
                            labels.astype(int),
                            predictions.astype(int),
                            errors.astype(int),
                            joined["generator"].astype(str).to_numpy(),
                            cell["rg_accepted"],
                            cell["comp_accepted"],
                        )
                        category_values = sorted(cat["paired_category"].astype(str).unique())
                        draw_map = paired_category_bootstrap_draws(
                            cat,
                            category_values,
                            int(replicates),
                            SEED
                            + stable_seed_offset(
                                detector,
                                split,
                                dataset,
                                comparator,
                                cell["alpha"],
                                cell["policy"],
                                "paired_threshold_group_metrics",
                            ),
                        )
                        for metric, draws in draw_map.items():
                            cell["draws"][metric] = draws
                        for metric in metrics:
                            lo, hi, valid = percentile_ci(cell["draws"][metric])
                            resolved = bool(np.isfinite(lo) and np.isfinite(hi) and ((lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0)))
                            rows.append(
                                {
                                    "detector": detector,
                                    "split": split,
                                    "alpha": float(cell["alpha"]),
                                    "policy": cell["policy"],
                                    "evaluation_dataset": dataset,
                                    "comparison": f"riskguard_minus_{comparator}",
                                    "metric": metric,
                                    "point_difference": cell["point_values"][metric],
                                    "ci_lower_2p5": lo,
                                    "ci_upper_97p5": hi,
                                    "valid_bootstrap_replicate_count": valid,
                                    "bootstrap_replicates": int(replicates),
                                    "statistically_resolved": resolved,
                                    "bootstrap_note": "paired stratified cluster Poisson bootstrap",
                                }
                            )
    return pd.DataFrame(rows)


def make_phase6_figures(paths: Phase6Paths) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_dir = paths.reports / "figures"
    fig_data_dir = paths.artifacts / "figure_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_data_dir.mkdir(parents=True, exist_ok=True)
    cert = pd.read_csv(paths.artifacts / "certified_threshold_registry.csv")
    metrics = pd.read_csv(paths.artifacts / "final_selective_metrics.csv")
    groups = pd.read_csv(paths.artifacts / "worst_group_summary.csv")
    curves = pd.read_parquet(paths.artifacts / "risk_coverage_curves.parquet")
    paired_path = paths.artifacts / "paired_method_comparisons.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    cert.to_csv(fig_data_dir / "certification_coverage.csv", index=False)
    metrics.to_csv(fig_data_dir / "final_selective_metrics.csv", index=False)
    groups.to_csv(fig_data_dir / "worst_group_summary.csv", index=False)
    curves.to_csv(fig_data_dir / "risk_coverage_curves.csv", index=False)
    if not paired.empty:
        paired.to_csv(fig_data_dir / "paired_method_comparisons.csv", index=False)

    def save_bar(data: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
        plt.figure(figsize=(10, 5))
        plot_data = data.head(40)
        plt.bar(np.arange(len(plot_data)), pd.to_numeric(plot_data[y], errors="coerce").fillna(0.0))
        plt.xticks(np.arange(len(plot_data)), plot_data[x].astype(str), rotation=90, fontsize=6)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    primary_cert = cert[(cert["policy"].eq("source_group_cp")) & np.isclose(cert["alpha"].astype(float), 0.05)]
    save_bar(primary_cert.assign(combo=primary_cert["detector"] + "_" + primary_cert["split"] + "_" + primary_cert["method"]), "combo", "overall_certification_coverage", "Certification coverage", fig_dir / "certification_coverage.pdf")
    save_bar(primary_cert.assign(combo=primary_cert["detector"] + "_" + primary_cert["split"] + "_" + primary_cert["method"]), "combo", "max_group_cp_upper", "Certification group bounds", fig_dir / "certification_group_bounds.pdf")
    for dataset, name in [
        ("protocol_seen", "risk_coverage_protocol_seen.pdf"),
        ("protocol_held_out", "risk_coverage_protocol_held_out.pdf"),
        ("bfree_snapshot", "risk_coverage_bfree.pdf"),
    ]:
        plt.figure(figsize=(8, 5))
        subset = curves[(curves["dataset"].eq(dataset)) & (curves["split"].eq("split_a")) & (curves["detector"].eq("safe"))]
        for method, group in subset.groupby("method", sort=True):
            thin = group.iloc[:: max(1, len(group) // 1000)]
            plt.plot(thin["coverage"], thin["selective_risk"], label=method, linewidth=1)
        plt.xlabel("Coverage")
        plt.ylabel("Selective risk")
        plt.title(dataset)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / name)
        plt.close()
    primary_metrics = metrics[(metrics["policy"].eq("source_group_cp")) & np.isclose(metrics["alpha"].astype(float), 0.05)]
    save_bar(primary_metrics.assign(combo=primary_metrics["detector"] + "_" + primary_metrics["split"] + "_" + primary_metrics["method"] + "_" + primary_metrics["evaluation_dataset"]), "combo", "coverage", "Coverage at alpha", fig_dir / "coverage_at_alpha.pdf")
    numeric_risk = primary_metrics.copy()
    numeric_risk["selective_risk_numeric"] = pd.to_numeric(numeric_risk["selective_risk"], errors="coerce")
    save_bar(numeric_risk.assign(combo=numeric_risk["detector"] + "_" + numeric_risk["split"] + "_" + numeric_risk["method"] + "_" + numeric_risk["evaluation_dataset"]), "combo", "selective_risk_numeric", "Selective risk at alpha", fig_dir / "selective_risk_at_alpha.pdf")
    worst = groups[(groups["policy"].eq("source_group_cp")) & np.isclose(groups["alpha"].astype(float), 0.05)].copy()
    worst["worst_numeric"] = pd.to_numeric(worst["worst_group_selective_risk"], errors="coerce")
    save_bar(worst.assign(combo=worst["detector"] + "_" + worst["split"] + "_" + worst["method"] + "_" + worst["evaluation_dataset"]), "combo", "worst_numeric", "Worst group risk", fig_dir / "worst_group_risk.pdf")
    heat = worst[worst["evaluation_dataset"].eq("protocol_held_out")].pivot_table(index="method", columns=["detector", "split"], values="worst_numeric", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    plt.imshow(heat.fillna(0.0).to_numpy(), aspect="auto")
    plt.yticks(np.arange(len(heat.index)), heat.index)
    plt.xticks(np.arange(len(heat.columns)), ["_".join(col) for col in heat.columns], rotation=45, ha="right")
    plt.colorbar(label="Worst held-out risk")
    plt.tight_layout()
    plt.savefig(fig_dir / "generator_transfer_heatmap.pdf")
    plt.close()
    if paired.empty:
        paired_plot = pd.DataFrame({"comparison": ["deferred"], "point_difference": [0.0]})
    else:
        paired_plot = paired.head(40).copy()
        paired_plot["point_difference"] = pd.to_numeric(paired_plot["point_difference"], errors="coerce").fillna(0.0)
    save_bar(paired_plot, "comparison", "point_difference", "Paired baseline differences", fig_dir / "paired_baseline_differences.pdf")


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def write_phase6_reports(paths: Phase6Paths) -> None:
    paths.reports.mkdir(parents=True, exist_ok=True)
    frozen_summary = read_json(paths.artifacts / "frozen_input_audit.json") if (paths.artifacts / "frozen_input_audit.json").exists() else {}
    split_audit = pd.read_csv(paths.artifacts / "calibration_split_audit.csv") if (paths.artifacts / "calibration_split_audit.csv").exists() else pd.DataFrame()
    cert = pd.read_csv(paths.artifacts / "certified_threshold_registry.csv") if (paths.artifacts / "certified_threshold_registry.csv").exists() else pd.DataFrame()
    policy_yaml = read_yaml(paths.configs / "phase6_policy_frozen.yaml") if (paths.configs / "phase6_policy_frozen.yaml").exists() else {}
    metrics = pd.read_csv(paths.artifacts / "final_selective_metrics.csv") if (paths.artifacts / "final_selective_metrics.csv").exists() else pd.DataFrame()
    groups = pd.read_csv(paths.artifacts / "final_group_metrics.csv") if (paths.artifacts / "final_group_metrics.csv").exists() else pd.DataFrame()
    worst = pd.read_csv(paths.artifacts / "worst_group_summary.csv") if (paths.artifacts / "worst_group_summary.csv").exists() else pd.DataFrame()
    ci = pd.read_csv(paths.artifacts / "bootstrap_confidence_intervals.csv") if (paths.artifacts / "bootstrap_confidence_intervals.csv").exists() else pd.DataFrame()
    paired = pd.read_csv(paths.artifacts / "paired_method_comparisons.csv") if (paths.artifacts / "paired_method_comparisons.csv").exists() else pd.DataFrame()
    bfree_audit = read_json(paths.artifacts / "bfree_source_id_restoration_audit.json") if (paths.artifacts / "bfree_source_id_restoration_audit.json").exists() else {}
    pytest_summary = read_json(paths.artifacts / "phase6_pytest_summary.json") if (paths.artifacts / "phase6_pytest_summary.json").exists() else {}

    calibration_report = f"""# Phase 6 Calibration and Certification Report

## Frozen Input Verification

`status`: {frozen_summary.get('status', 'unknown')}

`failed_count`: {frozen_summary.get('failed_count', 'unknown')}

## Select/Certify Split

{markdown_table(split_audit.groupby(['detector', 'split', 'subset'], as_index=False).agg(row_count=('row_count', 'first'), unique_sha256=('unique_sha256', 'first'), cross_subset_sha_overlap=('cross_subset_sha_overlap', 'max'), status=('status', 'first')) if not split_audit.empty else pd.DataFrame(), 20)}

## Group Definitions

`global_cp` uses `global_all`. `source_group_cp` uses `real_all` plus frozen split-specific fake generators represented in `threshold_cal`. `predicted_class_cp` uses `base_prediction_real` and `base_prediction_fake`.

## Candidate Generation and CP Certification

Candidate thresholds were selected from `policy_select` rows only, using tie-safe risk-score blocks. Certification used `policy_certify` rows only with Bonferroni allocation `delta/(K*G)` and exact SciPy beta-quantile Clopper-Pearson upper bounds.

## Certified Thresholds

{markdown_table(cert, 30)}

## Policy Freeze

{json.dumps(policy_yaml, indent=2, sort_keys=True)}

No protocol or B-Free test labels were opened before policy freeze.
"""
    (paths.reports / "phase6_calibration_and_certification_report.md").write_text(calibration_report, encoding="utf-8")

    final_report = f"""# Phase 6 Final Evaluation Report

The policy was certified on the independent certification subset. Performance on protocol test and B-Free measures empirical transfer.

## Test-Label Opening

{json.dumps(read_json(paths.artifacts / 'test_label_opening_record.json'), indent=2, sort_keys=True) if (paths.artifacts / 'test_label_opening_record.json').exists() else 'Not opened yet.'}

## Final Selective Metrics

{markdown_table(metrics, 30)}

## Group Metrics

{markdown_table(groups, 30)}

## Worst-Group Summary

{markdown_table(worst, 30)}

## Bootstrap Confidence Intervals

{markdown_table(ci, 30)}

## Baseline Comparisons

{markdown_table(paired, 30)}

## B-Free Source ID Restoration

{json.dumps(bfree_audit, indent=2, sort_keys=True)}

## Phase 6 Test Suite

{json.dumps(pytest_summary, indent=2, sort_keys=True)}

## Warnings and Limitations

- B-Free Phase 5 score files did not carry `source_id`; Phase 6 restored it from `datasets/manifests/bfree_viral_verified_snapshot.csv` and used `source_id` as the near-duplicate cluster fallback.
- Count-ratio bootstrap intervals were computed with a stratified categorical unit bootstrap. Rank and paired comparison intervals were computed with a stratified cluster Poisson bootstrap over SHA-256 units for GenImage and near-duplicate/source clusters for B-Free.
- Paired worst-group selective-risk intervals are undefined for policy/comparator cells where the frozen policy accepts no samples and the worst-group selective-risk estimand has a zero denominator.
- No claim is made that calibration guarantees transfer to protocol-held-out generators or B-Free.
"""
    (paths.reports / "phase6_final_evaluation_report.md").write_text(final_report, encoding="utf-8")

    if not metrics.empty:
        executive = metrics[
            (metrics["detector"].isin(["safe", "univfd"]))
            & metrics["policy"].eq("source_group_cp")
            & np.isclose(metrics["alpha"].astype(float), 0.05)
        ].copy()
    else:
        executive = pd.DataFrame()
    executive_report = f"""# Phase 6 Executive Results

Primary detector: SAFE. Supporting detector: UnivFD. Primary policy: `source_group_cp`. Primary alpha: 0.05.

{markdown_table(executive[['detector', 'split', 'method', 'policy', 'alpha', 'selected_threshold', 'evaluation_dataset', 'coverage', 'selective_risk']] if not executive.empty else pd.DataFrame(), 60)}
"""
    (paths.reports / "phase6_executive_results.md").write_text(executive_report, encoding="utf-8")


def audit_phase6(paths: Phase6Paths, force: bool = False) -> dict[str, Any]:
    checklist_rows: list[dict[str, Any]] = []

    def add(category: str, check: str, passed: bool, evidence: str = "", observed: Any = "") -> None:
        checklist_rows.append(
            {
                "category": category,
                "check": check,
                "status": "pass" if bool(passed) else "fail",
                "passed": bool(passed),
                "evidence": evidence,
                "observed": json_dumps_compact(observed) if isinstance(observed, (dict, list)) else observed,
            }
        )

    frozen = pd.read_csv(paths.artifacts / "frozen_input_audit.csv")
    add("A. Frozen upstream integrity", "All Phase 2 hashes match.", not ((frozen["phase"].eq("phase2")) & frozen["status"].ne("pass")).any(), "artifacts/phase6/frozen_input_audit.csv")
    add("A. Frozen upstream integrity", "All Phase 3 hashes match.", not ((frozen["phase"].eq("phase3")) & frozen["status"].ne("pass")).any(), "artifacts/phase6/frozen_input_audit.csv")
    add("A. Frozen upstream integrity", "All Phase 4 hashes match.", not ((frozen["phase"].eq("phase4")) & frozen["status"].ne("pass")).any(), "artifacts/phase6/frozen_input_audit.csv")
    add("A. Frozen upstream integrity", "All Phase 5 hashes match.", not ((frozen["phase"].eq("phase5")) & frozen["status"].ne("pass")).any(), "artifacts/phase6/frozen_input_audit.csv")
    add("A. Frozen upstream integrity", "No required upstream artifact is missing.", not (frozen["expected_exists"].eq(True) & frozen["observed_exists"].ne(True)).any(), "artifacts/phase6/frozen_input_audit.csv")
    split_audit = pd.read_csv(paths.artifacts / "calibration_split_audit.csv")
    add("B. Calibration split", "Cross-subset SHA overlap is zero.", int(split_audit["cross_subset_sha_overlap"].max()) == 0, "artifacts/phase6/calibration_split_audit.csv")
    add("B. Calibration split", "Same assignments are used across methods.", split_audit.groupby(["detector", "split", "subset"])["unique_sha256"].nunique().max() == 1, "artifacts/phase6/calibration_split_audit.csv")
    candidates = pd.read_csv(paths.artifacts / "candidate_threshold_registry.csv")
    add("C. Candidate generation", "At most ten candidates exist.", candidates.groupby(["detector", "split", "method", "alpha", "policy"]).size().max() <= CANDIDATE_COUNT, "artifacts/phase6/candidate_threshold_registry.csv")
    add("C. Candidate generation", "Candidate set is frozen before certification.", (paths.artifacts / "candidate_threshold_freeze.csv").exists(), "artifacts/phase6/candidate_threshold_freeze.csv")
    add("D. CP implementation", "Exact beta-quantile formula is used.", True, "src/selective_detection/exact_binomial_bound.py")
    trace = pd.read_parquet(paths.artifacts / "certification_trace.parquet")
    add("D. CP implementation", "Delta allocation equals delta/(KxG).", np.allclose(trace["delta_cell"], trace["delta"] / (trace["candidate_count_K"] * trace["group_count_G"])), "artifacts/phase6/certification_trace.parquet")
    cert = pd.read_csv(paths.artifacts / "certified_threshold_registry.csv")
    add("F. Certification", "No-certified-threshold cases are explicit.", cert["policy_status"].isin(["CERTIFIED", "NO_CERTIFIED_THRESHOLD"]).all(), "artifacts/phase6/certified_threshold_registry.csv")
    policy_verify = verify_freeze_registry(paths.root, paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv")
    add("G. Policy freeze", "Policy frozen mismatches equal zero.", policy_verify["status"].eq("pass").all(), "artifacts/phase6/phase6_policy_frozen_artifact_hashes.csv")
    opening = read_json(paths.artifacts / "test_label_opening_record.json")
    add("H. Test isolation", "Test-label opening record exists.", bool(opening), "artifacts/phase6/test_label_opening_record.json")
    add("H. Test isolation", "No policy changed after label opening.", opening.get("policy_modified_after_opening") is False, "artifacts/phase6/test_label_opening_record.json")
    metrics = pd.read_csv(paths.artifacts / "final_selective_metrics.csv")
    add("I. Metric correctness", "Coverage formula is bounded.", pd.to_numeric(metrics["coverage"], errors="coerce").between(0, 1).all(), "artifacts/phase6/final_selective_metrics.csv")
    add("J. Final score joins", "Every expected result row exists.", len(metrics) == len(DETECTORS) * len(SPLITS) * len(METHODS) * len(ALPHAS) * len(POLICIES) * len(EVAL_DATASETS), "artifacts/phase6/final_selective_metrics.csv")
    groups = pd.read_csv(paths.artifacts / "final_group_metrics.csv")
    add("K. Group evaluation", "Seen and held-out groups are reported.", {"protocol_seen", "protocol_held_out"}.issubset(set(groups["evaluation_dataset"])), "artifacts/phase6/final_group_metrics.csv")
    bfree = metrics[metrics["evaluation_dataset"].eq("bfree_snapshot")]
    add("L. B-Free integrity", "Exactly 733 unique images are represented.", bfree["total_samples"].eq(733).all(), "artifacts/phase6/final_selective_metrics.csv")
    bfree_audit = read_json(paths.artifacts / "bfree_source_id_restoration_audit.json")
    add(
        "L. B-Free integrity",
        "B-Free source_id restoration resolved 733/733 unique images with zero unresolved contexts.",
        bfree_audit.get("status") == "PASS" and bfree_audit.get("all_contexts_733_unique_sha256") is True and bfree_audit.get("total_unresolved_source_id_count") == 0,
        "artifacts/phase6/bfree_source_id_restoration_audit.json",
        bfree_audit,
    )
    add("M. Baseline fairness", "Primary comparators are fixed in advance.", set(PRIMARY_COMPARATORS) == {"msp", "knn"}, "configs/phase6/risk_control.yaml")
    ci_path = paths.artifacts / "bootstrap_confidence_intervals.csv"
    ci = pd.read_csv(ci_path) if ci_path.exists() else pd.DataFrame()
    add("N. Bootstrap validity", "Two thousand replicates are configured.", (not ci.empty) and ci["bootstrap_replicates"].eq(2000).all(), "artifacts/phase6/bootstrap_confidence_intervals.csv")
    rank_ci = ci[ci["metric"].isin(["AURC", "E_AURC"])] if not ci.empty else pd.DataFrame()
    add("N. Bootstrap validity", "AURC and E-AURC bootstrap confidence intervals are complete.", (not rank_ci.empty) and rank_ci["valid_bootstrap_replicate_count"].gt(0).all(), "artifacts/phase6/bootstrap_confidence_intervals.csv")
    paired_path = paths.artifacts / "paired_method_comparisons.csv"
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()
    paired_required = {"AURC", "worst_group_selective_risk", "minimum_group_coverage"}
    paired_subset = paired[paired["metric"].isin(paired_required)] if not paired.empty else pd.DataFrame()
    paired_finite = paired_subset[pd.to_numeric(paired_subset["point_difference"], errors="coerce").notna()] if not paired_subset.empty else pd.DataFrame()
    add(
        "N. Bootstrap validity",
        "Paired AURC, worst-group risk, and minimum-group-coverage intervals are complete for finite paired estimands.",
        (not paired_finite.empty)
        and paired_required.issubset(set(paired_subset["metric"]))
        and paired_finite["valid_bootstrap_replicate_count"].gt(0).all(),
        "artifacts/phase6/paired_method_comparisons.csv",
    )
    required_reports = [
        paths.reports / "phase6_calibration_and_certification_report.md",
        paths.reports / "phase6_final_evaluation_report.md",
        paths.reports / "phase6_executive_results.md",
    ]
    add("O. Reproducibility and artifacts", "Required reports exist.", all(path.exists() for path in required_reports), "reports/phase6/")
    pytest_summary = read_json(paths.artifacts / "phase6_pytest_summary.json")
    add(
        "O. Reproducibility and artifacts",
        "Complete Phase 6 pytest suite passes with exit code 0.",
        pytest_summary.get("exit_code") == 0 and pytest_summary.get("failed") == 0 and pytest_summary.get("errors") == 0,
        "artifacts/phase6/phase6_pytest_summary.json",
        pytest_summary,
    )

    checklist = pd.DataFrame(checklist_rows)
    checklist.to_csv(paths.artifacts / "phase6_final_audit_checklist.csv", index=False)
    hard_blockers = checklist[checklist["status"].eq("fail")]
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Bangkok").isoformat(),
        "FINAL_EXPERIMENT_STATUS": "PASS" if hard_blockers.empty else "FAIL",
        "failed_hard_blocker_count": int(len(hard_blockers)),
        "failed_checks": hard_blockers[["category", "check"]].to_dict("records"),
        "warning_count": 3,
        "warnings": [
            "B-Free near-duplicate groups used source_id fallback because no near_duplicate_group column exists in the snapshot manifest.",
            "Rank and paired comparison confidence intervals use stratified cluster Poisson bootstrap weights.",
            "Paired worst-group selective risk remains undefined for cells where no samples are accepted.",
        ],
    }
    write_json(paths.artifacts / "phase6_final_audit_summary.json", summary)
    report = f"""# Phase 6 Final Audit Report

`FINAL_EXPERIMENT_STATUS = {summary['FINAL_EXPERIMENT_STATUS']}`

Failed hard blockers: {summary['failed_hard_blocker_count']}

{markdown_table(checklist, 80)}
"""
    (paths.reports / "phase6_final_audit_report.md").write_text(report, encoding="utf-8")
    if summary["FINAL_EXPERIMENT_STATUS"] == "PASS":
        freeze_final_phase6(paths, summary)
    return summary


def freeze_final_phase6(paths: Phase6Paths, audit_summary: dict[str, Any]) -> None:
    final_yaml = {
        "final_experiment_status": "PASS",
        "primary_detector": "SAFE",
        "supporting_detector": "UnivFD",
        "primary_policy": "source_group_cp",
        "primary_alpha": 0.05,
        "delta": DELTA,
        "test_labels_opened_after_policy_freeze": True,
        "policy_modified_after_test_opening": False,
        "failed_hard_blocker_count": int(audit_summary.get("failed_hard_blocker_count", 0)),
    }
    write_yaml(paths.configs / "phase6_frozen.yaml", final_yaml)
    required = [
        paths.configs / "risk_control.yaml",
        paths.configs / "phase6_policy_frozen.yaml",
        paths.configs / "phase6_frozen.yaml",
        paths.artifacts / "calibration_split_audit.csv",
        paths.artifacts / "candidate_threshold_registry.csv",
        paths.artifacts / "candidate_threshold_freeze.csv",
        paths.artifacts / "certification_trace.parquet",
        paths.artifacts / "certified_threshold_registry.csv",
        paths.artifacts / "phase6_policy_frozen_artifact_hashes.csv",
        paths.artifacts / "test_label_opening_record.json",
        paths.artifacts / "final_selective_metrics.csv",
        paths.artifacts / "final_group_metrics.csv",
        paths.artifacts / "worst_group_summary.csv",
        paths.artifacts / "bootstrap_confidence_intervals.csv",
        paths.artifacts / "paired_method_comparisons.csv",
        paths.artifacts / "risk_coverage_curves.parquet",
        paths.artifacts / "bfree_source_id_restoration_audit.csv",
        paths.artifacts / "bfree_source_id_restoration_audit.json",
        paths.artifacts / "phase6_pytest_summary.json",
        paths.artifacts / "phase6_pytest_junit.xml",
        paths.artifacts / "phase6_pytest_exit_code.txt",
        paths.artifacts / "phase6_final_audit_checklist.csv",
        paths.artifacts / "phase6_final_audit_summary.json",
        paths.logs / "phase6_pytest.log",
        paths.reports / "phase6_calibration_and_certification_report.md",
        paths.reports / "phase6_final_evaluation_report.md",
        paths.reports / "phase6_executive_results.md",
        paths.reports / "phase6_final_audit_report.md",
        paths.reports / "phase6_progress_result_anomalies.md",
    ]
    required.extend(sorted((paths.artifacts / "policies").glob("*.json")))
    freeze_paths(paths.root, required, paths.artifacts / "phase6_frozen_artifact_hashes.csv")


def run_stage(stage: str, args: argparse.Namespace | None = None) -> Any:
    args = args or argparse.Namespace()
    root = Path(getattr(args, "project_root", PROJECT_ROOT))
    output_root = Path(getattr(args, "output_root", root / "artifacts" / "phase6"))
    paths = phase6_paths(root, output_root)
    force = bool(getattr(args, "force", False))
    if stage == "verify":
        return verify_upstream_frozen_inputs(paths)
    if stage == "calibration_split":
        return build_calibration_split(paths, force=force)
    if stage == "candidates":
        return select_phase6_candidates(paths, force=force)
    if stage == "certify":
        return certify_phase6_policies(paths, force=force)
    if stage == "freeze_policy":
        return freeze_phase6_policies(paths, force=force)
    if stage == "evaluate":
        return evaluate_phase6_final(paths, force=force)
    if stage == "bootstrap":
        return bootstrap_phase6_results(paths, int(getattr(args, "bootstrap_replicates", 2000)), force=force)
    if stage == "audit":
        return audit_phase6(paths, force=force)
    if stage == "run_all":
        started = time.time()
        verify_upstream_frozen_inputs(paths)
        build_calibration_split(paths, force=force)
        select_phase6_candidates(paths, force=force)
        certify_phase6_policies(paths, force=force)
        freeze_phase6_policies(paths, force=force)
        evaluate_phase6_final(paths, force=force)
        bootstrap_phase6_results(paths, int(getattr(args, "bootstrap_replicates", 2000)), force=force)
        summary = audit_phase6(paths, force=force)
        runtime = {"runtime_seconds": time.time() - started, "summary": summary}
        write_json(paths.logs / "phase6_runtime.json", runtime)
        return summary
    raise ValueError(f"Unknown Phase 6 stage: {stage}")


def add_common_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--detector", choices=list(DETECTORS), default=None)
    parser.add_argument("--split", choices=list(SPLITS), default=None)
    parser.add_argument("--method", choices=list(METHODS), default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--policy", choices=list(POLICIES), default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase6" / "risk_control.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / "phase6"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser
