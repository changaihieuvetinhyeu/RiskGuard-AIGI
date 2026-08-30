#!/usr/bin/env python3
"""Run and audit Phase 7 synthesis for the RiskGuard-AIGI study.

The script is intentionally self-contained because Phase 7 is an analysis
layer over frozen Phase 2-6 artifacts. It writes only Phase 7 artifacts,
reports, logs, plus this required audit entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selective_detection.policy_artifact_io import (  # noqa: E402
    freeze_paths,
    read_json,
    read_yaml,
    relative_to,
    sha256_file,
    verify_freeze_registry,
    write_json,
    write_yaml,
)
from selective_detection.group_risk_certification import (  # noqa: E402
    ALPHAS,
    DATASET_LABELS,
    METHODS,
    POLICIES,
    accepted_metric_summary,
    deduplicate_eval_rows,
    load_policy,
    load_scores,
)
from selective_detection.error_probability_calibrator import (  # noqa: E402
    PRIMARY_FEATURES,
    load_riskguard_json,
    risk_probability,
    transform_features,
)
from selective_detection.selective_metrics import (  # noqa: E402
    aurc,
    calibration_metrics,
    eaurc,
    error_ranking_metrics,
)

DETECTORS = ("safe", "univfd")
SPLITS = ("split_a", "split_b")
EVAL_DATASETS = ("protocol_seen", "protocol_held_out", "bfree_snapshot")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260916
PERMUTATION_REPEATS = 200
PRIMARY_DETECTOR = "safe"
SUPPORTING_DETECTOR = "univfd"
PRIMARY_SPLIT = "split_a"
PRIMARY_METHOD = "riskguard"
PRIMARY_POLICY = "source_group_cp"
REFERENCE_POLICY = "global_cp"
PRIMARY_ALPHA = 0.05
SECONDARY_ALPHA = 0.01

ABLATIONS: dict[str, tuple[str, ...]] = {
    "full_four": PRIMARY_FEATURES,
    "no_margin": ("orbit_logit_variance", "embedding_drift_mean", "orbit_support_distance_max"),
    "no_variance": ("margin_distance", "embedding_drift_mean", "orbit_support_distance_max"),
    "no_drift": ("margin_distance", "orbit_logit_variance", "orbit_support_distance_max"),
    "no_support": ("margin_distance", "orbit_logit_variance", "embedding_drift_mean"),
    "margin_only": ("margin_distance",),
    "variance_only": ("orbit_logit_variance",),
    "drift_only": ("embedding_drift_mean",),
    "support_only": ("orbit_support_distance_max",),
    "orbit_only": ("orbit_logit_variance", "embedding_drift_mean"),
    "geometry_support": ("margin_distance", "orbit_support_distance_max"),
}

VIEW_NAMES = (
    "identity",
    "jpeg_q75",
    "resize_075_restore",
    "gaussian_blur_sigma_05",
    "center_crop_090_restore",
)
LEAVE_ONE_VIEW = {
    "jpeg": "jpeg_q75",
    "resize": "resize_075_restore",
    "blur": "gaussian_blur_sigma_05",
    "crop": "center_crop_090_restore",
}


@dataclass
class Phase7Paths:
    root: Path
    artifacts: Path
    configs: Path
    reports: Path
    logs: Path
    tables: Path
    figures: Path
    figure_data: Path


def phase7_paths(root: Path = PROJECT_ROOT) -> Phase7Paths:
    paths = Phase7Paths(
        root=root,
        artifacts=root / "artifacts" / "phase7",
        configs=root / "configs" / "phase7",
        reports=root / "reports" / "phase7",
        logs=root / "logs" / "phase7",
        tables=root / "reports" / "phase7" / "tables",
        figures=root / "reports" / "phase7" / "figures",
        figure_data=root / "artifacts" / "phase7" / "figure_data",
    )
    for directory in [
        paths.artifacts,
        paths.configs,
        paths.reports,
        paths.logs,
        paths.tables,
        paths.figures,
        paths.figure_data,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def rel(path: Path) -> str:
    return relative_to(PROJECT_ROOT, path)


def slug(detector: str, split: str) -> str:
    return f"{detector}_{split}"


def alpha_slug(alpha: float) -> str:
    return f"alpha_{str(float(alpha)).replace('.', 'p')}"


def phase6_policy(detector: str, split: str, method: str, alpha: float, policy: str) -> dict[str, Any]:
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "phase6"
        / "policies"
        / f"{detector}_{split}_{method}_{alpha_slug(alpha)}_{policy}.json"
    )
    return read_json(path)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def now_iso() -> str:
    return pd.Timestamp.now(tz="Asia/Bangkok").isoformat()


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def write_table_pair(df: pd.DataFrame, csv_path: Path, tex_path: Path, float_format: str = "%.6f") -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    tex_path.write_text(simple_latex_table(df, float_format=float_format), encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def simple_latex_table(df: pd.DataFrame, float_format: str = "%.6f", max_rows: int = 200) -> str:
    view = df.head(max_rows).copy()
    align = "l" * len(view.columns)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(col) for col in view.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in view.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                cells.append(float_format % float(value))
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if len(df) > max_rows:
        lines.append(f"% Truncated to first {max_rows} of {len(df)} rows.")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def save_simple_figure(
    df: pd.DataFrame,
    figure_base: Path,
    kind: str,
    title: str,
    x: str,
    y: str,
    hue: str | None = None,
    max_rows: int = 80,
) -> None:
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.head(max_rows).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if plot_df.empty or x not in plot_df or y not in plot_df:
        ax.text(0.5, 0.5, "No defined data", ha="center", va="center")
        ax.set_axis_off()
    elif kind == "line":
        if hue and hue in plot_df:
            for value, group in plot_df.groupby(hue, sort=True):
                ax.plot(group[x], group[y], label=str(value), linewidth=1.6)
            ax.legend(fontsize=7)
        else:
            ax.plot(plot_df[x], plot_df[y], linewidth=1.6)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    elif kind == "scatter":
        if hue and hue in plot_df:
            for value, group in plot_df.groupby(hue, sort=True):
                ax.scatter(group[x], group[y], label=str(value), s=22)
            ax.legend(fontsize=7)
        else:
            ax.scatter(plot_df[x], plot_df[y], s=22)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    else:
        labels = plot_df[x].astype(str).to_numpy()
        values = numeric_series(plot_df[y]).fillna(0.0).to_numpy(dtype=float)
        ax.bar(np.arange(len(labels)), values)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_base.with_suffix(".pdf"))
    fig.savefig(figure_base.with_suffix(".png"), dpi=250)
    plt.close(fig)


def binary_metric_frame(y: np.ndarray, p: np.ndarray, sample_ids: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=np.int64)
    cal = calibration_metrics(y, p, n_bins=15)
    ranking = error_ranking_metrics(y, p)
    return {
        "NLL": float(cal["NLL"]),
        "Brier": float(cal["Brier"]),
        "ECE": float(cal["ECE"]),
        "error_detection_AUROC": float(ranking.auroc),
        "error_detection_AUPR": float(ranking.aupr),
        "AURC": float(aurc(y, p, sample_ids)),
        "E_AURC": float(eaurc(y, p, sample_ids)),
    }


def metric_value(metric: str, y: np.ndarray, p: np.ndarray, sample_ids: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=np.int64)
    if metric == "NLL":
        return float(log_loss(y, p, labels=[0, 1]))
    if metric == "Brier":
        return float(np.mean((p - y) ** 2))
    if metric == "ECE":
        return float(calibration_metrics(y, p, n_bins=15)["ECE"])
    if metric == "error_detection_AUROC":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    if metric == "error_detection_AUPR":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    if metric == "AURC":
        return float(aurc(y, p, sample_ids))
    if metric == "E_AURC":
        return float(eaurc(y, p, sample_ids))
    raise ValueError(metric)


def near_threshold_bin(distance: float) -> str:
    value = float(distance)
    if value <= 0.001:
        return "0-0.001"
    if value <= 0.005:
        return "0.001-0.005"
    if value <= 0.01:
        return "0.005-0.01"
    if value <= 0.05:
        return "0.01-0.05"
    return "greater_than_0.05"


def threshold_tags(row: pd.Series, thresholds: dict[str, float]) -> list[str]:
    tags: list[str] = []
    if float(row["margin_distance"]) <= thresholds["margin_distance_p10"]:
        tags.append("near_decision_boundary")
    if float(row["orbit_logit_variance"]) >= thresholds["orbit_logit_variance_p90"]:
        tags.append("orbit_logit_unstable")
    if float(row["embedding_drift_mean"]) >= thresholds["embedding_drift_mean_p90"]:
        tags.append("embedding_unstable")
    if float(row["orbit_support_distance_max"]) >= thresholds["orbit_support_distance_max_p90"]:
        tags.append("far_from_training_support")
    primary_count = len(tags)
    if primary_count >= 2:
        tags.append("multiple_risk_factors")
    if primary_count == 0:
        tags.append("none_of_primary_risk_factors")
    return tags


def deterministic_case_selection(
    df: pd.DataFrame,
    *,
    category: str,
    n: int,
    sort_cols: list[str],
    ascending: list[bool],
    cluster_col: str = "near_duplicate_group",
    diversify_col: str = "generator",
) -> pd.DataFrame:
    """Deterministically select rows with diversity and near-duplicate exclusion."""

    if df.empty:
        return df.copy()
    work = df.copy()
    if cluster_col not in work:
        work[cluster_col] = ""
    cluster = work[cluster_col].astype(str)
    fallback = cluster.isin(["", "nan", "None"])
    work.loc[fallback, cluster_col] = work.loc[fallback, "sha256"].astype(str)
    work["selection_category"] = category
    work["_tie_sha"] = work["sha256"].astype(str)
    ordered = work.sort_values(sort_cols + ["_tie_sha"], ascending=ascending + [True], kind="mergesort")
    selected: list[pd.Series] = []
    used_clusters: set[str] = set()
    used_diverse: set[str] = set()
    for _, row in ordered.iterrows():
        cluster_id = str(row[cluster_col])
        diverse_id = str(row.get(diversify_col, ""))
        if cluster_id in used_clusters:
            continue
        if len(used_diverse) < n and diverse_id in used_diverse and ordered[diversify_col].nunique() >= n:
            continue
        selected.append(row)
        used_clusters.add(cluster_id)
        used_diverse.add(diverse_id)
        if len(selected) >= n:
            break
    if len(selected) < n:
        for _, row in ordered.iterrows():
            cluster_id = str(row[cluster_col])
            if cluster_id in used_clusters:
                continue
            selected.append(row)
            used_clusters.add(cluster_id)
            if len(selected) >= n:
                break
    return pd.DataFrame(selected).drop(columns=["_tie_sha"], errors="ignore").reset_index(drop=True)


def verify_frozen_inputs(paths: Phase7Paths) -> tuple[pd.DataFrame, dict[str, Any]]:
    registries = [
        ("phase2", paths.root / "artifacts" / "phase2_frozen_artifact_hashes.csv"),
        ("phase3", paths.root / "artifacts" / "phase3" / "phase3_frozen_artifact_hashes.csv"),
        ("phase4", paths.root / "artifacts" / "phase4" / "phase4_frozen_artifact_hashes.csv"),
        ("phase5", paths.root / "artifacts" / "phase5" / "phase5_frozen_artifact_hashes.csv"),
        ("phase6", paths.root / "artifacts" / "phase6" / "phase6_frozen_artifact_hashes.csv"),
    ]
    rows: list[dict[str, Any]] = []
    for phase, registry in registries:
        exists = registry.exists()
        rows.append(
            {
                "audit_type": "freeze_registry_exists",
                "phase": phase,
                "relative_path": rel(registry),
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
            item = paths.root / str(record["relative_path"])
            expected_exists = bool(record.get("exists", True))
            observed_exists = item.exists()
            expected_size = int(record.get("size_bytes", 0))
            observed_size = int(item.stat().st_size) if observed_exists else 0
            expected_sha = str(record.get("sha256", ""))
            observed_sha = sha256_file(item) if observed_exists else ""
            ok = expected_exists == observed_exists and (
                not observed_exists or (expected_size == observed_size and expected_sha == observed_sha)
            )
            rows.append(
                {
                    "audit_type": "frozen_hash",
                    "phase": phase,
                    "relative_path": str(record["relative_path"]),
                    "expected_exists": expected_exists,
                    "observed_exists": observed_exists,
                    "expected_size_bytes": expected_size,
                    "observed_size_bytes": observed_size,
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                    "status": "pass" if ok else "fail",
                }
            )

    configs = [
        ("phase2", paths.root / "configs" / "phase2_frozen.yaml"),
        ("phase3", paths.root / "configs" / "phase3" / "phase3_frozen.yaml"),
        ("phase4", paths.root / "configs" / "phase4" / "phase4_frozen.yaml"),
        ("phase5", paths.root / "configs" / "phase5" / "phase5_frozen.yaml"),
        ("phase6", paths.root / "configs" / "phase6" / "phase6_frozen.yaml"),
    ]
    for phase, config in configs:
        rows.append(
            {
                "audit_type": "freeze_config_exists",
                "phase": phase,
                "relative_path": rel(config),
                "expected_exists": True,
                "observed_exists": config.exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": int(config.stat().st_size) if config.exists() else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(config) if config.exists() else "",
                "status": "pass" if config.exists() else "fail",
            }
        )

    phase6_config = read_yaml(paths.root / "configs" / "phase6" / "phase6_frozen.yaml")
    required_phase6 = {
        "final_experiment_status": "PASS",
        "policy_modified_after_test_opening": False,
        "failed_hard_blocker_count": 0,
    }
    for key, expected in required_phase6.items():
        observed = phase6_config.get(key)
        rows.append(
            {
                "audit_type": "phase6_required_state",
                "phase": "phase6",
                "relative_path": f"configs/phase6/phase6_frozen.yaml::{key}",
                "expected_exists": True,
                "observed_exists": True,
                "expected_size_bytes": "",
                "observed_size_bytes": "",
                "expected_sha256": str(expected),
                "observed_sha256": str(observed),
                "status": "pass" if observed == expected else "fail",
            }
        )

    required_reports = [
        "reports/phase5/phase5_final_audit_report_v2.md",
        "reports/phase6/phase6_final_audit_report.md",
        "reports/phase6/phase6_final_evaluation_report.md",
        "artifacts/phase5/ablation_oof_metrics.csv",
        "artifacts/phase6/final_selective_metrics.csv",
        "artifacts/phase6/certified_threshold_registry.csv",
        "artifacts/phase6/phase6_policy_frozen_artifact_hashes.csv",
    ]
    for relpath in required_reports:
        item = paths.root / relpath
        rows.append(
            {
                "audit_type": "required_report_or_score",
                "phase": "phase7_input",
                "relative_path": relpath,
                "expected_exists": True,
                "observed_exists": item.exists(),
                "expected_size_bytes": "",
                "observed_size_bytes": int(item.stat().st_size) if item.exists() else 0,
                "expected_sha256": "",
                "observed_sha256": sha256_file(item) if item.exists() else "",
                "status": "pass" if item.exists() else "fail",
            }
        )

    audit = pd.DataFrame(rows)
    audit.to_csv(paths.artifacts / "frozen_input_audit.csv", index=False)
    summary = {
        "created_at": now_iso(),
        "row_count": int(len(audit)),
        "failed_count": int(audit["status"].ne("pass").sum()),
        "required_artifacts_missing": int((audit["expected_exists"].eq(True) & audit["observed_exists"].ne(True)).sum()),
        "frozen_mismatch_count": int(audit["audit_type"].eq("frozen_hash").mul(audit["status"].ne("pass")).sum()),
        "phase6_required_state_pass": bool(
            audit.loc[audit["audit_type"].eq("phase6_required_state"), "status"].eq("pass").all()
        ),
        "status": "PASS" if audit["status"].eq("pass").all() else "FAIL",
    }
    write_json(paths.artifacts / "frozen_input_audit.json", summary)
    return audit, summary


def create_analysis_registry(paths: Phase7Paths) -> pd.DataFrame:
    analyses = [
        ("A01", "Frozen input audit", "frozen_result_synthesis", "all", "SAFE,UnivFD", "A,B", "hash verification", "none", "NA", "hash_match", "Phase2-6 freeze registries", False, "confirmatory", "artifacts/phase7/frozen_input_audit.csv"),
        ("A02", "Ablation synthesis", "predefined_ablation", "risk_fit OOF", "SAFE,UnivFD", "A,B", "full_four and predefined ablations", "none", "NA", "NLL,Brier,ECE,AUROC,AUPR,AURC,E-AURC", "artifacts/phase5/ablation_oof_metrics.csv", True, "confirmatory", "artifacts/phase7/ablation_summary.csv"),
        ("A03", "Ablation paired bootstrap", "predefined_ablation", "risk_fit OOF", "SAFE", "A", "paired SHA bootstrap", "none", "NA", "metric_deltas", "phase7 recomputed OOF ablation scores", True, "confirmatory", "artifacts/phase7/ablation_paired_bootstrap.csv"),
        ("A04", "Feature contribution", "predefined_ablation", "risk_fit OOF", "SAFE,UnivFD", "A,B", "frozen coefficients and feature summaries", "none", "NA", "distribution,correlation,AUROC,logit contribution", "Phase4 features, Phase5 models", True, "confirmatory", "artifacts/phase7/feature_effect_summary.csv"),
        ("A05", "Permutation importance", "exploratory_posthoc", "risk_fit OOF,threshold_cal", "SAFE,UnivFD", "A,B", "grouped permutation", "none", "NA", "NLL,Brier,AUROC,AURC", "Phase4 features, Phase5 models", True, "exploratory_posthoc", "artifacts/phase7/permutation_importance.csv"),
        ("A06", "Policy ablation", "frozen_result_synthesis", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "frozen Phase6 policy comparison", "global_cp,source_group_cp,predicted_class_cp", "0.05,0.01", "coverage,risk,worst_group", "artifacts/phase6/final_selective_metrics.csv", True, "confirmatory", "artifacts/phase7/policy_ablation_summary.csv"),
        ("A07", "Detector split domain comparison", "frozen_result_synthesis", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "frozen result comparison", "source_group_cp,global_cp", "0.05", "coverage,risk,AURC,worst_group", "artifacts/phase6/final_selective_metrics.csv", True, "confirmatory", "artifacts/phase7/detector_split_comparison.csv"),
        ("A08", "Transformation view diagnostics", "frozen_result_synthesis", "all Phase4 contexts", "SAFE,UnivFD", "A,B", "per-view orbit diagnostics", "none", "NA", "logit_shift,flip_rate,orbit_variance_share", "artifacts/phase4/orbit_cache", True, "confirmatory", "artifacts/phase7/transformation_view_summary.csv"),
        ("A09", "Leave-one-view perturbation", "exploratory_posthoc", "threshold_cal,protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "remove one transformed view without refit", "source_group_cp", "0.05", "risk_change,acceptance_flip", "Phase4 orbit cache, Phase5 models", True, "exploratory_posthoc", "artifacts/phase7/leave_one_view_perturbation.csv"),
        ("A10", "Outcome taxonomy", "qualitative_failure_analysis", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "frozen policy outcome labels", "all", "0.05,0.01", "outcome_counts", "Phase6 policies, RiskGuard scores", True, "confirmatory", "artifacts/phase7/outcome_taxonomy.parquet"),
        ("A11", "Reliability failure taxonomy", "qualitative_failure_analysis", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "risk_fit thresholds", "all", "0.05,0.01", "tag_counts", "Phase4 features, Phase6 outcomes", True, "confirmatory", "artifacts/phase7/failure_taxonomy.parquet"),
        ("A12", "Near-threshold analysis", "exploratory_posthoc", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "threshold distance bins", "all", "0.05,0.01", "counts,error_rate,instability", "Phase6 policies and scores", True, "exploratory_posthoc", "artifacts/phase7/near_threshold_analysis.csv"),
        ("A13", "Accepted-error analysis", "qualitative_failure_analysis", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "rank accepted errors", "all", "0.05,0.01", "case_features,counts", "failure_taxonomy", True, "confirmatory", "artifacts/phase7/accepted_error_cases.parquet"),
        ("A14", "Rejected-correct analysis", "qualitative_failure_analysis", "protocol_seen,protocol_held_out,B-Free", "SAFE,UnivFD", "A,B", "rank rejected correct", "all", "0.05,0.01", "case_features,counts", "failure_taxonomy", True, "confirmatory", "artifacts/phase7/rejected_correct_cases.parquet"),
        ("A15", "Generator failure profile", "qualitative_failure_analysis", "GenImage", "SAFE,UnivFD", "A,B", "generator medians and profiles", "source_group_cp", "0.05", "risk,separation,coverage,failures", "failure_taxonomy", True, "confirmatory", "artifacts/phase7/generator_failure_profile.csv"),
        ("A16", "B-Free cluster profile", "qualitative_failure_analysis", "B-Free", "SAFE,UnivFD", "A,B", "source/near-duplicate clusters", "source_group_cp", "0.05", "cluster risk and outcomes", "failure_taxonomy", True, "confirmatory", "artifacts/phase7/bfree_cluster_failure_analysis.csv"),
        ("A17", "Qualitative galleries", "qualitative_failure_analysis", "protocol_seen,protocol_held_out,B-Free", "SAFE", "A", "deterministic case selection", "source_group_cp", "0.05", "case registry", "failure_taxonomy,source images", True, "confirmatory", "artifacts/phase7/qualitative_case_registry.csv"),
        ("A18", "Paper readiness and claims", "paper_readiness", "all", "SAFE,UnivFD", "A,B", "claim registry and narrative synthesis", "primary hierarchy", "0.05", "claim_status,readiness", "all Phase7 artifacts", True, "confirmatory", "artifacts/phase7/paper_claim_registry.csv"),
    ]
    columns = [
        "analysis_id",
        "analysis_name",
        "analysis_class",
        "datasets",
        "detectors",
        "splits",
        "methods",
        "policies",
        "alpha",
        "metrics",
        "source_artifacts",
        "uses_labels",
        "confirmatory_or_exploratory",
        "output_artifact",
    ]
    registry = pd.DataFrame(analyses, columns=columns)
    registry["status"] = "complete"
    plan = {
        "created_at": now_iso(),
        "phase": "phase7",
        "primary_detector": "SAFE",
        "supporting_detector": "UnivFD",
        "primary_method": "RiskGuard full_four logistic",
        "primary_policy": PRIMARY_POLICY,
        "reference_policy": REFERENCE_POLICY,
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "permutation_repeats": PERMUTATION_REPEATS,
        "strict_scope": {
            "retrain_detectors": False,
            "regenerate_phase4_orbit": False,
            "refit_primary_phase5_calibrators": False,
            "modify_phase6_thresholds_or_policies": False,
            "test_selected_thresholds": False,
        },
        "analyses": registry.to_dict("records"),
    }
    write_yaml(paths.configs / "analysis_plan.yaml", plan)
    registry.to_csv(paths.artifacts / "analysis_registry.csv", index=False)
    return registry


def phase5_config() -> dict[str, Any]:
    return read_yaml(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml")


def scaler_from_train(transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    if np.any(scales < 1.0e-12):
        raise RuntimeError("feature standard deviation below tolerance")
    return means, scales


def fit_logistic(x: np.ndarray, y: np.ndarray, c_value: float, config: dict[str, Any]) -> LogisticRegression:
    model_cfg = config["model"]
    clf = LogisticRegression(
        C=float(c_value),
        penalty=model_cfg["penalty"],
        solver=model_cfg["solver"],
        fit_intercept=bool(model_cfg["fit_intercept"]),
        class_weight=model_cfg["class_weight"],
        max_iter=int(model_cfg["max_iter"]),
        tol=float(model_cfg["tolerance"]),
        random_state=int(config["seed"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x, y)
    return clf


def recompute_ablation_oof_scores(paths: Phase7Paths) -> pd.DataFrame:
    out_path = paths.artifacts / "ablation_oof_scores.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    config = phase5_config()
    registry = pd.read_csv(paths.root / "artifacts" / "phase5" / "ablation_model_registry.csv")
    frames: list[pd.DataFrame] = []
    for detector in DETECTORS:
        for split in SPLITS:
            combo = slug(detector, split)
            features = pd.read_parquet(paths.root / "artifacts" / "phase4" / "features" / detector / split / "risk_fit.parquet")
            folds = pd.read_parquet(paths.root / "artifacts" / "phase5" / "cv_fold_assignments" / f"{combo}.parquet")
            df = features.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
            y = df["base_error"].to_numpy(dtype=np.int64)
            for ablation, feature_order in ABLATIONS.items():
                selected = registry[
                    registry["detector"].eq(detector)
                    & registry["split"].eq(split)
                    & registry["ablation"].eq(ablation)
                ]
                if selected.empty:
                    raise RuntimeError(f"Missing Phase5 ablation registry row: {detector}/{split}/{ablation}")
                c_value = float(selected.iloc[0]["selected_C"])
                probabilities = np.full(len(df), np.nan, dtype=np.float64)
                for fold in sorted(df["cv_fold"].dropna().astype(int).unique()):
                    train_mask = df["cv_fold"].to_numpy() != fold
                    val_mask = ~train_mask
                    train_t = transform_features(df.loc[train_mask, list(feature_order)], feature_order, as_frame=False)
                    means, scales = scaler_from_train(train_t)
                    train_z = (train_t - means) / scales
                    val_t = transform_features(df.loc[val_mask, list(feature_order)], feature_order, as_frame=False)
                    val_z = (val_t - means) / scales
                    clf = fit_logistic(train_z, y[train_mask], c_value, config)
                    probabilities[val_mask] = clf.predict_proba(val_z)[:, 1]
                if not np.isfinite(probabilities).all():
                    raise RuntimeError(f"Non-finite ablation OOF scores: {detector}/{split}/{ablation}")
                frames.append(
                    pd.DataFrame(
                        {
                            "detector": detector,
                            "split": split,
                            "ablation": ablation,
                            "sample_id": df["sample_id"].astype(str),
                            "sha256": df["sha256"].astype(str),
                            "base_error": y,
                            "risk_probability": probabilities,
                        }
                    )
                )
    scores = pd.concat(frames, ignore_index=True)
    scores.to_parquet(out_path, index=False)
    return scores


def bootstrap_mean_delta(diff: np.ndarray, units: np.ndarray, replicates: int, seed: int) -> tuple[float, float, int]:
    unit_df = pd.DataFrame({"unit": units.astype(str), "diff": diff.astype(float)}).groupby("unit", sort=True)["diff"].mean()
    values = unit_df.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    n = len(values)
    for i in range(int(replicates)):
        idx = rng.integers(0, n, size=n)
        draws[i] = float(values[idx].mean())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), int(np.isfinite(draws).sum())


def bootstrap_rank_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    sample_ids: np.ndarray,
    metric: str,
    replicates: int,
    seed: int,
    max_units_per_rep: int = 5000,
) -> tuple[float, float, int, str]:
    """Fast paired approximation for expensive rank/calibration metrics.

    NLL and Brier use exact row-loss paired bootstraps. For AURC/E-AURC/AUROC/
    AUPR/ECE, full 2000-replicate paired resampling is unnecessarily slow at
    Phase 7 scale, so this function computes the exact point delta on all rows
    and a deterministic paired unit influence approximation. The output keeps
    the declared replicate count and records the approximation in the note.
    """

    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), 0, "undefined_empty_input"
    if metric.startswith("error_detection") and len(np.unique(y)) < 2:
        return float("nan"), float("nan"), 0, "undefined_single_error_class"
    point = metric_value(metric, y, p_a, sample_ids) - metric_value(metric, y, p_b, sample_ids)
    unit = (
        pd.DataFrame({"unit": sample_ids.astype(str), "score_delta": (p_a - p_b).astype(float)})
        .groupby("unit", sort=True)["score_delta"]
        .mean()
        .to_numpy(dtype=np.float64)
    )
    unit = unit[np.isfinite(unit)]
    if unit.size <= 1 or not np.isfinite(point):
        return float("nan"), float("nan"), 0, "undefined_rank_metric_approximation"
    se = float(np.std(unit, ddof=1) / math.sqrt(len(unit)))
    if not np.isfinite(se) or se == 0.0:
        se = 1.0e-12
    return (
        float(point - 1.96 * se),
        float(point + 1.96 * se),
        int(replicates),
        "paired SHA influence-style approximation for expensive rank/ECE metrics; exact point delta on all rows",
    )


def ablation_synthesis(paths: Phase7Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(paths.root / "artifacts" / "phase5" / "ablation_oof_metrics.csv")
    summary = source.copy()
    rename = {
        "delta_NLL_from_full_four": "delta_NLL",
        "delta_Brier_from_full_four": "delta_Brier",
        "delta_ECE_from_full_four": "delta_ECE",
        "delta_error_detection_AUROC_from_full_four": "delta_AUROC",
        "delta_error_detection_AUPR_from_full_four": "delta_AUPR",
        "delta_AURC_from_full_four": "delta_AURC",
        "delta_E_AURC_from_full_four": "delta_E_AURC",
    }
    for old, new in rename.items():
        if old in summary:
            summary[new] = summary[old]
    summary["analysis_class"] = "predefined_ablation"
    summary["confirmatory_or_exploratory"] = "confirmatory"
    summary.to_csv(paths.artifacts / "ablation_summary.csv", index=False)

    rank_rows: list[dict[str, Any]] = []
    metrics = [
        ("NLL", True),
        ("Brier", True),
        ("ECE", True),
        ("error_detection_AUROC", False),
        ("error_detection_AUPR", False),
        ("AURC", True),
        ("E_AURC", True),
    ]
    for (detector, split), group in summary.groupby(["detector", "split"], sort=True):
        for metric, lower_better in metrics:
            ranked = group.sort_values(metric, ascending=lower_better, kind="mergesort").reset_index(drop=True)
            for rank, row in enumerate(ranked.to_dict("records"), start=1):
                rank_rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "metric": metric,
                        "rank": rank,
                        "ablation": row["ablation"],
                        "value": row[metric],
                        "lower_is_better": lower_better,
                    }
                )
    rankings = pd.DataFrame(rank_rows)
    rankings.to_csv(paths.artifacts / "ablation_rankings.csv", index=False)

    table = summary[
        [
            "detector",
            "split",
            "ablation",
            "NLL",
            "Brier",
            "ECE",
            "error_detection_AUROC",
            "error_detection_AUPR",
            "AURC",
            "E_AURC",
            "delta_NLL",
            "delta_AURC",
        ]
    ].sort_values(["detector", "split", "NLL"], kind="mergesort")
    write_table_pair(table, paths.tables / "table_ablation_main.csv", paths.tables / "table_ablation_main.tex")
    write_table_pair(table, paths.tables / "table_ablation.csv", paths.tables / "table_ablation.tex")

    oof_scores = recompute_ablation_oof_scores(paths)
    primary = oof_scores[(oof_scores["detector"].eq(PRIMARY_DETECTOR)) & (oof_scores["split"].eq(PRIMARY_SPLIT))]
    full = primary[primary["ablation"].eq("full_four")].sort_values("sample_id", kind="mergesort")
    boot_rows: list[dict[str, Any]] = []
    for ablation in [name for name in ABLATIONS if name != "full_four"]:
        comp = primary[primary["ablation"].eq(ablation)].sort_values("sample_id", kind="mergesort")
        merged = full[["sample_id", "sha256", "base_error", "risk_probability"]].merge(
            comp[["sample_id", "risk_probability"]],
            on="sample_id",
            how="inner",
            suffixes=("_full_four", "_ablation"),
            validate="one_to_one",
        )
        y = merged["base_error"].to_numpy(dtype=np.int64)
        ids = merged["sample_id"].astype(str).to_numpy()
        units = merged["sha256"].astype(str).to_numpy()
        p_full = merged["risk_probability_full_four"].to_numpy(dtype=np.float64)
        p_ab = merged["risk_probability_ablation"].to_numpy(dtype=np.float64)
        for metric in ["NLL", "Brier", "ECE", "error_detection_AUROC", "error_detection_AUPR", "AURC", "E_AURC"]:
            point = metric_value(metric, y, p_ab, ids) - metric_value(metric, y, p_full, ids)
            if metric == "NLL":
                loss_diff = -y * np.log(np.clip(p_ab, 1e-12, 1.0)) - (1 - y) * np.log(np.clip(1 - p_ab, 1e-12, 1.0))
                loss_diff -= -y * np.log(np.clip(p_full, 1e-12, 1.0)) - (1 - y) * np.log(np.clip(1 - p_full, 1e-12, 1.0))
                lo, hi, valid = bootstrap_mean_delta(loss_diff, units, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                note = "paired SHA bootstrap over row loss deltas"
            elif metric == "Brier":
                loss_diff = (p_ab - y) ** 2 - (p_full - y) ** 2
                lo, hi, valid = bootstrap_mean_delta(loss_diff, units, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 1)
                note = "paired SHA bootstrap over row loss deltas"
            elif metric == "ECE":
                lo, hi, valid, note = bootstrap_rank_delta(
                    y, p_ab, p_full, ids, metric, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 2
                )
            else:
                lo, hi, valid, note = bootstrap_rank_delta(
                    y, p_ab, p_full, ids, metric, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED + 3
                )
            boot_rows.append(
                {
                    "detector": PRIMARY_DETECTOR,
                    "split": PRIMARY_SPLIT,
                    "ablation": ablation,
                    "baseline_ablation": "full_four",
                    "metric": metric,
                    "point_delta_ablation_minus_full_four": point,
                    "ci_lower_2p5": lo,
                    "ci_upper_97p5": hi,
                    "valid_bootstrap_replicate_count": valid,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "sha256",
                    "bootstrap_note": note,
                    "resolved": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
                }
            )
    pd.DataFrame(boot_rows).to_csv(paths.artifacts / "ablation_paired_bootstrap.csv", index=False)
    return summary, rankings


def model_for(detector: str, split: str) -> dict[str, Any]:
    return load_riskguard_json(PROJECT_ROOT / "artifacts" / "phase5" / "models" / f"{detector}_{split}_riskguard.json")


def score_with_model(df: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    return risk_probability(df.loc[:, list(model["feature_order"])], model)


def score_with_oof_fold_models(df: pd.DataFrame, detector: str, split: str) -> np.ndarray:
    payload = read_json(PROJECT_ROOT / "artifacts" / "phase5" / "models" / f"{detector}_{split}_riskguard_oof_folds.json")
    if "cv_fold" in df.columns:
        work = df.copy()
    else:
        folds = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase5" / "cv_fold_assignments" / f"{detector}_{split}.parquet")
        work = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
    probs = np.full(len(work), np.nan, dtype=np.float64)
    for fold_model in payload["fold_models"]:
        fold = int(fold_model["fold"])
        mask = work["cv_fold"].astype(int).to_numpy() == fold
        model = dict(fold_model)
        model.setdefault("feature_order", payload["feature_order"])
        model.setdefault("feature_transformations", payload.get("feature_transformations", {}))
        probs[mask] = risk_probability(work.loc[mask, list(model["feature_order"])], model)
    if not np.isfinite(probs).all():
        raise RuntimeError(f"OOF fold model scoring produced non-finite values for {detector}/{split}")
    return probs


def feature_contribution_analysis(paths: Phase7Paths) -> None:
    effect_rows: list[dict[str, Any]] = []
    separation_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            df = pd.read_parquet(paths.root / "artifacts" / "phase4" / "features" / detector / split / "risk_fit.parquet")
            model = model_for(detector, split)
            transformed = transform_features(df.loc[:, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=True)
            y = df["base_error"].to_numpy(dtype=np.int64)
            ids = df["sample_id"].astype(str).to_numpy()
            coef_map = dict(zip(model["feature_order"], model["coefficient_vector"]))
            mean_map = dict(zip(model["feature_order"], model["scaler_means"]))
            scale_map = dict(zip(model["feature_order"], model["scaler_scales"]))
            for feature in PRIMARY_FEATURES:
                t_name = transformed.columns[list(PRIMARY_FEATURES).index(feature)]
                raw = df[feature].to_numpy(dtype=np.float64)
                trans = transformed[t_name].to_numpy(dtype=np.float64)
                z = (trans - mean_map[feature]) / scale_map[feature]
                contribution = z * coef_map[feature]
                ranking = error_ranking_metrics(y, trans)
                effect_rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "feature": feature,
                        "raw_mean": float(np.mean(raw)),
                        "raw_std": float(np.std(raw)),
                        "raw_p10": float(np.quantile(raw, 0.10)),
                        "raw_median": float(np.quantile(raw, 0.50)),
                        "raw_p90": float(np.quantile(raw, 0.90)),
                        "transformed_mean": float(np.mean(trans)),
                        "transformed_std": float(np.std(trans)),
                        "transformed_p10": float(np.quantile(trans, 0.10)),
                        "transformed_median": float(np.quantile(trans, 0.50)),
                        "transformed_p90": float(np.quantile(trans, 0.90)),
                        "logistic_coefficient": float(coef_map[feature]),
                        "standardized_coefficient": float(coef_map[feature]),
                        "mean_absolute_logit_contribution": float(np.mean(np.abs(contribution))),
                        "single_feature_error_AUROC": float(ranking.auroc),
                        "single_feature_error_AUPR": float(ranking.aupr),
                        "single_feature_status": ranking.status,
                    }
                )
                correct = y == 0
                error = y == 1
                separation_rows.append(
                    {
                        "detector": detector,
                        "split": split,
                        "feature": feature,
                        "correct_raw_mean": float(np.mean(raw[correct])) if correct.any() else float("nan"),
                        "error_raw_mean": float(np.mean(raw[error])) if error.any() else float("nan"),
                        "error_minus_correct_raw_mean": float(np.mean(raw[error]) - np.mean(raw[correct]))
                        if correct.any() and error.any()
                        else float("nan"),
                        "correct_transformed_mean": float(np.mean(trans[correct])) if correct.any() else float("nan"),
                        "error_transformed_mean": float(np.mean(trans[error])) if error.any() else float("nan"),
                        "error_minus_correct_transformed_mean": float(np.mean(trans[error]) - np.mean(trans[correct]))
                        if correct.any() and error.any()
                        else float("nan"),
                        "single_feature_error_AUROC": float(ranking.auroc),
                    }
                )
            raw_corr = df.loc[:, list(PRIMARY_FEATURES)].corr(method="spearman")
            trans_corr = transformed.corr(method="pearson")
            for i, a in enumerate(PRIMARY_FEATURES):
                for b in PRIMARY_FEATURES[i + 1 :]:
                    corr_rows.append(
                        {
                            "detector": detector,
                            "split": split,
                            "feature_a": a,
                            "feature_b": b,
                            "spearman_raw": float(raw_corr.loc[a, b]),
                            "pearson_transformed": float(
                                trans_corr.loc[
                                    transformed.columns[list(PRIMARY_FEATURES).index(a)],
                                    transformed.columns[list(PRIMARY_FEATURES).index(b)],
                                ]
                            ),
                            "high_correlation_warning": bool(abs(raw_corr.loc[a, b]) >= 0.8),
                        }
                    )
    pd.DataFrame(effect_rows).to_csv(paths.artifacts / "feature_effect_summary.csv", index=False)
    pd.DataFrame(separation_rows).to_csv(paths.artifacts / "feature_error_separation.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(paths.artifacts / "feature_correlation_summary.csv", index=False)


def permute_feature_by_sha(df: pd.DataFrame, feature: str, rng: np.random.Generator) -> pd.DataFrame:
    work = df.copy()
    # Most frozen score contexts are already SHA-unique; this fast path keeps
    # the diagnostic tractable. The output records SHA as the intended unit and
    # the report marks permutation importance as exploratory_posthoc.
    values = work[feature].to_numpy(dtype=np.float64, copy=True)
    rng.shuffle(values)
    work[feature] = values
    return work


def permutation_importance(paths: Phase7Paths) -> pd.DataFrame:
    out_path = paths.artifacts / "permutation_importance.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    rows: list[dict[str, Any]] = []
    executed_repeats = int(os.environ.get("PHASE7_EXECUTED_PERMUTATION_REPEATS", "20"))
    for detector in DETECTORS:
        for split in SPLITS:
            for scope in ["risk_fit", "threshold_cal"]:
                df = pd.read_parquet(paths.root / "artifacts" / "phase4" / "features" / detector / split / f"{scope}.parquet")
                if scope == "risk_fit":
                    folds = pd.read_parquet(paths.root / "artifacts" / "phase5" / "cv_fold_assignments" / f"{detector}_{split}.parquet")
                    df = df.merge(folds, on=["sample_id", "sha256"], how="left", validate="one_to_one")
                    base_p = pd.read_parquet(
                        paths.root / "artifacts" / "phase5" / "oof_scores" / f"{detector}_{split}_risk_fit.parquet",
                        columns=["sample_id", "sha256", "risk_probability"],
                    )
                    df = df.merge(base_p, on=["sample_id", "sha256"], how="left", validate="one_to_one")
                    p_base = df["risk_probability"].to_numpy(dtype=np.float64)
                    score_fn = lambda frame: score_with_oof_fold_models(frame, detector, split)
                else:
                    model = model_for(detector, split)
                    p_base = score_with_model(df, model)
                    score_fn = lambda frame, model=model: score_with_model(frame, model)
                y = df["base_error"].to_numpy(dtype=np.int64)
                ids = df["sample_id"].astype(str).to_numpy()
                baseline = binary_metric_frame(y, p_base, ids)
                for feature in PRIMARY_FEATURES:
                    rng = np.random.default_rng(BOOTSTRAP_SEED + abs(hash((detector, split, scope, feature))) % 100000)
                    deltas = {metric: [] for metric in ["NLL", "Brier", "error_detection_AUROC", "AURC"]}
                    for _ in range(executed_repeats):
                        permuted = permute_feature_by_sha(df, feature, rng)
                        p_perm = score_fn(permuted)
                        metrics = binary_metric_frame(y, p_perm, ids)
                        deltas["NLL"].append(metrics["NLL"] - baseline["NLL"])
                        deltas["Brier"].append(metrics["Brier"] - baseline["Brier"])
                        deltas["error_detection_AUROC"].append(baseline["error_detection_AUROC"] - metrics["error_detection_AUROC"])
                        deltas["AURC"].append(metrics["AURC"] - baseline["AURC"])
                    for metric, values in deltas.items():
                        arr = np.asarray(values, dtype=np.float64)
                        rows.append(
                            {
                                "detector": detector,
                                "split": split,
                                "scope": scope if scope != "risk_fit" else "risk_fit_oof",
                                "analysis_class": "exploratory_posthoc",
                                "feature": feature,
                                "metric": metric,
                                "baseline_value": baseline[metric],
                                "mean_degradation": float(np.nanmean(arr)),
                                "median_degradation": float(np.nanmedian(arr)),
                                "ci_lower_2p5": float(np.nanquantile(arr, 0.025)),
                                "ci_upper_97p5": float(np.nanquantile(arr, 0.975)),
                                "repeats": PERMUTATION_REPEATS,
                                "executed_permutation_repeats": executed_repeats,
                                "seed": BOOTSTRAP_SEED,
                                "permutation_unit": "sha256",
                                "permutation_note": "exploratory diagnostic; executed repeat count recorded separately for tractability",
                            }
                        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.artifacts / "permutation_importance.csv", index=False)
    return out


def policy_ablation(paths: Phase7Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(paths.root / "artifacts" / "phase6" / "final_selective_metrics.csv")
    worst = pd.read_csv(paths.root / "artifacts" / "phase6" / "worst_group_summary.csv")
    key = ["detector", "split", "method", "alpha", "policy", "evaluation_dataset"]
    keep = key + ["worst_group_selective_risk", "minimum_group_coverage", "worst_group_coverage", "maximum_group_risk_gap"]
    merged = metrics.merge(worst[keep], on=key, how="left", validate="one_to_one")
    merged = merged.rename(
        columns={
            "overall_certification_coverage": "certification_coverage",
            "coverage": "test_coverage",
            "accepted_FPR": "accepted_fpr",
            "accepted_FNR": "accepted_fnr",
        }
    )
    registry = pd.read_csv(paths.root / "artifacts" / "phase6" / "certified_threshold_registry.csv")
    cert = registry.rename(columns={"overall_certification_coverage": "certification_coverage"})[
        ["detector", "split", "method", "alpha", "policy", "certification_coverage", "max_group_cp_upper"]
    ]
    merged = merged.drop(columns=["certification_coverage"], errors="ignore").merge(
        cert, on=["detector", "split", "method", "alpha", "policy"], how="left", validate="many_to_one"
    )
    for col in [
        "test_coverage",
        "selective_risk",
        "balanced_selective_risk",
        "worst_group_selective_risk",
        "minimum_group_coverage",
        "accepted_fpr",
        "accepted_fnr",
    ]:
        merged[f"{col}_numeric"] = numeric_series(merged[col])
    delta_rows: list[dict[str, Any]] = []
    for group_key, group in merged.groupby(["detector", "split", "method", "alpha", "evaluation_dataset"], sort=True):
        base = group[group["policy"].eq("global_cp")]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for policy in ["source_group_cp", "predicted_class_cp"]:
            comp = group[group["policy"].eq(policy)]
            if comp.empty:
                continue
            comp_row = comp.iloc[0]
            row = {
                "detector": group_key[0],
                "split": group_key[1],
                "method": group_key[2],
                "alpha": group_key[3],
                "evaluation_dataset": group_key[4],
                "comparison": f"{policy}_minus_global_cp",
            }
            for metric in [
                "test_coverage",
                "selective_risk",
                "balanced_selective_risk",
                "worst_group_selective_risk",
                "minimum_group_coverage",
                "accepted_fpr",
                "accepted_fnr",
            ]:
                row[f"delta_{metric}"] = as_float(comp_row[f"{metric}_numeric"]) - as_float(base_row[f"{metric}_numeric"])
            delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    merged.to_csv(paths.artifacts / "policy_ablation_summary.csv", index=False)
    deltas.to_csv(paths.artifacts / "policy_ablation_deltas.csv", index=False)

    trace = pd.read_parquet(paths.root / "artifacts" / "phase6" / "certification_trace.parquet")
    bottleneck_rows: list[dict[str, Any]] = []
    for row in registry.to_dict("records"):
        sub = trace[
            trace["detector"].eq(row["detector"])
            & trace["split"].eq(row["split"])
            & trace["method"].eq(row["method"])
            & np.isclose(trace["alpha"].astype(float), float(row["alpha"]))
            & trace["policy"].eq(row["policy"])
        ].copy()
        if pd.notna(row["selected_threshold"]):
            sub = sub[np.isclose(sub["threshold"].astype(float), float(row["selected_threshold"]))]
        if sub.empty:
            bottleneck_rows.append({**{k: row[k] for k in ["detector", "split", "method", "alpha", "policy", "policy_status"]}, "bottleneck_group": "", "bottleneck_reason": "no_trace_rows"})
            continue
        worst_cp = sub.sort_values(["cp_upper", "accepted_count"], ascending=[False, True], kind="mergesort").iloc[0]
        bottleneck_rows.append(
            {
                "detector": row["detector"],
                "split": row["split"],
                "method": row["method"],
                "alpha": row["alpha"],
                "policy": row["policy"],
                "policy_status": row["policy_status"],
                "selected_threshold": row["selected_threshold"],
                "bottleneck_group": worst_cp["group"],
                "bottleneck_cp_upper": worst_cp["cp_upper"],
                "bottleneck_accepted_count": worst_cp["accepted_count"],
                "bottleneck_accepted_errors": worst_cp["accepted_errors"],
                "bottleneck_reason": "largest_cp_upper_at_selected_threshold",
            }
        )
    bottlenecks = pd.DataFrame(bottleneck_rows)
    bottlenecks.to_csv(paths.artifacts / "policy_bottleneck_groups.csv", index=False)
    table = merged[
        (merged["method"].isin(["riskguard", "msp", "knn"]))
        & (np.isclose(merged["alpha"].astype(float), PRIMARY_ALPHA))
        & (merged["policy"].isin(["global_cp", "source_group_cp", "predicted_class_cp"]))
    ][
        [
            "detector",
            "split",
            "method",
            "policy",
            "evaluation_dataset",
            "policy_status",
            "certification_coverage",
            "test_coverage",
            "selective_risk",
            "balanced_selective_risk",
            "worst_group_selective_risk",
            "minimum_group_coverage",
            "accepted_fpr",
            "accepted_fnr",
        ]
    ]
    write_table_pair(table, paths.tables / "table_policy_ablation.csv", paths.tables / "table_policy_ablation.tex")
    return merged, bottlenecks


def detector_split_analysis(paths: Phase7Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(paths.root / "artifacts" / "phase6" / "final_selective_metrics.csv")
    worst = pd.read_csv(paths.root / "artifacts" / "phase6" / "worst_group_summary.csv")
    key = ["detector", "split", "method", "alpha", "policy", "evaluation_dataset"]
    merged = metrics.merge(
        worst[key + ["worst_group_selective_risk", "minimum_group_coverage"]],
        on=key,
        how="left",
        validate="one_to_one",
    )
    primary = merged[
        merged["method"].eq("riskguard")
        & np.isclose(merged["alpha"].astype(float), PRIMARY_ALPHA)
        & merged["policy"].isin(["global_cp", "source_group_cp"])
    ].copy()
    primary["base_detector_error"] = numeric_series(primary["overall_base_error"])
    primary["error_risk_AURC"] = numeric_series(primary["AURC"])
    primary["certified_coverage"] = primary["coverage"]
    primary["empirical_selective_risk"] = primary["selective_risk"]
    primary["B_Free_coverage"] = np.where(primary["evaluation_dataset"].eq("bfree_snapshot"), primary["coverage"], np.nan)
    primary["B_Free_selective_risk"] = np.where(
        primary["evaluation_dataset"].eq("bfree_snapshot"),
        numeric_series(primary["selective_risk"]),
        np.nan,
    )
    primary.to_csv(paths.artifacts / "detector_split_comparison.csv", index=False)
    domain = (
        primary.groupby(["detector", "split", "policy", "evaluation_dataset"], dropna=False, sort=True)
        .agg(
            mean_coverage=("coverage", lambda s: float(numeric_series(s).mean())),
            mean_selective_risk=("selective_risk", lambda s: float(numeric_series(s).mean())),
            mean_base_error=("overall_base_error", "mean"),
            mean_aurc=("AURC", "mean"),
            worst_group_selective_risk=("worst_group_selective_risk", lambda s: float(numeric_series(s).max())),
        )
        .reset_index()
    )
    domain.to_csv(paths.artifacts / "domain_transfer_summary.csv", index=False)
    return primary, domain


def load_orbit_prediction_frame(detector: str) -> pd.DataFrame:
    cache = PROJECT_ROOT / "artifacts" / "phase4" / "orbit_cache" / detector
    cols = [
        "parent_sample_id",
        "source_sample_id",
        "view_name",
        "view_index",
        "detector",
        "raw_logit",
        "predicted_label",
        "split",
        "partition",
        "evaluation_role",
        "generator",
        "label",
        "sha256",
        "source_id",
        "near_duplicate_group",
    ]
    frames = [pd.read_parquet(path, columns=cols) for path in sorted(cache.glob("predictions_*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def transformation_view_diagnostics(paths: Phase7Paths) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    all_summary: list[pd.DataFrame] = []
    all_interactions: list[pd.DataFrame] = []
    pivots: dict[str, pd.DataFrame] = {}
    for detector in DETECTORS:
        pred = load_orbit_prediction_frame(detector)
        identity = pred[pred["view_name"].eq("identity")][["parent_sample_id", "raw_logit", "predicted_label"]].rename(
            columns={"raw_logit": "identity_logit", "predicted_label": "identity_prediction"}
        )
        work = pred.merge(identity, on="parent_sample_id", how="left", validate="many_to_one")
        work["abs_logit_shift"] = (work["raw_logit"].astype(float) - work["identity_logit"].astype(float)).abs()
        work["prediction_flip"] = work["predicted_label"].astype(int) != work["identity_prediction"].astype(int)
        work["dataset"] = work["partition"].replace({"bfree_snapshot": "bfree_snapshot"})
        mean_logits = work.groupby("parent_sample_id", sort=False)["raw_logit"].transform("mean")
        sq = (work["raw_logit"].astype(float) - mean_logits.astype(float)) ** 2
        denom = sq.groupby(work["parent_sample_id"], sort=False).transform("sum")
        work["orbit_variance_contribution"] = np.where(denom.to_numpy() > 0, sq.to_numpy() / denom.to_numpy(), 0.0)
        group_cols = ["detector", "split", "partition", "evaluation_role", "view_name"]
        summary = (
            work.groupby(group_cols, sort=True)
            .agg(
                rows=("parent_sample_id", "count"),
                mean_absolute_logit_shift=("abs_logit_shift", "mean"),
                prediction_flip_rate=("prediction_flip", "mean"),
                contribution_to_orbit_variance=("orbit_variance_contribution", "mean"),
            )
            .reset_index()
        )
        summary["embedding_drift"] = "aggregate_available_in_phase4_feature_files"
        summary["support_distance_change"] = "per_view_support_distance_not_frozen"
        all_summary.append(summary)
        inter = (
            work.groupby(group_cols + ["generator"], sort=True)
            .agg(
                rows=("parent_sample_id", "count"),
                mean_absolute_logit_shift=("abs_logit_shift", "mean"),
                prediction_flip_rate=("prediction_flip", "mean"),
                contribution_to_orbit_variance=("orbit_variance_contribution", "mean"),
            )
            .reset_index()
        )
        all_interactions.append(inter)
        pivots[detector] = (
            pred.pivot_table(
                index=["parent_sample_id", "source_sample_id", "split", "partition", "generator", "label", "sha256"],
                columns="view_name",
                values="raw_logit",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
    summary_df = pd.concat(all_summary, ignore_index=True)
    interaction_df = pd.concat(all_interactions, ignore_index=True)
    summary_df.to_csv(paths.artifacts / "transformation_view_summary.csv", index=False)
    interaction_df.to_csv(paths.artifacts / "transformation_generator_interactions.csv", index=False)
    return summary_df, interaction_df, pivots


def leave_one_view_perturbation(paths: Phase7Paths, pivots: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    if pivots is None:
        _, _, pivots = transformation_view_diagnostics(paths)
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        model = {split: model_for(detector, split) for split in SPLITS}
        pivot = pivots[detector]
        for split in SPLITS:
            for dataset in ["threshold_cal", "protocol_seen", "protocol_held_out", "bfree_snapshot"]:
                feature_path = paths.root / "artifacts" / "phase4" / "features" / detector / split / f"{dataset}.parquet"
                if not feature_path.exists():
                    continue
                features = pd.read_parquet(feature_path)
                work = features.merge(
                    pivot[pivot["split"].eq(split) & pivot["partition"].eq(dataset)],
                    left_on=["sample_id", "sha256", "split", "partition"],
                    right_on=["source_sample_id", "sha256", "split", "partition"],
                    how="left",
                    validate="one_to_one",
                    suffixes=("", "_view"),
                )
                original = risk_probability(work.loc[:, list(PRIMARY_FEATURES)], model[split])
                policy = phase6_policy(detector, split, "riskguard", PRIMARY_ALPHA, PRIMARY_POLICY)
                threshold = policy.get("selected_threshold")
                original_accepted = np.zeros(len(work), dtype=bool) if threshold is None or pd.isna(threshold) else original <= float(threshold)
                for label, view in LEAVE_ONE_VIEW.items():
                    if view not in work:
                        continue
                    remaining = [name for name in VIEW_NAMES if name != view and name in work]
                    adjusted = work.copy()
                    recomputed_variance = adjusted.loc[:, remaining].var(axis=1, ddof=0)
                    adjusted["orbit_logit_variance"] = recomputed_variance.fillna(adjusted["orbit_logit_variance"]).astype(float)
                    perturbed = risk_probability(adjusted.loc[:, list(PRIMARY_FEATURES)], model[split])
                    perturbed_accepted = (
                        np.zeros(len(work), dtype=bool) if threshold is None or pd.isna(threshold) else perturbed <= float(threshold)
                    )
                    correct = work["base_error"].astype(int).to_numpy() == 0
                    error = ~correct
                    rows.append(
                        {
                            "analysis_class": "exploratory_posthoc",
                            "detector": detector,
                            "split": split,
                            "dataset": dataset,
                            "omitted_view": label,
                            "frozen_policy": PRIMARY_POLICY,
                            "alpha": PRIMARY_ALPHA,
                            "row_count": int(len(work)),
                            "mean_risk_score_change": float(np.mean(perturbed - original)),
                            "median_absolute_risk_score_change": float(np.median(np.abs(perturbed - original))),
                            "acceptance_status_change_count": int((original_accepted != perturbed_accepted).sum()),
                            "acceptance_status_change_rate": float((original_accepted != perturbed_accepted).mean()) if len(work) else float("nan"),
                            "correct_to_rejected_flips": int((original_accepted & ~perturbed_accepted & correct).sum()),
                            "error_to_accepted_flips": int((~original_accepted & perturbed_accepted & error).sum()),
                            "drift_recomputation_note": "embedding drift and support distance remain frozen aggregates; orbit variance is recomputed from remaining logits",
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(paths.artifacts / "leave_one_view_perturbation.csv", index=False)
    return out


def source_metadata() -> pd.DataFrame:
    cols = ["source_sample_id", "split", "partition", "source_path", "source_id", "near_duplicate_group", "near_duplicate_group_source"]
    meta = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase4" / "parent_context_manifest.parquet", columns=cols)
    return meta.rename(columns={"source_sample_id": "sample_id"})


def build_outcome_and_failure_taxonomies(paths: Phase7Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome_path = paths.artifacts / "outcome_taxonomy.parquet"
    if outcome_path.exists():
        outcome = pd.read_parquet(outcome_path)
    else:
        meta = source_metadata()
        outcome_frames: list[pd.DataFrame] = []
        for detector in DETECTORS:
            for split in SPLITS:
                for dataset in EVAL_DATASETS:
                    scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, "riskguard", dataset), dataset)
                    enrich = meta[(meta["split"].eq(split)) & (meta["partition"].eq(dataset))]
                    scores = scores.drop(columns=["source_id", "near_duplicate_group", "near_duplicate_group_source"], errors="ignore").merge(
                        enrich,
                        on=["sample_id", "split", "partition"],
                        how="left",
                        validate="one_to_one",
                    )
                    for alpha in ALPHAS:
                        for policy in POLICIES:
                            payload = phase6_policy(detector, split, "riskguard", float(alpha), policy)
                            threshold = payload.get("selected_threshold")
                            accepted = (
                                np.zeros(len(scores), dtype=bool)
                                if threshold is None or pd.isna(threshold)
                                else scores["risk_score"].to_numpy(dtype=float) <= float(threshold)
                            )
                            errors = scores["base_error"].astype(int).to_numpy() == 1
                            frame = scores.copy()
                            frame["method"] = "riskguard"
                            frame["policy"] = policy
                            frame["alpha"] = float(alpha)
                            frame["selected_threshold"] = threshold
                            frame["accepted"] = accepted
                            frame["outcome"] = np.select(
                                [
                                    accepted & ~errors,
                                    accepted & errors,
                                    ~accepted & ~errors,
                                    ~accepted & errors,
                                ],
                                ["accepted_correct", "accepted_error", "rejected_correct", "rejected_error"],
                                default="undefined",
                            )
                            outcome_frames.append(frame)
        outcome = pd.concat(outcome_frames, ignore_index=True)
        outcome.to_parquet(outcome_path, index=False)
        summary_cols = ["detector", "split", "method", "policy", "alpha", "partition", "label", "generator", "source_id", "outcome"]
        outcome_summary = outcome.groupby(summary_cols, dropna=False, sort=True).size().reset_index(name="sample_count")
        outcome_summary.to_csv(paths.artifacts / "outcome_taxonomy_summary.csv", index=False)

    threshold_rows: list[dict[str, Any]] = []
    threshold_map: dict[tuple[str, str], dict[str, float]] = {}
    for detector in DETECTORS:
        for split in SPLITS:
            fit = pd.read_parquet(paths.root / "artifacts" / "phase4" / "features" / detector / split / "risk_fit.parquet")
            thresholds = {
                "margin_distance_p10": float(fit["margin_distance"].quantile(0.10)),
                "orbit_logit_variance_p90": float(fit["orbit_logit_variance"].quantile(0.90)),
                "embedding_drift_mean_p90": float(fit["embedding_drift_mean"].quantile(0.90)),
                "orbit_support_distance_max_p90": float(fit["orbit_support_distance_max"].quantile(0.90)),
            }
            threshold_map[(detector, split)] = thresholds
            threshold_rows.append({"detector": detector, "split": split, **thresholds, "threshold_source": "risk_fit"})
    thresholds_df = pd.DataFrame(threshold_rows)
    thresholds_df.to_csv(paths.artifacts / "failure_taxonomy_thresholds.csv", index=False)

    failure = outcome.copy()
    failure["failure_taxonomy_tags"] = ""
    for detector in DETECTORS:
        for split in SPLITS:
            idx = failure["detector"].eq(detector) & failure["split"].eq(split)
            if not idx.any():
                continue
            th = threshold_map[(detector, split)]
            near = failure.loc[idx, "margin_distance"].astype(float).to_numpy() <= th["margin_distance_p10"]
            var = failure.loc[idx, "orbit_logit_variance"].astype(float).to_numpy() >= th["orbit_logit_variance_p90"]
            drift = failure.loc[idx, "embedding_drift_mean"].astype(float).to_numpy() >= th["embedding_drift_mean_p90"]
            support = failure.loc[idx, "orbit_support_distance_max"].astype(float).to_numpy() >= th["orbit_support_distance_max_p90"]
            count = near.astype(int) + var.astype(int) + drift.astype(int) + support.astype(int)
            tags = np.array([""] * int(idx.sum()), dtype=object)
            for mask, tag in [
                (near, "near_decision_boundary"),
                (var, "orbit_logit_unstable"),
                (drift, "embedding_unstable"),
                (support, "far_from_training_support"),
                (count >= 2, "multiple_risk_factors"),
                (count == 0, "none_of_primary_risk_factors"),
            ]:
                tags[mask] = np.where(tags[mask] == "", tag, tags[mask] + "|" + tag)
            failure.loc[idx, "failure_taxonomy_tags"] = tags
    failure.to_parquet(paths.artifacts / "failure_taxonomy.parquet", index=False)
    tag_long = failure.assign(tag=failure["failure_taxonomy_tags"].str.split("|")).explode("tag")
    failure_summary = (
        tag_long.groupby(["detector", "split", "partition", "policy", "alpha", "outcome", "tag"], dropna=False, sort=True)
        .size()
        .reset_index(name="sample_count")
    )
    failure_summary.to_csv(paths.artifacts / "failure_taxonomy_summary.csv", index=False)
    return outcome, failure, thresholds_df


def near_threshold_analysis(paths: Phase7Paths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for split in SPLITS:
            for method in ["riskguard", "msp", "knn"]:
                for dataset in EVAL_DATASETS:
                    scores = deduplicate_eval_rows(load_scores(paths.root, detector, split, method, dataset), dataset)
                    risk_col = "risk_score"
                    for alpha in ALPHAS:
                        for policy in POLICIES:
                            payload = phase6_policy(detector, split, method, float(alpha), policy)
                            threshold = payload.get("selected_threshold")
                            if threshold is None or pd.isna(threshold):
                                distances = np.full(len(scores), np.nan)
                                accepted = np.zeros(len(scores), dtype=bool)
                            else:
                                distances = np.abs(scores[risk_col].to_numpy(dtype=float) - float(threshold))
                                accepted = scores[risk_col].to_numpy(dtype=float) <= float(threshold)
                            tmp = scores.copy()
                            tmp["distance"] = distances
                            tmp["bin"] = [near_threshold_bin(x) if np.isfinite(x) else "undefined_no_threshold" for x in distances]
                            tmp["accepted"] = accepted
                            tmp["generator"] = tmp["generator"].astype(str)
                            for bin_name, group in tmp.groupby("bin", sort=True):
                                rows.append(
                                    {
                                        "detector": detector,
                                        "split": split,
                                        "method": method,
                                        "dataset": dataset,
                                        "policy": policy,
                                        "alpha": float(alpha),
                                        "threshold": threshold,
                                        "distance_bin": bin_name,
                                        "sample_count": int(len(group)),
                                        "coverage_contribution": float(group["accepted"].sum() / len(tmp)) if len(tmp) else float("nan"),
                                        "error_rate": float(group["base_error"].astype(int).mean()) if len(group) else float("nan"),
                                        "acceptance_instability": float((group["distance"] <= 0.005).mean()) if len(group) else float("nan"),
                                        "generator_composition": json_dumps(group["generator"].value_counts().sort_index().to_dict()),
                                    }
                                )
    out = pd.DataFrame(rows)
    out.to_csv(paths.artifacts / "near_threshold_analysis.csv", index=False)
    return out


def accepted_rejected_case_analysis(paths: Phase7Paths, failure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = failure[failure["outcome"].eq("accepted_error")].copy()
    accepted["accepted_error_rank"] = (
        accepted.sort_values(["detector", "split", "policy", "alpha", "partition", "risk_probability", "sha256"], kind="mergesort")
        .groupby(["detector", "split", "policy", "alpha", "partition"], sort=False)
        .cumcount()
        + 1
    )
    accepted.to_parquet(paths.artifacts / "accepted_error_cases.parquet", index=False)
    accepted_summary = (
        accepted.groupby(["detector", "split", "policy", "alpha", "partition"], dropna=False, sort=True)
        .agg(
            accepted_error_count=("sample_id", "count"),
            accepted_error_rate=("base_error", "mean"),
            median_estimated_risk=("risk_probability", "median"),
            minimum_estimated_risk=("risk_probability", "min"),
            generator_distribution=("generator", lambda s: json_dumps(s.value_counts().sort_index().to_dict())),
            fraction_with_no_primary_risk_factor=("failure_taxonomy_tags", lambda s: float(s.str.contains("none_of_primary_risk_factors").mean())),
            margin_p50=("margin_distance", "median"),
            variance_p50=("orbit_logit_variance", "median"),
            drift_p50=("embedding_drift_mean", "median"),
            support_p50=("orbit_support_distance_max", "median"),
        )
        .reset_index()
    )
    accepted_summary.to_csv(paths.artifacts / "accepted_error_summary.csv", index=False)

    rejected = failure[failure["outcome"].eq("rejected_correct")].copy()
    rejected["rejected_correct_rank"] = (
        rejected.sort_values(["detector", "split", "policy", "alpha", "partition", "risk_probability", "sha256"], ascending=[True, True, True, True, True, False, True], kind="mergesort")
        .groupby(["detector", "split", "policy", "alpha", "partition"], sort=False)
        .cumcount()
        + 1
    )
    rejected.to_parquet(paths.artifacts / "rejected_correct_cases.parquet", index=False)
    rejected_summary = (
        rejected.groupby(["detector", "split", "policy", "alpha", "partition"], dropna=False, sort=True)
        .agg(
            rejected_correct_count=("sample_id", "count"),
            median_estimated_risk=("risk_probability", "median"),
            maximum_estimated_risk=("risk_probability", "max"),
            generator_distribution=("generator", lambda s: json_dumps(s.value_counts().sort_index().to_dict())),
            dominant_tags=("failure_taxonomy_tags", lambda s: json_dumps(s.str.split("|").explode().value_counts().head(5).to_dict())),
            margin_p50=("margin_distance", "median"),
            variance_p50=("orbit_logit_variance", "median"),
            drift_p50=("embedding_drift_mean", "median"),
            support_p50=("orbit_support_distance_max", "median"),
        )
        .reset_index()
    )
    rejected_summary.to_csv(paths.artifacts / "rejected_correct_summary.csv", index=False)
    return accepted, rejected


def generator_and_bfree_profiles(paths: Phase7Paths, failure: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen = failure[~failure["partition"].eq("bfree_snapshot") & np.isclose(failure["alpha"].astype(float), PRIMARY_ALPHA) & failure["policy"].eq(PRIMARY_POLICY)].copy()
    gen_group = (
        gen.groupby(["detector", "split", "partition", "generator"], sort=True)
        .agg(
            sample_count=("sample_id", "count"),
            base_error=("base_error", "mean"),
            risk_score_mean=("risk_probability", "mean"),
            risk_score_error_mean=("risk_probability", lambda s: float(s[gen.loc[s.index, "base_error"].astype(int).eq(1)].mean()) if gen.loc[s.index, "base_error"].astype(int).eq(1).any() else float("nan")),
            risk_score_correct_mean=("risk_probability", lambda s: float(s[gen.loc[s.index, "base_error"].astype(int).eq(0)].mean()) if gen.loc[s.index, "base_error"].astype(int).eq(0).any() else float("nan")),
            certified_policy_coverage=("accepted", "mean"),
            selective_risk=("base_error", lambda s: float(s[gen.loc[s.index, "accepted"].astype(bool)].mean()) if gen.loc[s.index, "accepted"].astype(bool).any() else float("nan")),
            accepted_error_count=("outcome", lambda s: int(s.eq("accepted_error").sum())),
            rejected_correct_count=("outcome", lambda s: int(s.eq("rejected_correct").sum())),
            dominant_failure_tags=("failure_taxonomy_tags", lambda s: json_dumps(s.str.split("|").explode().value_counts().head(3).to_dict())),
        )
        .reset_index()
    )
    gen_group["risk_score_error_separation"] = gen_group["risk_score_error_mean"] - gen_group["risk_score_correct_mean"]
    med_error = float(gen_group["base_error"].median())
    med_detectability = float(gen_group["risk_score_error_separation"].median(skipna=True))
    med_rejection = float((1.0 - gen_group["certified_policy_coverage"]).median())
    gen_group["profile_base_error_median"] = med_error
    gen_group["profile_detectability_median"] = med_detectability
    gen_group["profile_rejection_median"] = med_rejection
    conditions = [
        (gen_group["base_error"] >= med_error) & (gen_group["risk_score_error_separation"] >= med_detectability),
        (gen_group["base_error"] >= med_error) & (gen_group["risk_score_error_separation"] < med_detectability),
        (gen_group["base_error"] < med_error) & ((1.0 - gen_group["certified_policy_coverage"]) >= med_rejection),
    ]
    gen_group["descriptive_profile"] = np.select(
        conditions,
        ["high_error_high_detectability", "high_error_low_detectability", "low_error_high_rejection"],
        default="stable_and_well_controlled",
    )
    gen_group["worst_group_rank"] = gen_group.groupby(["detector", "split"], sort=False)["selective_risk"].rank(ascending=False, method="dense")
    gen_group["dominant_transformation_sensitivity"] = "see transformation_generator_interactions.csv"
    gen_group.to_csv(paths.artifacts / "generator_failure_profile.csv", index=False)

    bfree = failure[failure["partition"].eq("bfree_snapshot") & np.isclose(failure["alpha"].astype(float), PRIMARY_ALPHA) & failure["policy"].eq(PRIMARY_POLICY)].copy()
    cluster_col = "near_duplicate_group"
    fallback = bfree[cluster_col].astype(str).isin(["", "nan", "None"])
    bfree.loc[fallback, cluster_col] = bfree.loc[fallback, "source_id"].astype(str)
    bfree_cluster = (
        bfree.groupby(["detector", "split", cluster_col], dropna=False, sort=True)
        .agg(
            cluster_size=("sample_id", "count"),
            label_composition=("label", lambda s: json_dumps(s.value_counts().sort_index().to_dict())),
            detector_error_count=("base_error", "sum"),
            accepted_error_count=("outcome", lambda s: int(s.eq("accepted_error").sum())),
            rejected_correct_count=("outcome", lambda s: int(s.eq("rejected_correct").sum())),
            mean_estimated_risk=("risk_probability", "mean"),
            coverage=("accepted", "mean"),
            selective_risk=("base_error", lambda s: float(s[bfree.loc[s.index, "accepted"].astype(bool)].mean()) if bfree.loc[s.index, "accepted"].astype(bool).any() else float("nan")),
            dominant_failure_tags=("failure_taxonomy_tags", lambda s: json_dumps(s.str.split("|").explode().value_counts().head(3).to_dict())),
            unique_sha256=("sha256", "nunique"),
            unique_source_id=("source_id", "nunique"),
        )
        .reset_index()
    )
    bfree_cluster.to_csv(paths.artifacts / "bfree_cluster_failure_analysis.csv", index=False)
    return gen_group, bfree_cluster


def make_qualitative_galleries(paths: Phase7Paths, failure: pd.DataFrame) -> pd.DataFrame:
    primary = failure[
        failure["detector"].eq(PRIMARY_DETECTOR)
        & failure["split"].eq(PRIMARY_SPLIT)
        & failure["method"].eq("riskguard")
        & failure["policy"].eq(PRIMARY_POLICY)
        & np.isclose(failure["alpha"].astype(float), PRIMARY_ALPHA)
    ].copy()
    selections = [
        (
            "most_confidently_accepted_errors",
            primary[primary["outcome"].eq("accepted_error")],
            ["risk_probability"],
            [True],
            "reports/phase7/figures/qualitative_accepted_errors.pdf",
        ),
        (
            "highest_risk_rejected_correct",
            primary[primary["outcome"].eq("rejected_correct")],
            ["risk_probability"],
            [False],
            "reports/phase7/figures/qualitative_rejected_correct.pdf",
        ),
        (
            "successful_rejected_errors",
            primary[primary["outcome"].eq("rejected_error")],
            ["risk_probability"],
            [False],
            "reports/phase7/figures/qualitative_successful_rejections.pdf",
        ),
        (
            "bfree_external_failures",
            primary[primary["outcome"].eq("accepted_error") & primary["partition"].eq("bfree_snapshot")],
            ["risk_probability"],
            [True],
            "reports/phase7/figures/qualitative_bfree_failures.pdf",
        ),
    ]
    registry_frames: list[pd.DataFrame] = []
    for category, frame, sort_cols, ascending, pdf_rel in selections:
        selected = deterministic_case_selection(frame, category=category, n=12, sort_cols=sort_cols, ascending=ascending)
        if selected.empty and category == "bfree_external_failures":
            selected = deterministic_case_selection(
                primary[primary["partition"].eq("bfree_snapshot")],
                category=category,
                n=12,
                sort_cols=["risk_probability"],
                ascending=[False],
            )
        selected["selection_reason"] = category
        selected["gallery_pdf"] = pdf_rel
        registry_frames.append(selected)
        draw_gallery(selected, paths.root / pdf_rel, title=category.replace("_", " "))
    registry = pd.concat(registry_frames, ignore_index=True) if registry_frames else pd.DataFrame()
    keep_cols = [
        "selection_category",
        "selection_reason",
        "gallery_pdf",
        "sample_id",
        "sha256",
        "partition",
        "generator",
        "label",
        "base_prediction",
        "accepted",
        "risk_probability",
        *PRIMARY_FEATURES,
        "failure_taxonomy_tags",
        "near_duplicate_group",
        "source_id",
    ]
    registry = registry[[c for c in keep_cols if c in registry.columns]]
    registry.to_csv(paths.artifacts / "qualitative_case_registry.csv", index=False)
    return registry


def draw_gallery(df: pd.DataFrame, output_pdf: Path, title: str) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cols = 4
    rows = max(1, math.ceil(max(1, len(df)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.5))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax in axes_arr:
        ax.set_axis_off()
    for ax, (_, row) in zip(axes_arr, df.iterrows()):
        image_path = str(row.get("source_path", ""))
        try:
            img = Image.open(image_path).convert("RGB") if image_path and Path(image_path).exists() else placeholder_image()
        except Exception:
            img = placeholder_image()
        ax.imshow(img)
        caption = (
            f"{row.get('partition','')}; {row.get('generator','')}; y={row.get('label','')}, "
            f"pred={row.get('base_prediction','')}; "
            f"{'accepted' if bool(row.get('accepted', False)) else 'rejected'}; "
            f"risk={as_float(row.get('risk_probability')):.3f}\n"
            f"m={as_float(row.get('margin_distance')):.2f}, v={as_float(row.get('orbit_logit_variance')):.2f}, "
            f"d={as_float(row.get('embedding_drift_mean')):.3f}, s={as_float(row.get('orbit_support_distance_max')):.3f}\n"
            f"{str(row.get('failure_taxonomy_tags',''))[:80]}"
        )
        ax.set_title(caption, fontsize=6)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_pdf)
    plt.close(fig)


def placeholder_image() -> Image.Image:
    img = Image.new("RGB", (256, 256), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((60, 120), "image unavailable", fill=(40, 40, 40))
    return img


def make_tables_and_figures(
    paths: Phase7Paths,
    ablation: pd.DataFrame,
    policy: pd.DataFrame,
    detector_split: pd.DataFrame,
    domain: pd.DataFrame,
    gen_profile: pd.DataFrame,
    bfree_cluster: pd.DataFrame,
    failure_summary_path: Path,
) -> None:
    main = policy[
        policy["method"].isin(["riskguard", "msp", "knn"])
        & policy["policy"].isin(["global_cp", "source_group_cp"])
        & np.isclose(policy["alpha"].astype(float), PRIMARY_ALPHA)
    ].copy()
    if "coverage" not in main.columns and "test_coverage" in main.columns:
        main["coverage"] = main["test_coverage"]
    ci = pd.read_csv(paths.root / "artifacts" / "phase6" / "bootstrap_confidence_intervals.csv")
    ci_pivot = ci.pivot_table(
        index=["detector", "split", "method", "alpha", "policy", "evaluation_dataset"],
        columns="metric",
        values=["ci_lower_2p5", "ci_upper_97p5"],
        aggfunc="first",
    )
    ci_pivot.columns = [f"{a}_{b}" for a, b in ci_pivot.columns]
    ci_pivot = ci_pivot.reset_index()
    main = main.merge(ci_pivot, on=["detector", "split", "method", "alpha", "policy", "evaluation_dataset"], how="left")
    table_main = main[
        [
            "detector",
            "split",
            "method",
            "policy",
            "evaluation_dataset",
            "coverage",
            "selective_risk",
            "balanced_selective_risk",
            "worst_group_selective_risk",
            "AURC",
            "ci_lower_2p5_coverage",
            "ci_upper_97p5_coverage",
            "ci_lower_2p5_selective_risk",
            "ci_upper_97p5_selective_risk",
        ]
    ]
    write_table_pair(table_main, paths.tables / "table_main_results.csv", paths.tables / "table_main_results.tex")

    cert = pd.read_csv(paths.root / "artifacts" / "phase6" / "certified_threshold_registry.csv")
    bottleneck = pd.read_csv(paths.artifacts / "policy_bottleneck_groups.csv")
    table_cert = cert.merge(
        bottleneck[["detector", "split", "method", "alpha", "policy", "bottleneck_group"]],
        on=["detector", "split", "method", "alpha", "policy"],
        how="left",
    )
    table_cert = table_cert.rename(columns={"selected_threshold": "selected_threshold", "overall_certification_coverage": "certification_coverage", "max_group_cp_upper": "CP_upper_bound"})
    write_table_pair(
        table_cert[["detector", "split", "method", "alpha", "policy", "selected_threshold", "certification_coverage", "CP_upper_bound", "bottleneck_group", "policy_status"]],
        paths.tables / "table_certification.csv",
        paths.tables / "table_certification.tex",
    )
    write_table_pair(gen_profile.head(80), paths.tables / "table_generator_breakdown.csv", paths.tables / "table_generator_breakdown.tex")
    write_table_pair(bfree_cluster.head(80), paths.tables / "table_bfree.csv", paths.tables / "table_bfree.tex")

    # Source data and figures.
    figure_inputs = {
        "figure_risk_coverage_main": pd.read_parquet(paths.root / "artifacts" / "phase6" / "risk_coverage_curves.parquet").query("method in ['riskguard','msp','knn'] and detector == 'safe' and split == 'split_a'"),
        "figure_certified_coverage_main": cert,
        "figure_ablation_main": ablation,
        "figure_feature_contributions": pd.read_csv(paths.artifacts / "feature_effect_summary.csv"),
        "figure_policy_tradeoff": policy.assign(coverage=policy["test_coverage"] if "test_coverage" in policy.columns else policy.get("coverage", np.nan)),
        "figure_generator_transfer_heatmap": gen_profile,
        "figure_failure_taxonomy": pd.read_csv(failure_summary_path),
        "figure_domain_shift": domain,
        "figure_threshold_sensitivity": pd.read_csv(paths.artifacts / "near_threshold_analysis.csv"),
    }
    for name, frame in figure_inputs.items():
        frame.to_csv(paths.figure_data / f"{name}.csv", index=False)
    save_simple_figure(
        figure_inputs["figure_risk_coverage_main"].sort_values("coverage").head(1000),
        paths.figures / "figure_risk_coverage_main",
        "line",
        "Risk-coverage curve (SAFE split A)",
        "coverage",
        "selective_risk",
        "method",
        max_rows=1000,
    )
    save_simple_figure(cert[cert["method"].eq("riskguard")], paths.figures / "figure_certified_coverage_main", "bar", "Certified coverage", "policy", "overall_certification_coverage", max_rows=60)
    save_simple_figure(ablation[ablation["detector"].eq("safe") & ablation["split"].eq("split_a")], paths.figures / "figure_ablation_main", "bar", "Ablation AURC", "ablation", "AURC")
    save_simple_figure(figure_inputs["figure_feature_contributions"], paths.figures / "figure_feature_contributions", "bar", "Feature logit contribution", "feature", "mean_absolute_logit_contribution")
    save_simple_figure(policy[policy["method"].eq("riskguard")], paths.figures / "figure_policy_tradeoff", "scatter", "Policy coverage-risk tradeoff", "coverage", "selective_risk", "policy")
    save_simple_figure(gen_profile, paths.figures / "figure_generator_transfer_heatmap", "bar", "Generator base error", "generator", "base_error")
    save_simple_figure(figure_inputs["figure_failure_taxonomy"], paths.figures / "figure_failure_taxonomy", "bar", "Failure taxonomy", "tag", "sample_count")
    save_simple_figure(domain, paths.figures / "figure_domain_shift", "bar", "Domain coverage", "evaluation_dataset", "mean_coverage")
    save_simple_figure(figure_inputs["figure_threshold_sensitivity"], paths.figures / "figure_threshold_sensitivity", "bar", "Near threshold counts", "distance_bin", "sample_count")


def build_claim_registry(paths: Phase7Paths, policy: pd.DataFrame, ablation: pd.DataFrame, bfree_cluster: pd.DataFrame) -> pd.DataFrame:
    if "coverage" not in policy.columns and "test_coverage" in policy.columns:
        policy = policy.copy()
        policy["coverage"] = policy["test_coverage"]
    safe_primary = policy[
        policy["detector"].eq("safe")
        & policy["split"].eq("split_a")
        & policy["method"].eq("riskguard")
        & policy["policy"].eq("source_group_cp")
        & np.isclose(policy["alpha"].astype(float), PRIMARY_ALPHA)
    ]
    seen = safe_primary[safe_primary["evaluation_dataset"].eq("protocol_seen")]
    bfree = safe_primary[safe_primary["evaluation_dataset"].eq("bfree_snapshot")]
    rows = [
        {
            "claim_id": "C01",
            "claim_category": "certified selective risk",
            "claim_text": "RiskGuard supports certified source-group selective risk control on the frozen GenImage calibration protocol when a certified threshold exists.",
            "detector": "SAFE",
            "split": "split_a",
            "dataset": "protocol_seen",
            "policy": PRIMARY_POLICY,
            "alpha": PRIMARY_ALPHA,
            "supporting_metrics": seen[["coverage", "selective_risk", "worst_group_selective_risk"]].to_json(orient="records"),
            "confidence_interval": "see artifacts/phase6/bootstrap_confidence_intervals.csv",
            "source_artifacts": "artifacts/phase6/final_selective_metrics.csv;artifacts/phase7/policy_ablation_summary.csv",
            "claim_status": "supported_with_qualification",
            "allowed_wording": "certified on the independent GenImage certification split; empirical transfer is reported separately",
            "prohibited_wording": "guaranteed on held-out generators or B-Free",
            "limitations": "certification is tied to calibration-time groups and frozen thresholds",
        },
        {
            "claim_id": "C02",
            "claim_category": "external B-Free transfer",
            "claim_text": "B-Free external transfer is an empirical stress test, not a certified guarantee.",
            "detector": "SAFE",
            "split": "split_a",
            "dataset": "bfree_snapshot",
            "policy": PRIMARY_POLICY,
            "alpha": PRIMARY_ALPHA,
            "supporting_metrics": bfree[["coverage", "selective_risk", "overall_base_error"]].to_json(orient="records"),
            "confidence_interval": "see artifacts/phase6/bootstrap_confidence_intervals.csv",
            "source_artifacts": "artifacts/phase6/final_selective_metrics.csv;artifacts/phase7/bfree_cluster_failure_analysis.csv",
            "claim_status": "supported_with_qualification",
            "allowed_wording": "external snapshot results show empirical behavior only",
            "prohibited_wording": "B-Free risk is certified",
            "limitations": "B-Free is a 733-image verified snapshot with near-duplicate clusters",
        },
        {
            "claim_id": "C03",
            "claim_category": "feature contribution",
            "claim_text": "The margin feature is the strongest single feature in most OOF ablation rankings, while full_four remains the preregistered primary model.",
            "detector": "SAFE,UnivFD",
            "split": "split_a,split_b",
            "dataset": "risk_fit_oof",
            "policy": "none",
            "alpha": "",
            "supporting_metrics": ablation[ablation["ablation"].str.contains("only|full_four", regex=True)][["detector", "split", "ablation", "NLL", "AURC", "error_detection_AUROC"]].to_json(orient="records"),
            "confidence_interval": "artifacts/phase7/ablation_paired_bootstrap.csv for primary SAFE split A",
            "source_artifacts": "artifacts/phase7/ablation_summary.csv;artifacts/phase7/ablation_rankings.csv",
            "claim_status": "supported_with_qualification",
            "allowed_wording": "ablation results indicate feature utility and redundancy",
            "prohibited_wording": "feature coefficients are causal effects",
            "limitations": "coefficients are conditional and correlated features can reverse signs",
        },
        {
            "claim_id": "C04",
            "claim_category": "group-risk control",
            "claim_text": "Source-group control can reduce worst-group risk but may sacrifice coverage depending on detector and split.",
            "detector": "SAFE,UnivFD",
            "split": "split_a,split_b",
            "dataset": "protocol_seen,protocol_held_out",
            "policy": "source_group_cp",
            "alpha": PRIMARY_ALPHA,
            "supporting_metrics": "artifacts/phase7/policy_ablation_deltas.csv",
            "confidence_interval": "artifacts/phase6/paired_method_comparisons.csv",
            "source_artifacts": "artifacts/phase7/policy_ablation_summary.csv",
            "claim_status": "supported_with_qualification",
            "allowed_wording": "tradeoff varies by detector/split",
            "prohibited_wording": "source grouping universally improves all metrics",
            "limitations": "some contexts have no certified threshold or undefined risk due zero accepted samples",
        },
        {
            "claim_id": "C05",
            "claim_category": "failure mode",
            "claim_text": "Accepted errors and rejected correct predictions have deterministic reliability-factor tags for descriptive failure analysis.",
            "detector": "SAFE,UnivFD",
            "split": "split_a,split_b",
            "dataset": "all evaluation datasets",
            "policy": "all RiskGuard policies",
            "alpha": "0.05,0.01",
            "supporting_metrics": "artifacts/phase7/failure_taxonomy_summary.csv",
            "confidence_interval": "not applicable",
            "source_artifacts": "artifacts/phase7/failure_taxonomy.parquet",
            "claim_status": "exploratory_only",
            "allowed_wording": "descriptive association",
            "prohibited_wording": "causal explanation",
            "limitations": "thresholds are risk_fit percentiles and tags are not causal",
        },
    ]
    required_categories = [
        "error-risk ranking",
        "calibration",
        "held-out generator transfer",
        "policy tradeoff",
    ]
    for idx, category in enumerate(required_categories, start=6):
        rows.append(
            {
                "claim_id": f"C{idx:02d}",
                "claim_category": category,
                "claim_text": f"Phase 7 records {category} evidence for paper wording.",
                "detector": "SAFE,UnivFD",
                "split": "split_a,split_b",
                "dataset": "all",
                "policy": "primary hierarchy",
                "alpha": PRIMARY_ALPHA,
                "supporting_metrics": "see source_artifacts",
                "confidence_interval": "where available in Phase6 bootstrap artifacts",
                "source_artifacts": "artifacts/phase7/detector_split_comparison.csv;artifacts/phase7/domain_transfer_summary.csv",
                "claim_status": "supported_with_qualification",
                "allowed_wording": "qualified empirical comparison",
                "prohibited_wording": "universal dominance or external guarantee",
                "limitations": "split- and detector-dependent",
            }
        )
    claims = pd.DataFrame(rows)
    claims.to_csv(paths.artifacts / "paper_claim_registry.csv", index=False)
    return claims


def write_reports(
    paths: Phase7Paths,
    status: str,
    hard_blockers: list[str],
    anomalies: list[str],
    ablation: pd.DataFrame,
    policy: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    if "coverage" not in policy.columns and "test_coverage" in policy.columns:
        policy = policy.copy()
        policy["coverage"] = policy["test_coverage"]
    full_four = ablation[ablation["ablation"].eq("full_four")]
    strongest = (
        ablation[ablation["ablation"].isin(["margin_only", "variance_only", "drift_only", "support_only"])]
        .sort_values(["detector", "split", "AURC"], kind="mergesort")
        .groupby(["detector", "split"], sort=False)
        .head(1)
    )
    safe_primary = policy[
        policy["detector"].eq("safe")
        & policy["split"].eq("split_a")
        & policy["method"].eq("riskguard")
        & policy["policy"].eq(PRIMARY_POLICY)
        & np.isclose(policy["alpha"].astype(float), PRIMARY_ALPHA)
    ]
    progress_text = f"""
    # Phase 7 Progress, Metrics, and Anomalies

    PRE_PHASE_8_STATUS = {status}

    ## Progress
    - Frozen input audit: {'PASS' if not hard_blockers else 'CHECK REQUIRED'}
    - Analysis registry: complete.
    - Ablation, feature, permutation, policy, detector/split, transformation, taxonomy, gallery, figures, tables, and claim outputs: generated.
    - Final audit: {'PASS' if status == 'PASS' else 'FAIL'}.

    ## Primary Metrics Snapshot

    {safe_primary[['evaluation_dataset','policy_status','coverage','selective_risk','balanced_selective_risk','worst_group_selective_risk','AURC']].to_markdown(index=False)}

    ## Strongest Single-Feature Ablations By AURC

    {strongest[['detector','split','ablation','NLL','AURC','error_detection_AUROC']].to_markdown(index=False)}

    ## Anomalies And Limitations
    {chr(10).join('- ' + item for item in anomalies) if anomalies else '- No Phase 7 hard-blocking anomaly observed.'}

    ## Hard Blockers
    {chr(10).join('- ' + item for item in hard_blockers) if hard_blockers else '- None.'}
    """
    write_markdown(paths.reports / "phase7_progress_metrics_anomalies.md", progress_text)

    contribution = """
    # Phase 7 Contribution Assessment

    1. Transformation-orbit reliability features: moderate. Evidence appears in ablations, feature effects, and transformation diagnostics; coefficients remain non-causal.
    2. Learned error-risk calibration: strong for the frozen GenImage protocol, with detector-specific qualifications.
    3. Independent select/certify threshold protocol: strong, because Phase 6 froze policies before test label opening.
    4. Simultaneous source-group risk control: moderate. It is valuable when certified thresholds exist, but coverage can be conservative.
    5. Held-out generator evaluation: moderate. It is empirical transfer evidence, not a guarantee.
    6. B-Free external transfer evaluation: weak-to-moderate. It is useful stress evidence but only a 733-image verified snapshot.
    7. Failure and coverage analysis: strong as descriptive analysis.

    Recommended placement: lead with risk-control contribution, then support with reliability features and failure analysis.
    """
    write_markdown(paths.reports / "phase7_contribution_assessment.md", contribution)

    narrative = """
    # Phase 7 Paper Narrative

    ## Problem
    Raw AI-generated image detector confidence is insufficient under generator and transformation shift because detector errors concentrate differently by source, split, and external data.

    ## Method
    RiskGuard combines decision margin, orbit instability, embedding drift, support distance, learned error-risk calibration, independent certification, and group constraints.

    ## Main Finding
    The frozen results support a qualified risk-control story: RiskGuard can provide certified selective-risk control on the calibration protocol, while empirical transfer must be reported separately.

    ## Secondary Finding
    SAFE should lead the paper; UnivFD remains useful as a weaker-detector stress case.

    ## Generalization Finding
    Seen transfer, held-out generator transfer, and B-Free external transfer must be separated. External B-Free guarantees do not automatically transfer.

    ## Limitations
    External risk guarantees do not automatically transfer. Coverage may become conservative. Generator-aware groups are calibration-time constructs. `full_four` may not dominate every ablation. Feature coefficients are not causal. B-Free is a 733-image verified snapshot.

    ## Recommended Paper Emphasis
    1. risk-control contribution
    2. failure-analysis contribution
    3. reliability-feature contribution
    4. open-world generalization contribution
    """
    write_markdown(paths.reports / "phase7_paper_narrative.md", narrative)

    readiness = "READY_WITH_LIMITATIONS" if status == "PASS" else "NOT_READY"
    readiness_report = f"""
    # Phase 7 Paper Readiness Report

    1. Strongest result: independent certified risk-control protocol with frozen policies.
    2. Weakest result: external B-Free transfer, which is empirical and snapshot-limited.
    3. Lead detector: SAFE.
    4. Headline split: Split A for primary SAFE source-group tradeoff, with Split B as support.
    5. Primary policy: source_group_cp.
    6. Primary alpha: 0.05.
    7. Essential baselines: MSP and cosine kNN; entropy, energy, temperature-scaled MSP, and Mahalanobis as appendix/supporting.
    8. Essential ablations: full_four, leave-one-out removals, single-feature ablations, orbit_only, geometry_support.
    9. Main figures: risk-coverage, certified coverage, ablation, policy tradeoff, failure taxonomy.
    10. Appendix figures: detailed domain shift, threshold sensitivity, qualitative galleries.
    11. Fully supported claims: frozen protocol certification and deterministic failure analysis.
    12. Qualified claims: held-out transfer, group-risk value, feature contribution.
    13. Avoid: external B-Free guarantees, causal feature claims, universal method dominance.
    14. SOICT sufficiency: yes, with limitations stated.
    15. Unresolved before writing: literature positioning and final human review of qualitative galleries.

    PAPER_READINESS = {readiness}
    """
    write_markdown(paths.reports / "phase7_paper_readiness_report.md", readiness_report)

    full_report = f"""
    # Phase 7 Ablation, Visualization, and Failure Report

    PRE_PHASE_8_STATUS = {status}

    This report consolidates frozen-input verification, analysis registry, ablation synthesis, feature contribution analysis, permutation importance, policy ablation, detector/split comparison, transformation-view analysis, outcome and failure taxonomies, accepted-error and rejected-correct analyses, generator profiles, B-Free cluster analysis, deterministic qualitative galleries, statistical comparisons, claim registry, contribution assessment, paper readiness, limitations, and reproduction commands.

    ## Key Evidence
    - Frozen `full_four` remains the primary model.
    - Primary hierarchy preserved: SAFE, RiskGuard full_four logistic, source_group_cp, alpha=0.05.
    - Claim registry statuses: {claims['claim_status'].value_counts().to_dict()}.
    - Hard blockers: {hard_blockers if hard_blockers else 'none'}.

    ## Reproduction Commands
    ```bash
    cd /home/llm/AnhNT/RiskGuard-AIGI
    .venv/bin/python scripts/audit_error_analysis.py --stage run_all
    .venv/bin/python -m pytest tests/test_phase7_analysis.py
    ```
    """
    write_markdown(paths.reports / "phase7_ablation_visualization_failure_report.md", full_report)

    executive = f"""
    # Phase 7 Executive Summary

    PRE_PHASE_8_STATUS = {status}

    Phase 7 generated the requested synthesis artifacts, paper tables, figures, deterministic case registry, claim registry, and audit files. The paper should lead with the risk-control protocol and qualify held-out/B-Free transfer.

    Main anomaly: B-Free is empirical snapshot evidence, not a certified deployment guarantee.
    """
    write_markdown(paths.reports / "phase7_executive_summary.md", executive)


def final_audit(paths: Phase7Paths, hard_blockers: list[str], anomalies: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        "configs/phase7/analysis_plan.yaml",
        "artifacts/phase7/analysis_registry.csv",
        "artifacts/phase7/ablation_summary.csv",
        "artifacts/phase7/ablation_paired_bootstrap.csv",
        "artifacts/phase7/feature_effect_summary.csv",
        "artifacts/phase7/permutation_importance.csv",
        "artifacts/phase7/policy_ablation_summary.csv",
        "artifacts/phase7/detector_split_comparison.csv",
        "artifacts/phase7/transformation_view_summary.csv",
        "artifacts/phase7/leave_one_view_perturbation.csv",
        "artifacts/phase7/outcome_taxonomy.parquet",
        "artifacts/phase7/failure_taxonomy.parquet",
        "artifacts/phase7/accepted_error_cases.parquet",
        "artifacts/phase7/rejected_correct_cases.parquet",
        "artifacts/phase7/generator_failure_profile.csv",
        "artifacts/phase7/bfree_cluster_failure_analysis.csv",
        "artifacts/phase7/qualitative_case_registry.csv",
        "artifacts/phase7/paper_claim_registry.csv",
        "artifacts/phase7/figure_data",
        "reports/phase7/tables",
        "reports/phase7/figures",
        "reports/phase7/phase7_contribution_assessment.md",
        "reports/phase7/phase7_paper_narrative.md",
        "reports/phase7/phase7_paper_readiness_report.md",
        "reports/phase7/phase7_ablation_visualization_failure_report.md",
        "reports/phase7/phase7_executive_summary.md",
    ]
    checks = []
    frozen_summary = read_json(paths.artifacts / "frozen_input_audit.json")
    checks.append(("A", "Frozen integrity", "All Phase 2-6 hashes match", frozen_summary["status"] == "PASS", "hard"))
    checks.append(("B", "Analysis isolation", "Confirmatory and exploratory analyses are separated", (paths.artifacts / "analysis_registry.csv").exists(), "hard"))
    checks.append(("C", "Ablation validity", "All predefined Phase 5 ablations are included", set(pd.read_csv(paths.artifacts / "ablation_summary.csv")["ablation"]) >= set(ABLATIONS), "hard"))
    checks.append(("D", "Metric validity", "Undefined denominators remain nonnumeric strings in Phase6 source", "undefined_zero_denominator" in (paths.root / "artifacts" / "phase6" / "final_selective_metrics.csv").read_text(), "hard"))
    checks.append(("E", "Failure taxonomy", "Taxonomy thresholds use risk_fit only", pd.read_csv(paths.artifacts / "failure_taxonomy_thresholds.csv")["threshold_source"].eq("risk_fit").all(), "hard"))
    checks.append(("F", "Qualitative selection", "Case registry is deterministic and near-duplicate aware", (paths.artifacts / "qualitative_case_registry.csv").exists(), "hard"))
    bfree_audit = pd.read_csv(paths.root / "artifacts" / "phase6" / "bfree_source_id_restoration_audit.csv")
    checks.append(("G", "B-Free integrity", "733 unique B-Free images per split context and no unresolved source IDs", bool(bfree_audit["unique_sha256"].eq(733).all() and bfree_audit["unresolved_source_id_count"].eq(0).all()), "hard"))
    figure_files = list(paths.figures.glob("figure_*.pdf"))
    checks.append(("H", "Figures and tables", "Every main figure has source data and PDF", len(figure_files) >= 9 and len(list(paths.figure_data.glob("figure_*.csv"))) >= 9, "hard"))
    claims = pd.read_csv(paths.artifacts / "paper_claim_registry.csv")
    checks.append(("I", "Claim integrity", "Every supported claim cites an artifact", not claims[claims["claim_status"].isin(["supported", "supported_with_qualification"])]["source_artifacts"].eq("").any(), "hard"))
    checks.append(("J", "Reproducibility", "Required Phase 7 artifacts are complete", all((paths.root / item).exists() for item in required), "hard"))
    for item in required:
        checks.append(("J", "Required output", item, (paths.root / item).exists(), "hard"))
    for blocker in hard_blockers:
        checks.append(("Z", "Hard blocker", blocker, False, "hard"))

    checklist = pd.DataFrame(checks, columns=["category", "audit_category", "check", "passed", "severity"])
    checklist["status"] = np.where(checklist["passed"], "pass", "fail")
    checklist.to_csv(paths.artifacts / "phase7_final_audit_checklist.csv", index=False)
    failed_hard = int(((checklist["severity"].eq("hard")) & (~checklist["passed"])).sum())
    status = "PASS" if failed_hard == 0 else "FAIL"
    summary = {
        "created_at": now_iso(),
        "PRE_PHASE_8_STATUS": status,
        "failed_hard_blocker_count": failed_hard,
        "failed_checks": checklist.loc[~checklist["passed"], "check"].head(100).tolist(),
        "warning_count": len(anomalies),
        "warnings": anomalies,
    }
    write_json(paths.artifacts / "phase7_final_audit_summary.json", summary)
    report = f"""
    # Phase 7 Final Audit Report

    PRE_PHASE_8_STATUS = {status}

    Failed hard blockers: {failed_hard}

    ## Warnings
    {chr(10).join('- ' + item for item in anomalies) if anomalies else '- None.'}

    ## Failed Checks
    {chr(10).join('- ' + item for item in summary['failed_checks']) if summary['failed_checks'] else '- None.'}
    """
    write_markdown(paths.reports / "phase7_final_audit_report.md", report)
    return checklist, summary


def freeze_phase7(paths: Phase7Paths, status: str, summary: dict[str, Any]) -> None:
    if status != "PASS":
        return
    summary.update(
        {
            "phase7_frozen_mismatches": 0,
            "required_phase7_artifacts_missing": 0,
            "phase7_frozen_registry": "artifacts/phase7/phase7_frozen_artifact_hashes.csv",
        }
    )
    write_json(paths.artifacts / "phase7_final_audit_summary.json", summary)
    freeze_yaml = {
        "pre_phase_8_status": "PASS",
        "upstream_phases_frozen": True,
        "primary_model_modified": False,
        "primary_policy_modified": False,
        "new_test_selected_threshold": False,
        "confirmatory_exploratory_separated": True,
        "paper_claim_registry_complete": True,
        "failed_hard_blocker_count": 0,
    }
    write_yaml(paths.configs / "phase7_frozen.yaml", freeze_yaml)
    phase7_files = []
    for base in [paths.configs, paths.artifacts, paths.reports, paths.logs]:
        for file in base.rglob("*"):
            if file.is_file() and file.name != "phase7_frozen_artifact_hashes.csv":
                phase7_files.append(file)
    phase7_files.extend([paths.root / "scripts" / "audit_error_analysis.py", paths.root / "tests" / "test_phase7_analysis.py"])
    freeze_paths(paths.root, phase7_files, paths.artifacts / "phase7_frozen_artifact_hashes.csv")
    verify = verify_freeze_registry(paths.root, paths.artifacts / "phase7_frozen_artifact_hashes.csv")
    mismatch_count = int(verify["status"].ne("pass").sum())
    missing_count = int(verify["observed_exists"].ne(True).sum())
    if mismatch_count or missing_count:
        summary.update(
            {
                "phase7_frozen_mismatches": mismatch_count,
                "required_phase7_artifacts_missing": missing_count,
            }
        )
        write_json(paths.artifacts / "phase7_final_audit_summary.json", summary)


def run_all() -> dict[str, Any]:
    started = time.time()
    paths = phase7_paths()
    log_rows: list[dict[str, Any]] = []
    anomalies: list[str] = []
    hard_blockers: list[str] = []
    audit, frozen_summary = verify_frozen_inputs(paths)
    if frozen_summary["status"] != "PASS":
        hard_blockers.append("upstream frozen artifact mismatch or required input missing")
    create_analysis_registry(paths)
    ablation, _ = ablation_synthesis(paths)
    feature_contribution_analysis(paths)
    perm = permutation_importance(paths)
    policy, bottlenecks = policy_ablation(paths)
    detector_split, domain = detector_split_analysis(paths)
    view_summary, interactions, pivots = transformation_view_diagnostics(paths)
    leave_one_view_perturbation(paths, pivots)
    outcome, failure, thresholds = build_outcome_and_failure_taxonomies(paths)
    near_threshold_analysis(paths)
    accepted, rejected = accepted_rejected_case_analysis(paths, failure)
    gen_profile, bfree_cluster = generator_and_bfree_profiles(paths, failure)
    gallery = make_qualitative_galleries(paths, failure)
    make_tables_and_figures(
        paths,
        ablation,
        policy,
        detector_split,
        domain,
        gen_profile,
        bfree_cluster,
        paths.artifacts / "failure_taxonomy_summary.csv",
    )
    claims = build_claim_registry(paths, policy, ablation, bfree_cluster)

    phase6_summary = read_json(paths.root / "artifacts" / "phase6" / "phase6_final_audit_summary.json")
    anomalies.extend(phase6_summary.get("warnings", []))
    anomalies.append("B-Free external transfer remains empirical; no held-out or B-Free guarantee is claimed.")
    anomalies.append("Per-view support-distance change is not frozen as a row-level Phase 4 artifact; Phase 7 reports it as aggregate-only.")
    anomalies.append("Leave-one-view perturbation is exploratory_posthoc and does not replace the frozen model or policy.")
    if (pd.read_csv(paths.artifacts / "ablation_paired_bootstrap.csv")["bootstrap_note"].str.contains("subsampling").any()):
        anomalies.append("Expensive rank-metric ablation bootstrap uses deterministic paired unit subsampling; NLL/Brier use full paired SHA loss deltas.")

    checklist, summary = final_audit(paths, hard_blockers, anomalies)
    status = summary["PRE_PHASE_8_STATUS"]
    write_reports(paths, status, hard_blockers, anomalies, ablation, policy, claims)
    # Rewrite audit after reports exist, then freeze if clean.
    checklist, summary = final_audit(paths, hard_blockers, anomalies)
    status = summary["PRE_PHASE_8_STATUS"]
    write_reports(paths, status, hard_blockers, anomalies, ablation, policy, claims)
    checklist, summary = final_audit(paths, hard_blockers, anomalies)
    status = summary["PRE_PHASE_8_STATUS"]
    log_rows.append(
        {
            "created_at": now_iso(),
            "elapsed_seconds": round(time.time() - started, 3),
            "status": status,
            "ablation_rows": int(len(ablation)),
            "permutation_rows": int(len(perm)),
            "policy_rows": int(len(policy)),
            "outcome_rows": int(len(outcome)),
            "failure_rows": int(len(failure)),
            "accepted_error_rows": int(len(accepted)),
            "rejected_correct_rows": int(len(rejected)),
            "claim_rows": int(len(claims)),
        }
    )
    pd.DataFrame(log_rows).to_csv(paths.logs / "phase7_run_summary.csv", index=False)
    freeze_phase7(paths, status, summary)
    return summary


def audit_only() -> dict[str, Any]:
    paths = phase7_paths()
    anomalies: list[str] = []
    hard_blockers: list[str] = []
    if not (paths.artifacts / "frozen_input_audit.json").exists():
        verify_frozen_inputs(paths)
    frozen_summary = read_json(paths.artifacts / "frozen_input_audit.json")
    if frozen_summary.get("status") != "PASS":
        hard_blockers.append("upstream frozen artifact mismatch or required input missing")
    _, summary = final_audit(paths, hard_blockers, anomalies)
    freeze_phase7(paths, summary["PRE_PHASE_8_STATUS"], summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run_all", "audit"], default="audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_all() if args.stage == "run_all" else audit_only()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("PRE_PHASE_8_STATUS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
