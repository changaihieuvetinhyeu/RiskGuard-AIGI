#!/usr/bin/env python3
"""Build the Phase 8 SAFE margin-only certification comparison.

This script does not run the base detectors. It fits the predefined
margin-only logistic ablation on frozen Phase 4 risk_fit features using the
Phase 5 ablation-selected C value, then applies the same Phase 6
policy_select/policy_certify source-group certification procedure on frozen
threshold_cal features.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from selective_detection.exact_binomial_bound import clopper_pearson_upper
from selective_detection.policy_artifact_io import sha256_file
from selective_detection.group_risk_certification import (
    CANDIDATE_COUNT,
    DELTA,
    alpha_slug,
    policy_groups,
    threshold_for_target_count,
    threshold_scan,
)
from selective_detection.error_probability_calibrator import transform_features


DETECTOR = "safe"
SPLITS = ("split_a", "split_b")
POLICY = "source_group_cp"
ALPHA = 0.05
FEATURE_ORDER = ("margin_distance",)
SEED = 20260916


def stable_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_path(split: str, partition: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "phase4" / "features" / DETECTOR / split / f"{partition}.parquet"


def fit_margin_only_model(split: str) -> dict[str, Any]:
    registry = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase5" / "ablation_model_registry.csv")
    row = registry[
        registry["detector"].eq(DETECTOR)
        & registry["split"].eq(split)
        & registry["ablation"].eq("margin_only")
    ]
    if len(row) != 1:
        raise RuntimeError(f"Expected one margin_only registry row for {DETECTOR}/{split}, found {len(row)}")
    selected_c = float(row.iloc[0]["selected_C"])

    risk_fit = pd.read_parquet(feature_path(split, "risk_fit"))
    transformed = transform_features(risk_fit.loc[:, list(FEATURE_ORDER)], FEATURE_ORDER, as_frame=False)
    means = transformed.mean(axis=0)
    scales = transformed.std(axis=0, ddof=0)
    if (scales <= 0.0).any():
        raise RuntimeError(f"Non-positive margin-only scale for {split}: {scales.tolist()}")

    z = (transformed - means) / scales
    y = risk_fit["base_error"].to_numpy(dtype=np.int64)
    clf = LogisticRegression(
        C=selected_c,
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        max_iter=5000,
        tol=1.0e-10,
        random_state=SEED,
    )
    clf.fit(z, y)

    payload: dict[str, Any] = {
        "model_version": "phase8_margin_only_logistic_v1",
        "detector": DETECTOR,
        "split": split,
        "feature_order": list(FEATURE_ORDER),
        "selected_C": selected_c,
        "scaler_means": [float(v) for v in means],
        "scaler_scales": [float(v) for v in scales],
        "coefficient_vector": [float(v) for v in clf.coef_.reshape(-1)],
        "intercept": float(clf.intercept_[0]),
        "risk_fit_row_count": int(len(risk_fit)),
        "risk_fit_unique_sha256": int(risk_fit["sha256"].nunique()),
        "risk_fit_error_count": int(risk_fit["base_error"].sum()),
        "phase4_risk_fit_sha256": sha256_file(feature_path(split, "risk_fit")),
        "phase5_ablation_registry_sha256": sha256_file(
            PROJECT_ROOT / "artifacts" / "phase5" / "ablation_model_registry.csv"
        ),
    }
    payload["model_sha256"] = stable_payload_hash(payload)
    return payload


def score_margin_only(split: str, model: dict[str, Any], partition: str) -> pd.DataFrame:
    df = pd.read_parquet(feature_path(split, partition)).copy()
    transformed = transform_features(df.loc[:, list(FEATURE_ORDER)], FEATURE_ORDER, as_frame=False)
    means = np.asarray(model["scaler_means"], dtype=np.float64)
    scales = np.asarray(model["scaler_scales"], dtype=np.float64)
    coefs = np.asarray(model["coefficient_vector"], dtype=np.float64)
    logits = ((transformed - means) / scales) @ coefs + float(model["intercept"])
    probs = 1.0 / (1.0 + np.exp(-logits))
    out = df[
        [
            "sample_id",
            "sha256",
            "generator",
            "label",
            "base_prediction",
            "base_error",
            "partition",
        ]
    ].copy()
    out["risk_score"] = probs.astype(np.float64)
    out["method"] = "margin_only"
    out["model_sha256"] = str(model["model_sha256"])
    out["score_source"] = f"phase8 margin_only from frozen Phase4 {partition} features"
    out["feature_artifact_sha256"] = sha256_file(feature_path(split, partition))
    return out.sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def candidate_id(split: str, rank: int) -> str:
    return f"{DETECTOR}_{split}_margin_only_{alpha_slug(ALPHA)}_{POLICY}_C{rank:02d}"


def generate_candidates(split: str, threshold_cal_scores: pd.DataFrame) -> pd.DataFrame:
    assignments = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase6" / "calibration_split_assignments" / f"{DETECTOR}_{split}.csv")
    df = threshold_cal_scores.merge(assignments[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
    select = df[df["calibration_subset"].eq("policy_select")].copy()
    curve = threshold_scan(select, POLICY, ALPHA)
    if curve.empty:
        return pd.DataFrame()

    feasible = curve[curve["select_feasible"]].copy()
    rows: list[dict[str, Any]] = []
    if not feasible.empty:
        largest_count = int(feasible["accepted_count"].max())
        fractions = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
        source = "select_feasible_fraction"
    else:
        largest_count = int(curve["accepted_count"].max())
        fractions = [0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.01]
        source = "fallback_select_coverage"

    for fraction in fractions:
        target = max(1, int(np.floor(largest_count * fraction)))
        row = threshold_for_target_count(curve, target)
        if row is None:
            continue
        record = row.to_dict()
        record["candidate_source"] = source
        record["target_fraction"] = float(fraction)
        rows.append(record)

    dedup: dict[float, dict[str, Any]] = {}
    for row in rows:
        dedup.setdefault(float(row["threshold"]), row)
    ordered = sorted(dedup.values(), key=lambda item: (-float(item["threshold"]), -int(item["accepted_count"])))[:CANDIDATE_COUNT]

    out: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        payload = {
            "detector": DETECTOR,
            "split": split,
            "method": "margin_only",
            "alpha": float(ALPHA),
            "delta": float(DELTA),
            "policy": POLICY,
            "candidate_id": candidate_id(split, rank),
            "candidate_rank": int(rank),
            "threshold": float(row["threshold"]),
            "select_accepted_count": int(row["accepted_count"]),
            "select_coverage": float(row["coverage"]),
            "select_error_count": int(row["accepted_errors"]),
            "select_empirical_risk": float(row["empirical_risk"]),
            "candidate_source": str(row["candidate_source"]),
            "target_fraction": float(row["target_fraction"]),
        }
        payload["candidate_sha256"] = stable_payload_hash(payload)
        out.append(payload)
    return pd.DataFrame(out)


def certify(split: str, threshold_cal_scores: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignments = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase6" / "calibration_split_assignments" / f"{DETECTOR}_{split}.csv")
    df = threshold_cal_scores.merge(assignments[["sha256", "calibration_subset"]], on="sha256", how="left", validate="many_to_one")
    certify_df = df[df["calibration_subset"].eq("policy_certify")].copy()
    groups = policy_groups(certify_df, POLICY)
    unique_groups = sorted(groups.unique())
    k_candidates = int(len(candidates))
    group_count = int(len(unique_groups))
    delta_cell = float(DELTA / (k_candidates * group_count)) if k_candidates and group_count else float("nan")

    trace_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for candidate in candidates.sort_values("candidate_rank", kind="mergesort").to_dict("records"):
        threshold = float(candidate["threshold"])
        accepted_all = certify_df["risk_score"].to_numpy(dtype=float) <= threshold
        total_accepted = int(accepted_all.sum())
        group_bounds: dict[str, float] = {}
        group_counts: dict[str, dict[str, Any]] = {}
        all_certified = True
        candidate_trace_rows: list[dict[str, Any]] = []
        for group in unique_groups:
            mask = groups.eq(group).to_numpy()
            accepted = accepted_all & mask
            accepted_count = int(accepted.sum())
            errors = int(certify_df.loc[accepted, "base_error"].astype(int).sum())
            cp_upper = clopper_pearson_upper(errors, accepted_count, delta_cell)
            empirical = float(errors / accepted_count) if accepted_count else float("nan")
            group_certified = bool(accepted_count > 0 and cp_upper <= ALPHA)
            all_certified = all_certified and group_certified
            group_bounds[str(group)] = float(cp_upper)
            group_counts[str(group)] = {
                "accepted_count": accepted_count,
                "accepted_errors": errors,
                "empirical_selective_risk": empirical,
                "cp_upper": float(cp_upper),
                "certified": group_certified,
            }
            candidate_trace_rows.append(
                {
                    "detector": DETECTOR,
                    "split": split,
                    "method": "margin_only",
                    "alpha": float(ALPHA),
                    "delta": float(DELTA),
                    "policy": POLICY,
                    "candidate_id": str(candidate["candidate_id"]),
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "threshold": threshold,
                    "candidate_count_K": k_candidates,
                    "group_count_G": group_count,
                    "delta_cell": delta_cell,
                    "group": str(group),
                    "certify_group_size": int(mask.sum()),
                    "accepted_count": accepted_count,
                    "accepted_errors": errors,
                    "empirical_selective_risk": empirical,
                    "cp_upper": float(cp_upper),
                    "group_certified": group_certified,
                    "candidate_certified": False,
                }
            )
        for row in candidate_trace_rows:
            row["candidate_certified"] = bool(all_certified)
        trace_rows.extend(candidate_trace_rows)
        summaries.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_rank": int(candidate["candidate_rank"]),
                "threshold": threshold,
                "certification_coverage": float(total_accepted / len(certify_df)) if len(certify_df) else 0.0,
                "certification_accepted_count": total_accepted,
                "max_group_cp_upper": max(group_bounds.values()) if group_bounds else float("nan"),
                "candidate_certified": bool(all_certified),
                "group_bounds_json": json.dumps(group_bounds, sort_keys=True),
                "group_counts_json": json.dumps(group_counts, sort_keys=True),
            }
        )

    certified = [row for row in summaries if row["candidate_certified"]]
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
        status = "CERTIFIED"
    else:
        selected = None
        status = "NO_CERTIFIED_THRESHOLD"

    summary = {
        "detector": DETECTOR,
        "split": split,
        "method": "margin_only",
        "alpha": float(ALPHA),
        "delta": float(DELTA),
        "policy": POLICY,
        "policy_status": status,
        "selected_threshold": float(selected["threshold"]) if selected else np.nan,
        "candidate_count": k_candidates,
        "delta_cell": delta_cell,
        "overall_certification_coverage": float(selected["certification_coverage"]) if selected else 0.0,
        "max_group_cp_upper": float(selected["max_group_cp_upper"]) if selected else np.nan,
        "selected_candidate_id": str(selected["candidate_id"]) if selected else "",
        "calibration_split_assignment_sha256": sha256_file(
            PROJECT_ROOT / "artifacts" / "phase6" / "calibration_split_assignments" / f"{DETECTOR}_{split}.csv"
        ),
    }
    return pd.DataFrame(trace_rows), summary


def write_table(registry: pd.DataFrame) -> None:
    phase6 = pd.read_csv(PROJECT_ROOT / "artifacts" / "phase6" / "certified_threshold_registry.csv")
    full = phase6[
        phase6["detector"].eq(DETECTOR)
        & phase6["method"].eq("riskguard")
        & phase6["policy"].eq(POLICY)
        & np.isclose(phase6["alpha"].astype(float), ALPHA)
        & np.isclose(phase6["delta"].astype(float), DELTA)
    ].copy()
    full["risk_score"] = "Full four"
    margin = registry.copy()
    margin["risk_score"] = "Margin only"

    combined = pd.concat([full, margin], ignore_index=True)
    combined["Split"] = combined["split"].map({"split_a": "A", "split_b": "B"})
    combined["Risk score"] = combined["risk_score"]
    combined["Status"] = combined["policy_status"].map(
        {
            "CERTIFIED": "Certified",
            "NO_CERTIFIED_THRESHOLD": "No certified threshold",
        }
    )
    combined["Certified coverage"] = combined["overall_certification_coverage"].astype(float)
    combined["Max group CP"] = combined["max_group_cp_upper"].astype(float)
    table = combined[
        [
            "Split",
            "Risk score",
            "Status",
            "Certified coverage",
            "Max group CP",
            "selected_threshold",
            "candidate_count",
        ]
    ].sort_values(["Split", "Risk score"], kind="mergesort")

    table_dir = PROJECT_ROOT / "reports" / "phase8" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir / "table_margin_only_certification.csv", index=False)


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts" / "phase8"
    model_dir = artifact_dir / "margin_only_models"
    score_dir = artifact_dir / "margin_only_scores" / DETECTOR
    model_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    all_candidates: list[pd.DataFrame] = []
    all_trace: list[pd.DataFrame] = []
    registry_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        model = fit_margin_only_model(split)
        model_path = model_dir / f"{DETECTOR}_{split}_margin_only.json"
        model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        split_score_dir = score_dir / split
        split_score_dir.mkdir(parents=True, exist_ok=True)
        threshold_scores = score_margin_only(split, model, "threshold_cal")
        threshold_scores.to_parquet(split_score_dir / "threshold_cal.parquet", index=False)

        candidates = generate_candidates(split, threshold_scores)
        all_candidates.append(candidates)
        trace, summary = certify(split, threshold_scores, candidates)
        all_trace.append(trace)
        summary["model_json"] = str(model_path.relative_to(PROJECT_ROOT))
        summary["model_json_sha256"] = sha256_file(model_path)
        summary["score_artifact"] = str((split_score_dir / "threshold_cal.parquet").relative_to(PROJECT_ROOT))
        summary["score_artifact_sha256"] = sha256_file(split_score_dir / "threshold_cal.parquet")
        registry_rows.append(summary)

    pd.concat(all_candidates, ignore_index=True).to_csv(artifact_dir / "margin_only_candidate_threshold_registry.csv", index=False)
    pd.concat(all_trace, ignore_index=True).to_parquet(artifact_dir / "margin_only_certification_trace.parquet", index=False)
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(artifact_dir / "margin_only_certification_registry.csv", index=False)
    write_table(registry)


if __name__ == "__main__":
    main()
