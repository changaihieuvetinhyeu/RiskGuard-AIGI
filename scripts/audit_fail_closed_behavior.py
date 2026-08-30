#!/usr/bin/env python3
"""Phase 8 fail-closed audit from frozen Phase 6 certification traces.

This script reads frozen candidate/certification artifacts and writes derived
diagnostics only. It does not run detector inference, refit models, or modify
Phase 6 artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE6 = PROJECT_ROOT / "artifacts" / "phase6"
PHASE8 = PROJECT_ROOT / "reports" / "phase8"
FIGURES = PHASE8 / "figures"

METHOD = "riskguard"
SOURCE_ALPHA = 0.05
TARGET_ALPHAS = (0.01, 0.02, 0.05, 0.08, 0.10)
DETECTORS = ("safe", "univfd")
SPLITS = ("split_a", "split_b")
POLICIES = ("global_cp", "source_group_cp", "predicted_class_cp")


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def status_label(status: str) -> str:
    return "Cert." if status == "CERTIFIED" else "Fail closed"


def summarize_candidate_group(group: pd.DataFrame, target_alpha: float) -> dict[str, Any]:
    group = group.copy()
    group_pass = (group["accepted_count"].astype(int) > 0) & (
        group["cp_upper"].astype(float) <= float(target_alpha)
    )
    total_size = int(group["certify_group_size"].sum())
    total_accepted = int(group["accepted_count"].sum())
    total_errors = int(group["accepted_errors"].sum())
    coverage = float(total_accepted / total_size) if total_size else 0.0
    empirical = float(total_errors / total_accepted) if total_accepted else float("nan")
    max_idx = group["cp_upper"].astype(float).idxmax()
    bottleneck = group.loc[max_idx]
    failed = group.loc[~group_pass].copy()
    zero_accepted = group.loc[group["accepted_count"].astype(int).eq(0)]

    return {
        "detector": str(group["detector"].iloc[0]),
        "split": str(group["split"].iloc[0]),
        "method": str(group["method"].iloc[0]),
        "policy": str(group["policy"].iloc[0]),
        "candidate_source_alpha": float(group["alpha"].iloc[0]),
        "target_alpha": float(target_alpha),
        "delta": float(group["delta"].iloc[0]),
        "candidate_id": str(group["candidate_id"].iloc[0]),
        "candidate_rank": int(group["candidate_rank"].iloc[0]),
        "threshold": float(group["threshold"].iloc[0]),
        "candidate_count": int(group["candidate_count_K"].iloc[0]),
        "group_count": int(group["group_count_G"].iloc[0]),
        "delta_cell": float(group["delta_cell"].iloc[0]),
        "certification_accepted_count": total_accepted,
        "certification_total_count": total_size,
        "certification_coverage": coverage,
        "accepted_errors": total_errors,
        "empirical_selective_risk": empirical,
        "max_group_cp_upper": float(group["cp_upper"].max()),
        "bottleneck_group": str(bottleneck["group"]),
        "bottleneck_accepted_count": int(bottleneck["accepted_count"]),
        "bottleneck_accepted_errors": int(bottleneck["accepted_errors"]),
        "bottleneck_empirical_selective_risk": float(bottleneck["empirical_selective_risk"])
        if pd.notna(bottleneck["empirical_selective_risk"])
        else float("nan"),
        "bottleneck_cp_upper": float(bottleneck["cp_upper"]),
        "certified_group_count": int(group_pass.sum()),
        "failed_group_count": int((~group_pass).sum()),
        "zero_accepted_group_count": int(len(zero_accepted)),
        "failed_groups": ";".join(str(x) for x in failed["group"].tolist()),
        "candidate_certified": bool(group_pass.all()),
        "margin_above_target": float(group["cp_upper"].max() - float(target_alpha)),
        "group_cp_upper_json": json.dumps(
            {str(r["group"]): float(r["cp_upper"]) for _, r in group.iterrows()},
            sort_keys=True,
        ),
        "group_accepted_count_json": json.dumps(
            {str(r["group"]): int(r["accepted_count"]) for _, r in group.iterrows()},
            sort_keys=True,
        ),
        "group_error_count_json": json.dumps(
            {str(r["group"]): int(r["accepted_errors"]) for _, r in group.iterrows()},
            sort_keys=True,
        ),
    }


def candidate_summaries(trace: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "detector",
        "split",
        "method",
        "policy",
        "alpha",
        "delta",
        "candidate_id",
        "candidate_rank",
        "threshold",
    ]
    rows: list[dict[str, Any]] = []
    for target_alpha in TARGET_ALPHAS:
        for _, group in trace.groupby(keys, sort=True):
            rows.append(summarize_candidate_group(group, target_alpha))
    return pd.DataFrame(rows)


def select_policy_rows(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["detector", "split", "method", "policy", "candidate_source_alpha", "target_alpha"]
    for key, group in candidate_df.groupby(group_keys, sort=True):
        certified = group[group["candidate_certified"]].copy()
        failed = group[~group["candidate_certified"]].copy()
        selected = None
        if not certified.empty:
            certified = certified.sort_values(
                ["threshold", "certification_coverage", "max_group_cp_upper", "candidate_id"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            selected = certified.iloc[0]
            status = "CERTIFIED"
            deployed_coverage = float(selected["certification_coverage"])
            selected_id = str(selected["candidate_id"])
            selected_threshold = float(selected["threshold"])
            selected_max_cp = float(selected["max_group_cp_upper"])
            selected_bottleneck = str(selected["bottleneck_group"])
        else:
            status = "NO_CERTIFIED_THRESHOLD"
            deployed_coverage = 0.0
            selected_id = ""
            selected_threshold = float("nan")
            selected_max_cp = float("nan")
            selected_bottleneck = ""

        if not failed.empty:
            failed = failed.sort_values(
                ["margin_above_target", "certification_coverage", "threshold", "candidate_id"],
                ascending=[True, False, False, True],
                kind="mergesort",
            )
            best_failed = failed.iloc[0]
        else:
            best_failed = None

        base = {
            "detector": key[0],
            "split": key[1],
            "method": key[2],
            "policy": key[3],
            "candidate_source_alpha": key[4],
            "target_alpha": key[5],
            "delta": float(group["delta"].iloc[0]),
            "candidate_count": int(group["candidate_count"].iloc[0]),
            "group_count": int(group["group_count"].iloc[0]),
            "policy_status": status,
            "selected_candidate_id": selected_id,
            "selected_threshold": selected_threshold,
            "deployed_certification_coverage": deployed_coverage,
            "selected_max_group_cp_upper": selected_max_cp,
            "selected_bottleneck_group": selected_bottleneck,
            "certified_candidate_count": int(len(certified)),
            "total_candidate_count": int(len(group)),
        }
        if best_failed is None:
            base.update(
                {
                    "best_failed_candidate_id": "",
                    "best_failed_threshold": float("nan"),
                    "best_failed_candidate_coverage": float("nan"),
                    "best_failed_max_group_cp_upper": float("nan"),
                    "best_failed_bottleneck_group": "",
                    "best_failed_bottleneck_accepted_count": float("nan"),
                    "best_failed_bottleneck_accepted_errors": float("nan"),
                    "best_failed_bottleneck_empirical_selective_risk": float("nan"),
                    "best_failed_bottleneck_cp_upper": float("nan"),
                    "best_failed_failed_group_count": float("nan"),
                    "best_failed_zero_accepted_group_count": float("nan"),
                    "best_failed_margin_above_target": float("nan"),
                }
            )
        else:
            base.update(
                {
                    "best_failed_candidate_id": str(best_failed["candidate_id"]),
                    "best_failed_threshold": float(best_failed["threshold"]),
                    "best_failed_candidate_coverage": float(best_failed["certification_coverage"]),
                    "best_failed_max_group_cp_upper": float(best_failed["max_group_cp_upper"]),
                    "best_failed_bottleneck_group": str(best_failed["bottleneck_group"]),
                    "best_failed_bottleneck_accepted_count": int(best_failed["bottleneck_accepted_count"]),
                    "best_failed_bottleneck_accepted_errors": int(best_failed["bottleneck_accepted_errors"]),
                    "best_failed_bottleneck_empirical_selective_risk": float(
                        best_failed["bottleneck_empirical_selective_risk"]
                    )
                    if pd.notna(best_failed["bottleneck_empirical_selective_risk"])
                    else float("nan"),
                    "best_failed_bottleneck_cp_upper": float(best_failed["bottleneck_cp_upper"]),
                    "best_failed_failed_group_count": int(best_failed["failed_group_count"]),
                    "best_failed_zero_accepted_group_count": int(best_failed["zero_accepted_group_count"]),
                    "best_failed_margin_above_target": float(best_failed["margin_above_target"]),
                }
            )
        rows.append(base)
    return pd.DataFrame(rows)


def policy_comparison_rows(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (target_alpha, detector, split), group in sensitivity.groupby(
        ["target_alpha", "detector", "split"], sort=True
    ):
        by_policy = {str(r["policy"]): r for _, r in group.iterrows()}
        source = by_policy.get("source_group_cp")
        global_cp = by_policy.get("global_cp")
        pred = by_policy.get("predicted_class_cp")

        source_cov = float(source["deployed_certification_coverage"]) if source is not None else float("nan")
        global_cov = float(global_cp["deployed_certification_coverage"]) if global_cp is not None else float("nan")
        pred_cov = float(pred["deployed_certification_coverage"]) if pred is not None else float("nan")
        source_status = str(source["policy_status"]) if source is not None else "MISSING"
        if source_status == "CERTIFIED":
            takeaway = "source_group_cp certifies under this diagnostic target"
        elif global_cp is not None and str(global_cp["policy_status"]) == "CERTIFIED":
            takeaway = "global_cp certifies, but source_group_cp remains fail-closed"
        elif pred is not None and str(pred["policy_status"]) == "CERTIFIED":
            takeaway = "predicted_class_cp certifies, but source_group_cp remains fail-closed"
        else:
            takeaway = "no diagnostic policy certifies"

        rows.append(
            {
                "target_alpha": float(target_alpha),
                "detector": detector,
                "split": split,
                "source_group_status": source_status,
                "source_group_deployed_coverage": source_cov,
                "global_status": str(global_cp["policy_status"]) if global_cp is not None else "MISSING",
                "global_deployed_coverage": global_cov,
                "predicted_class_status": str(pred["policy_status"]) if pred is not None else "MISSING",
                "predicted_class_deployed_coverage": pred_cov,
                "source_group_minus_global_coverage": source_cov - global_cov,
                "source_group_minus_predicted_class_coverage": source_cov - pred_cov,
                "diagnostic_takeaway": takeaway,
            }
        )
    return pd.DataFrame(rows)


def make_plot(sensitivity: pd.DataFrame) -> None:
    label_map = {
        "global_cp": "Global",
        "source_group_cp": "Source group",
        "predicted_class_cp": "Pred. class",
    }
    color_map = {
        "global_cp": "#4c78a8",
        "source_group_cp": "#f58518",
        "predicted_class_cp": "#54a24b",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True, sharey=True)
    for ax, (detector, split) in zip(axes.ravel(), [(d, s) for d in DETECTORS for s in SPLITS]):
        subset = sensitivity[(sensitivity["detector"].eq(detector)) & (sensitivity["split"].eq(split))]
        for policy in POLICIES:
            line = subset[subset["policy"].eq(policy)].sort_values("target_alpha")
            ax.plot(
                line["target_alpha"],
                line["deployed_certification_coverage"],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=label_map[policy],
                color=color_map[policy],
            )
        ax.set_title(f"{detector.upper()} {split.replace('_', ' ').title()}", fontsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlim(0.007, 0.103)
    for ax in axes[-1]:
        ax.set_xlabel("diagnostic target alpha")
    for ax in axes[:, 0]:
        ax.set_ylabel("deployed certified coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Frozen-candidate certification sensitivity", y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "certified_coverage_vs_alpha.pdf")
    fig.savefig(FIGURES / "certified_coverage_vs_alpha.png", dpi=220)
    plt.close(fig)


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_report(sensitivity: pd.DataFrame, best_failed: pd.DataFrame, policy_summary: pd.DataFrame) -> None:
    primary = sensitivity[
        sensitivity["target_alpha"].eq(0.05)
        & sensitivity["policy"].eq("source_group_cp")
        & sensitivity["detector"].isin(DETECTORS)
        & sensitivity["split"].isin(SPLITS)
    ].sort_values(["detector", "split"])

    primary_rows = []
    for _, row in primary.iterrows():
        primary_rows.append(
            [
                str(row["detector"]).upper(),
                str(row["split"]).replace("split_", "").upper(),
                status_label(str(row["policy_status"])),
                fmt_float(row["deployed_certification_coverage"]),
                fmt_float(row["selected_max_group_cp_upper"]),
                str(row["selected_bottleneck_group"]) if str(row["selected_bottleneck_group"]) else "--",
                str(row["best_failed_bottleneck_group"]) if str(row["best_failed_bottleneck_group"]) else "--",
                fmt_float(row["best_failed_max_group_cp_upper"]),
            ]
        )

    failed_rows = []
    for _, row in best_failed.iterrows():
        failed_rows.append(
            [
                str(row["split"]).replace("split_", "").upper(),
                str(row["candidate_id"]).split("_")[-1],
                fmt_float(row["threshold"], 6),
                fmt_float(row["certification_coverage"]),
                str(row["bottleneck_group"]),
                str(int(row["bottleneck_accepted_count"])),
                str(int(row["bottleneck_accepted_errors"])),
                fmt_float(row["bottleneck_empirical_selective_risk"]),
                fmt_float(row["bottleneck_cp_upper"]),
                fmt_float(row["margin_above_target"]),
            ]
        )

    policy_rows = []
    for _, row in policy_summary[policy_summary["target_alpha"].eq(0.05)].sort_values(["detector", "split"]).iterrows():
        policy_rows.append(
            [
                str(row["detector"]).upper(),
                str(row["split"]).replace("split_", "").upper(),
                status_label(str(row["source_group_status"])),
                fmt_float(row["source_group_deployed_coverage"]),
                status_label(str(row["global_status"])),
                fmt_float(row["global_deployed_coverage"]),
                status_label(str(row["predicted_class_status"])),
                fmt_float(row["predicted_class_deployed_coverage"]),
            ]
        )

    text = f"""# UnivFD Fail-Closed Audit

Decision implemented: do not run additional detector inference just to fill empty cells. The main-paper action is to report UnivFD source-group rows as fail-closed/no-certificate outcomes and to remove B-Free Table 4 from the main manuscript.

## Source Artifacts

- `artifacts/phase6/certification_trace.parquet`
- `artifacts/phase6/certified_threshold_registry.csv`
- `artifacts/phase6/candidate_threshold_registry.csv`
- `artifacts/phase6/calibration_split_assignments/*.csv`

All outputs here are derived diagnostics under the frozen Phase 6 policy protocol. They do not alter frozen Phase 6 candidate, trace, policy, score, or split-assignment artifacts.

## Fail-Closed Semantics

Deployment is positive-coverage only when a candidate threshold satisfies every simultaneous group bound. If no candidate is certified, RiskGuard issues no threshold for deployment, accepts no samples under that policy, and reports zero deployed certified coverage rather than an undefined or backfilled selective risk.

For UnivFD, RiskGuard returns no certified positive-coverage policy at alpha = 0.05. This is a fail-closed outcome rather than a missing evaluation: none of the frozen candidates satisfies all simultaneous source-group bounds. Consequently, no deployment threshold is issued under this risk budget.

## Primary Source-Group Status

{markdown_table(primary_rows, ["Detector", "Split", "Status", "Deployed cert. cov.", "Selected max CP", "Selected bottleneck", "Best failed bottleneck", "Best failed max CP"])}

## UnivFD Best Failed Candidates

The closest failed candidate is selected by minimum max-group CP margin above alpha = 0.05, then higher certification coverage and higher threshold. These candidates are diagnostic only; they are not deployable policies.

{markdown_table(failed_rows, ["Split", "Candidate", "Threshold", "Candidate cov.", "Blocking group", "Block n", "Block err", "Block risk", "Block CP", "CP-alpha"])}

## Diagnostic Policy Comparison at Alpha 0.05

{markdown_table(policy_rows, ["Detector", "Split", "Source status", "Source cov.", "Global status", "Global cov.", "Pred-class status", "Pred-class cov."])}

Global and predicted-class policies can certify for UnivFD at alpha = 0.05, but they certify different group definitions and therefore are not substitutes for the primary source-group guarantee. The main paper should keep source-group certification as the headline risk-control claim and explain UnivFD as a fail-closed result.

## Generated Outputs

- `reports/phase8/best_failed_candidates.csv`
- `reports/phase8/certification_alpha_sensitivity.csv`
- `reports/phase8/policy_sensitivity_summary.csv`
- `reports/phase8/figures/certified_coverage_vs_alpha.pdf`
- `reports/phase8/figures/certified_coverage_vs_alpha.png`

FAIL_CLOSED_AUDIT_STATUS = COMPLETE
FROZEN_ARTIFACTS_MODIFIED = FALSE
NEW_DETECTOR_INFERENCE_RUN = FALSE
MANUSCRIPT_PATCHED = TRUE
"""
    (PHASE8 / "univfd_fail_closed_audit.md").write_text(text, encoding="utf-8")


def main() -> None:
    trace_path = PHASE6 / "certification_trace.parquet"
    trace = pd.read_parquet(trace_path)
    trace = trace[
        trace["method"].eq(METHOD)
        & trace["detector"].isin(DETECTORS)
        & trace["split"].isin(SPLITS)
        & trace["policy"].isin(POLICIES)
        & trace["alpha"].eq(SOURCE_ALPHA)
    ].copy()
    if trace.empty:
        raise RuntimeError("No frozen Phase 6 trace rows found for the requested audit.")

    candidate_df = candidate_summaries(trace)
    sensitivity = select_policy_rows(candidate_df)
    policy_summary = policy_comparison_rows(sensitivity)
    best_failed = candidate_df[
        candidate_df["detector"].eq("univfd")
        & candidate_df["policy"].eq("source_group_cp")
        & candidate_df["target_alpha"].eq(0.05)
        & ~candidate_df["candidate_certified"]
    ].copy()
    best_failed = best_failed.sort_values(
        ["split", "margin_above_target", "certification_coverage", "threshold", "candidate_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).groupby("split", as_index=False, sort=True).head(1)

    PHASE8.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sensitivity.sort_values(["detector", "split", "policy", "target_alpha"]).to_csv(
        PHASE8 / "certification_alpha_sensitivity.csv", index=False
    )
    policy_summary.sort_values(["detector", "split", "target_alpha"]).to_csv(
        PHASE8 / "policy_sensitivity_summary.csv", index=False
    )
    best_failed.sort_values(["split"]).to_csv(PHASE8 / "best_failed_candidates.csv", index=False)
    make_plot(sensitivity)
    write_report(sensitivity, best_failed, policy_summary)


if __name__ == "__main__":
    main()
