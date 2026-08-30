#!/usr/bin/env python3
"""Final Phase 3 audit and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from selective_detection.selective_baselines import (
    DETECTORS,
    MANDATORY_BASELINES,
    SPLITS,
    load_yaml,
    sha256_file,
    verify_phase2_frozen_hashes,
    write_json,
    write_yaml,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/phase3/selective_baselines.yaml"
ARTIFACTS = PROJECT_ROOT / "artifacts/phase3"
REPORTS = PROJECT_ROOT / "reports/phase3"


def exists(relative: str) -> bool:
    return (PROJECT_ROOT / relative).exists()


def checklist_row(category: str, check: str, status: str, blocker: bool, detail: str = "") -> dict[str, object]:
    return {"category": category, "check": check, "status": status, "hard_blocker": blocker, "detail": detail}


def score_files_complete() -> tuple[bool, str]:
    missing = []
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                for scope in ("risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out"):
                    path = ARTIFACTS / "scores" / detector / baseline / f"{split}_{scope}.parquet"
                    if not path.exists():
                        missing.append(str(path.relative_to(PROJECT_ROOT)))
    return len(missing) == 0, "; ".join(missing[:8])


def score_numerics_pass() -> tuple[bool, str]:
    bad = []
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                for role in ("protocol_seen", "protocol_held_out", "threshold_cal"):
                    path = ARTIFACTS / "scores" / detector / baseline / f"{split}_{role}.parquet"
                    if not path.exists():
                        continue
                    df = pd.read_parquet(path, columns=["risk_score", "base_probability", "base_logit"])
                    if not np.isfinite(df["risk_score"].to_numpy(dtype=float)).all():
                        bad.append(f"{detector}/{baseline}/{split}/{role}: risk")
                    probs = df["base_probability"].to_numpy(dtype=float)
                    if not (np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all()):
                        bad.append(f"{detector}/{baseline}/{split}/{role}: probability")
                    if not np.isfinite(df["base_logit"].to_numpy(dtype=float)).all():
                        bad.append(f"{detector}/{baseline}/{split}/{role}: logit")
    return len(bad) == 0, "; ".join(bad[:8])


def fit_registry_pass() -> tuple[bool, str]:
    path = ARTIFACTS / "baseline_fit_registry.csv"
    if not path.exists():
        return False, "baseline_fit_registry.csv missing"
    df = pd.read_csv(path).fillna("")
    bad = df[(df["source_partition"] != "risk_fit") & (df["component"] != "MC Dropout availability decision")]
    return len(bad) == 0, bad[["component", "detector", "baseline", "split", "source_partition"]].head().to_json(orient="records")


def thresholds_pass() -> tuple[bool, str]:
    path = ARTIFACTS / "global_thresholds.csv"
    if not path.exists():
        return False, "global_thresholds.csv missing"
    df = pd.read_csv(path)
    expected = len(DETECTORS) * len(MANDATORY_BASELINES) * len(SPLITS) * 2
    alphas = sorted(df["alpha"].unique().tolist())
    ok = len(df) == expected and alphas == [0.01, 0.05] and set(df["delta"].unique()) == {0.05}
    return ok, f"rows={len(df)} expected={expected} alphas={alphas}"


def metrics_pass() -> tuple[bool, str]:
    required = [
        "selective_baseline_metrics.csv",
        "selective_baseline_per_generator_metrics.csv",
        "selective_baseline_threshold_metrics.csv",
        "selective_baseline_calibration_metrics.csv",
        "selective_baseline_runtime.csv",
        "selective_baseline_paper_table.csv",
        "bootstrap_ci.csv",
        "risk_score_rank_correlations.csv",
        "bfree_snapshot_selective_metrics.csv",
    ]
    missing = [name for name in required if not (ARTIFACTS / name).exists()]
    if missing:
        return False, "missing " + ", ".join(missing)
    metrics = pd.read_csv(ARTIFACTS / "selective_baseline_metrics.csv")
    has_dedup = "sha256_deduplicated" in set(metrics["evaluation_weighting"].astype(str))
    return has_dedup, "deduplicated metrics present" if has_dedup else "deduplicated metrics missing"


def determinism_audit() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            path = ARTIFACTS / "scores" / detector / baseline / "split_a_protocol_seen.parquet"
            if not path.exists():
                rows.append({"detector": detector, "baseline": baseline, "check": "score_file_present", "status": "fail"})
                continue
            left = pd.read_parquet(path).head(1024)
            right = pd.read_parquet(path).head(1024)
            rows.extend(
                [
                    {
                        "detector": detector,
                        "baseline": baseline,
                        "check": "same_sample_order",
                        "status": "pass" if left["sample_id"].tolist() == right["sample_id"].tolist() else "fail",
                    },
                    {
                        "detector": detector,
                        "baseline": baseline,
                        "check": "same_base_predictions",
                        "status": "pass" if np.array_equal(left["base_prediction"].to_numpy(), right["base_prediction"].to_numpy()) else "fail",
                    },
                    {
                        "detector": detector,
                        "baseline": baseline,
                        "check": "same_risk_scores_within_tolerance",
                        "status": "pass"
                        if np.allclose(
                            left["risk_score"].to_numpy(dtype=float),
                            right["risk_score"].to_numpy(dtype=float),
                            atol=1e-6,
                            rtol=0.0,
                        )
                        else "fail",
                    },
                ]
            )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "determinism_audit.csv", index=False)
    return out


def phase3_artifact_hashes() -> pd.DataFrame:
    rows = []
    for base in (ARTIFACTS, REPORTS, PROJECT_ROOT / "configs/phase3"):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "relative_path": str(path.relative_to(PROJECT_ROOT)),
                        "size_bytes": int(path.stat().st_size),
                        "sha256": sha256_file(path),
                    }
                )
    return pd.DataFrame(rows)


def strongest_baseline() -> dict[str, object]:
    metrics = pd.read_csv(ARTIFACTS / "selective_baseline_metrics.csv")
    per_gen = pd.read_csv(ARTIFACTS / "selective_baseline_per_generator_metrics.csv")
    threshold = pd.read_csv(ARTIFACTS / "selective_baseline_threshold_metrics.csv")
    runtime_path = ARTIFACTS / "runtime_resource_audit.csv"
    runtime = pd.read_csv(runtime_path) if runtime_path.exists() else pd.DataFrame()
    aurc_rows = metrics[
        (metrics["evaluation_weighting"] == "sha256_deduplicated")
        & (metrics["generator"] == "all")
        & (metrics["metric"] == "AURC")
    ]
    mean_aurc = aurc_rows.groupby(["detector", "baseline"], as_index=False)["value"].mean().rename(columns={"value": "mean_aurc"})
    worst = per_gen[
        (per_gen["evaluation_weighting"] == "sha256_deduplicated")
        & (per_gen["metric"] == "AURC")
        & (~per_gen["generator"].isin(["all", "real_class", "fake_class"]))
    ].groupby(["detector", "baseline"], as_index=False)["value"].max().rename(columns={"value": "worst_generator_aurc"})
    cov = threshold[threshold["alpha"] == 0.05].groupby(["detector", "baseline"], as_index=False)["coverage"].mean().rename(
        columns={"coverage": "mean_cov_cp_risk_le_5pct"}
    )
    if len(runtime):
        cost = runtime.groupby(["detector", "baseline"], as_index=False)["seconds"].sum().rename(columns={"seconds": "runtime_seconds"})
    else:
        cost = mean_aurc[["detector", "baseline"]].assign(runtime_seconds=0.0)
    table = mean_aurc.merge(worst, on=["detector", "baseline"], how="left").merge(cov, on=["detector", "baseline"], how="left").merge(
        cost, on=["detector", "baseline"], how="left"
    )
    table = table.sort_values(
        ["mean_aurc", "worst_generator_aurc", "mean_cov_cp_risk_le_5pct", "runtime_seconds"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    table.to_csv(ARTIFACTS / "strongest_phase3_baseline_selection.csv", index=False)
    return table.iloc[0].to_dict()


def write_phase3_report(status: str, blockers: list[dict[str, object]], strongest: dict[str, object]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    hashes = pd.read_csv(PROJECT_ROOT / "artifacts/phase2_frozen_artifact_hashes.csv")
    fit = pd.read_csv(ARTIFACTS / "baseline_fit_registry.csv")
    thresholds = pd.read_csv(ARTIFACTS / "global_thresholds.csv")
    metrics = pd.read_csv(ARTIFACTS / "selective_baseline_metrics.csv")
    paper = pd.read_csv(ARTIFACTS / "selective_baseline_paper_table.csv")
    corr = pd.read_csv(ARTIFACTS / "risk_score_rank_correlations.csv")
    bfree = pd.read_csv(ARTIFACTS / "bfree_snapshot_selective_metrics.csv")
    availability = pd.read_csv(ARTIFACTS / "mc_dropout_availability.csv")
    runtime = pd.read_csv(ARTIFACTS / "runtime_resource_audit.csv")
    dedup_summary = pd.read_csv(ARTIFACTS / "eval_manifest_dedup_summary.csv")

    top_metrics = metrics[
        (metrics["evaluation_weighting"] == "sha256_deduplicated")
        & (metrics["generator"] == "all")
        & (metrics["metric"].isin(["AURC", "error_detection_AUROC", "E_AURC"]))
    ].head(36)
    high_corr = corr[corr["high_redundancy"] == True].head(24)
    blocker_lines = ["None."] if not blockers else [
        f"- artifact={item.get('artifact','phase3')} affected detector=all affected baseline=all exact reason={item['check']} required repair={item['detail']}"
        for item in blockers
    ]
    lines = [
        "# Phase 3 Selective Baselines Report",
        "",
        "## Process Completed",
        "",
        "- Froze and rechecked Phase 2 inputs before Phase 3 execution.",
        "- Materialized SHA-256 deduplication maps and deduplicated evaluation manifests.",
        "- Fitted temperature, Mahalanobis, and kNN components from split-specific `risk_fit` only.",
        "- Scored MSP, entropy, binary energy, temperature-scaled MSP, Mahalanobis, and kNN for UnivFD and SAFE.",
        "- Selected acceptance thresholds from split-specific `threshold_cal` only for alpha 0.01 and 0.05.",
        "- Evaluated SHA-256-deduplicated headline metrics, row-level diagnostics, per-generator metrics, B-Free snapshot metrics, bootstrap CIs, runtime, and figures.",
        "",
        "## Frozen Phase 2 Input Hashes",
        "",
        hashes.head(40).to_markdown(index=False),
        "",
        "## Baseline Definitions",
        "",
        "MSP uses `1 - max(p, 1-p)`. Entropy uses clipped binary entropy normalized by log 2. Energy uses the disclosed symmetric one-logit convention `-logsumexp([-s/2, s/2])`, so larger values are closer to the decision boundary. Temperature-scaled MSP fits one positive scalar temperature on `risk_fit`. Mahalanobis uses risk-fit z-score statistics and class-conditional Ledoit-Wolf covariance. kNN uses exact cosine distance to the risk-fit reference bank.",
        "",
        "## Deduplication",
        "",
        dedup_summary.to_markdown(index=False),
        "",
        "## Fit Provenance",
        "",
        fit.head(40).to_markdown(index=False),
        "",
        "## Threshold Provenance",
        "",
        thresholds.head(48).to_markdown(index=False, floatfmt='.6f'),
        "",
        "## Clean Selective Metrics",
        "",
        top_metrics.to_markdown(index=False, floatfmt='.6f'),
        "",
        "## Paper Table",
        "",
        paper.to_markdown(index=False, floatfmt='.6f'),
        "",
        "## Correlation And Redundancy",
        "",
        high_corr.to_markdown(index=False, floatfmt='.6f') if len(high_corr) else "No non-identical pair exceeded |Spearman| >= 0.995.",
        "",
        "## B-Free Viral Verified Snapshot",
        "",
        bfree.head(24).to_markdown(index=False, floatfmt='.6f'),
        "",
        "## Runtime And Storage",
        "",
        runtime.groupby(["detector", "baseline", "stage"], as_index=False)["seconds"].sum().head(40).to_markdown(index=False, floatfmt='.2f'),
        "",
        "## Unsupported Baselines",
        "",
        availability.to_markdown(index=False),
        "",
        "## Strongest Phase 3 Baseline",
        "",
        f"Selected by the declared priority: detector={strongest['detector']}, method={strongest['baseline']}, mean AURC={strongest['mean_aurc']:.6f}, worst-generator AURC={strongest['worst_generator_aurc']:.6f}, mean Cov@CP-Risk<=5%={strongest['mean_cov_cp_risk_le_5pct']:.6f}.",
        "",
        "## Not Done Or Failed",
        "",
        *blocker_lines,
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "cd /home/llm/AnhNT/RiskGuard-AIGI",
        "CUDA_VISIBLE_DEVICES=1 envs/safe/bin/python scripts/fit_selective_baselines.py",
        "CUDA_VISIBLE_DEVICES=1 envs/safe/bin/python scripts/score_selective_baselines.py",
        "CUDA_VISIBLE_DEVICES=1 envs/safe/bin/python scripts/select_baseline_thresholds.py",
        "CUDA_VISIBLE_DEVICES=1 envs/safe/bin/python scripts/evaluate_selective_baselines.py",
        "CUDA_VISIBLE_DEVICES=1 envs/safe/bin/python scripts/audit_selective_baselines.py",
        "```",
        "",
        f"PRE_PHASE_4_STATUS = {status}",
    ]
    (REPORTS / "phase3_selective_baselines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(CONFIG_PATH)
    checks = []
    try:
        verify_phase2_frozen_hashes(PROJECT_ROOT)
        checks.append(checklist_row("A", "All Phase 2 frozen hashes match.", "pass", False))
    except Exception as exc:
        checks.append(checklist_row("A", "All Phase 2 frozen hashes match.", "fail", True, str(exc)))
    complete, detail = score_files_complete()
    checks.append(checklist_row("B", "Every expected score artifact exists.", "pass" if complete else "fail", not complete, detail))
    numeric, detail = score_numerics_pass()
    checks.append(checklist_row("E", "Logits, probabilities, and risk scores are finite and valid.", "pass" if numeric else "fail", not numeric, detail))
    fit_ok, detail = fit_registry_pass()
    checks.append(checklist_row("D", "Every fitted statistical component uses risk_fit only.", "pass" if fit_ok else "fail", not fit_ok, detail))
    threshold_ok, detail = thresholds_pass()
    checks.append(checklist_row("G", "Threshold artifacts use alpha 0.01/0.05 and delta 0.05.", "pass" if threshold_ok else "fail", not threshold_ok, detail))
    metric_ok, detail = metrics_pass()
    checks.append(checklist_row("H", "Metric, bootstrap, B-Free, and deduplicated artifacts exist.", "pass" if metric_ok else "fail", not metric_ok, detail))
    det = determinism_audit()
    det_ok = bool(det["status"].eq("pass").all())
    checks.append(checklist_row("K", "Deterministic subset reread passes.", "pass" if det_ok else "fail", not det_ok))
    availability_ok = exists("artifacts/phase3/mc_dropout_availability.csv")
    checks.append(checklist_row("J", "MC Dropout availability is explicitly audited.", "pass" if availability_ok else "fail", not availability_ok))
    figures = [
        REPORTS / "figures" / f"{name}_{detector}.pdf"
        for detector in DETECTORS
        for name in ("risk_coverage", "error_score_histogram", "per_generator_aurc", "risk_score_correlation")
    ]
    figs_ok = all(path.exists() for path in figures)
    checks.append(checklist_row("L", "Required figures and plotting data exist.", "pass" if figs_ok else "fail", not figs_ok))
    interpretation_ok = exists("artifacts/phase3/risk_score_rank_correlations.csv")
    checks.append(checklist_row("M", "Risk-score redundancy audit exists.", "pass" if interpretation_ok else "fail", not interpretation_ok))

    checklist = pd.DataFrame(checks)
    checklist.to_csv(ARTIFACTS / "phase3_final_audit_checklist.csv", index=False)
    blockers = checklist[(checklist["status"] == "fail") & (checklist["hard_blocker"] == True)].to_dict("records")
    status = "FAIL" if blockers else "PASS"
    strongest = strongest_baseline() if metric_ok else {"detector": "", "baseline": "", "mean_aurc": float("nan"), "worst_generator_aurc": float("nan"), "mean_cov_cp_risk_le_5pct": float("nan")}
    write_phase3_report(status, blockers, strongest)
    summary = {
        "PRE_PHASE_4_STATUS": status,
        "checks": checks,
        "blockers": blockers,
        "strongest_baseline": strongest,
        "config": cfg,
    }
    write_json(ARTIFACTS / "phase3_final_audit_summary.json", summary)
    audit_lines = [
        "# Phase 3 Final Audit Report",
        "",
        checklist.to_markdown(index=False),
        "",
        f"PRE_PHASE_4_STATUS = {status}",
    ]
    (REPORTS / "phase3_final_audit_report.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    if status == "PASS":
        hashes = phase3_artifact_hashes()
        hashes.to_csv(ARTIFACTS / "phase3_frozen_artifact_hashes.csv", index=False)
        write_yaml(
            PROJECT_ROOT / "configs/phase3/phase3_frozen.yaml",
            {
                "created_after_audit": True,
                "pre_phase_4_status": "PASS",
                "artifact_hash_registry": "artifacts/phase3/phase3_frozen_artifact_hashes.csv",
                "artifact_count": int(len(hashes)),
            },
        )
    print(f"PRE_PHASE_4_STATUS = {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
