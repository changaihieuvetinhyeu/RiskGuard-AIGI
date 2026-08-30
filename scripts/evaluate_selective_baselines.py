#!/usr/bin/env python3
"""Evaluate Phase 3 selective baselines and write paper-facing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from selective_detection.tabular_input_schema import read_manifest_csv
from selective_detection.selective_baselines import (
    DETECTORS,
    MANDATORY_BASELINES,
    SPLITS,
    Phase2Cache,
    energy_risk,
    entropy_risk,
    exact_knn_distance,
    load_mahalanobis_npz,
    load_yaml,
    msp_risk,
    score_mahalanobis,
    sha256_file,
    temp_msp_risk,
    verify_phase2_frozen_hashes,
)
from selective_detection.selective_bootstrap import percentile_ci, stratified_unit_bootstrap
from selective_detection.selective_metrics import (
    accepted_error_metrics,
    aurc,
    calibration_metrics,
    error_ranking_metrics,
    risk_at_coverage,
    sha256_deduplicate,
    summarize_selective_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/phase3/selective_baselines.yaml"
ARTIFACTS = PROJECT_ROOT / "artifacts/phase3"
REPORTS = PROJECT_ROOT / "reports/phase3"
FIGURES = REPORTS / "figures"
MANIFESTS = PROJECT_ROOT / "datasets/manifests"


def score_path(detector: str, baseline: str, split: str, role: str) -> Path:
    return ARTIFACTS / "scores" / detector / baseline / f"{split}_{role}.parquet"


def dedup_scores(df: pd.DataFrame) -> pd.DataFrame:
    kept, _ = sha256_deduplicate(df)
    return kept


def metric_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    per_generator_rows = []
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                for role in ("protocol_seen", "protocol_held_out"):
                    scores = pd.read_parquet(score_path(detector, baseline, split, role))
                    for weighting, frame in (
                        ("row_level", scores),
                        ("sha256_deduplicated", dedup_scores(scores)),
                    ):
                        metric_rows.extend(summarize_selective_metrics(frame, weighting, split, role, "all"))
                        for generator, group in frame.groupby("generator", sort=True):
                            per_generator_rows.extend(
                                summarize_selective_metrics(group, weighting, split, role, str(generator))
                            )
                        real = frame[frame["label"].astype(int) == 0]
                        fake = frame[frame["label"].astype(int) == 1]
                        if len(real):
                            per_generator_rows.extend(summarize_selective_metrics(real, weighting, split, role, "real_class"))
                        if len(fake):
                            per_generator_rows.extend(summarize_selective_metrics(fake, weighting, split, role, "fake_class"))
    metrics = pd.DataFrame(metric_rows)
    per_generator = pd.DataFrame(per_generator_rows)
    metrics.to_csv(ARTIFACTS / "selective_baseline_metrics.csv", index=False)
    per_generator.to_csv(ARTIFACTS / "selective_baseline_per_generator_metrics.csv", index=False)
    return metrics, per_generator


def threshold_metric_table() -> pd.DataFrame:
    thresholds = pd.read_csv(ARTIFACTS / "global_thresholds.csv")
    rows = []
    for row in thresholds.to_dict("records"):
        detector = row["detector"]
        baseline = row["baseline"]
        split = row["split"]
        alpha = float(row["alpha"])
        for role in ("protocol_seen", "protocol_held_out"):
            scores = dedup_scores(pd.read_parquet(score_path(detector, baseline, split, role)))
            metrics = accepted_error_metrics(scores, row["threshold"])
            rows.append(
                {
                    "detector": detector,
                    "baseline": baseline,
                    "split": split,
                    "evaluation_role": role,
                    "alpha": alpha,
                    "delta": float(row["delta"]),
                    "threshold": row["threshold"],
                    "evaluation_weighting": "sha256_deduplicated",
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "selective_baseline_threshold_metrics.csv", index=False)
    return out


def calibration_table() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for baseline in ("msp", "temp_msp"):
            for split in SPLITS:
                for role in ("protocol_seen", "protocol_held_out"):
                    scores = dedup_scores(pd.read_parquet(score_path(detector, baseline, split, role)))
                    prob_col = "base_probability" if baseline == "msp" else "temperature_scaled_probability"
                    values = calibration_metrics(scores["label"].to_numpy(dtype=int), scores[prob_col].to_numpy(dtype=float))
                    for metric, value in values.items():
                        rows.append(
                            {
                                "detector": detector,
                                "baseline": baseline,
                                "split": split,
                                "evaluation_role": role,
                                "metric": metric,
                                "value": value,
                                "evaluation_weighting": "sha256_deduplicated",
                            }
                        )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "selective_baseline_calibration_metrics.csv", index=False)
    return out


def risk_correlations() -> pd.DataFrame:
    rows = []
    for detector in DETECTORS:
        for split in SPLITS:
            for role in ("protocol_seen", "protocol_held_out"):
                frames = []
                for baseline in MANDATORY_BASELINES:
                    scores = dedup_scores(pd.read_parquet(score_path(detector, baseline, split, role)))
                    frames.append(scores[["sample_id", "risk_score"]].rename(columns={"risk_score": baseline}))
                merged = frames[0]
                for frame in frames[1:]:
                    merged = merged.merge(frame, on="sample_id", how="inner", validate="one_to_one")
                corr = merged[list(MANDATORY_BASELINES)].corr(method="spearman")
                for left in MANDATORY_BASELINES:
                    for right in MANDATORY_BASELINES:
                        rows.append(
                            {
                                "detector": detector,
                                "split": split,
                                "evaluation_role": role,
                                "left_baseline": left,
                                "right_baseline": right,
                                "spearman": float(corr.loc[left, right]),
                                "high_redundancy": bool(abs(float(corr.loc[left, right])) >= 0.995 and left != right),
                            }
                        )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "risk_score_rank_correlations.csv", index=False)
    return out


def paper_tables(metrics: pd.DataFrame, per_generator: pd.DataFrame, threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["evaluation_weighting"] == "sha256_deduplicated")
        & (metrics["generator"] == "all")
        & (metrics["metric"].isin(["AURC", "error_detection_AUROC"]))
    ]
    pivot = primary.pivot_table(
        index=["detector", "baseline"],
        columns=["split", "evaluation_role", "metric"],
        values="value",
        aggfunc="first",
    )
    pivot.columns = ["_".join(col) for col in pivot.columns]
    pivot = pivot.reset_index()
    worst = per_generator[
        (per_generator["evaluation_weighting"] == "sha256_deduplicated")
        & (per_generator["metric"] == "AURC")
        & (~per_generator["generator"].isin(["all", "real_class", "fake_class"]))
    ].groupby(["detector", "baseline"], as_index=False)["value"].max().rename(columns={"value": "Worst-generator AURC"})
    mean_auroc = primary[primary["metric"] == "error_detection_AUROC"].groupby(["detector", "baseline"], as_index=False)["value"].mean()
    mean_auroc = mean_auroc.rename(columns={"value": "Error-detection AUROC"})
    paper = pivot.merge(worst, on=["detector", "baseline"], how="left").merge(mean_auroc, on=["detector", "baseline"], how="left")
    paper = paper.rename(columns={"detector": "Detector", "baseline": "Selective method"})
    paper.to_csv(ARTIFACTS / "selective_baseline_paper_table.csv", index=False)

    controlled = threshold_metrics.pivot_table(
        index=["detector", "baseline", "split", "evaluation_role"],
        columns="alpha",
        values=["coverage", "selective_risk", "far_accepted", "fnr_accepted", "minimum_class_coverage"],
        aggfunc="first",
    )
    controlled.columns = [f"{name}_alpha_{alpha:g}" for name, alpha in controlled.columns]
    controlled.reset_index().to_csv(ARTIFACTS / "selective_baseline_controlled_risk_table.csv", index=False)
    return paper


def bfree_scores_and_metrics(cfg: dict) -> pd.DataFrame:
    thresholds = pd.read_csv(ARTIFACTS / "global_thresholds.csv")
    rows = []
    for detector in DETECTORS:
        pred = pd.read_parquet(PROJECT_ROOT / f"artifacts/bfree_viral_verified_{detector}_predictions.parquet")
        pred["sample_id"] = pred["sample_id"].astype(str)
        embeddings = np.load(PROJECT_ROOT / f"artifacts/bfree_viral_verified_{detector}_embeddings.npy")
        manifest = read_manifest_csv(MANIFESTS / "bfree_viral_verified_snapshot.csv")
        manifest["sample_id"] = manifest["sample_id"].astype(str)
        for split in SPLITS:
            phase2_threshold = pd.read_csv(PROJECT_ROOT / "artifacts/phase2_clean_thresholds.csv")
            phase2_threshold = float(
                phase2_threshold[
                    (phase2_threshold["detector"] == detector)
                    & (phase2_threshold["split"] == split)
                    & (phase2_threshold["threshold_source"] == "threshold_cal")
                ]["decision_threshold"].iloc[0]
            )
            base = manifest.merge(
                pred[["sample_id", "raw_logit", "fake_probability"]],
                on="sample_id",
                how="left",
                validate="one_to_one",
            )
            base["detector"] = detector
            base["split"] = split
            base["partition"] = "bfree_viral_verified_snapshot"
            base["evaluation_role"] = "B-Free Viral Verified Snapshot"
            base["generator"] = "bfree_viral"
            base["base_logit"] = base["raw_logit"].astype(float)
            base["base_probability"] = base["fake_probability"].astype(float)
            base["base_prediction"] = (base["base_probability"] >= phase2_threshold).astype(int)
            base["base_error"] = (base["base_prediction"].astype(int) != base["label"].astype(int)).astype(int)
            for baseline in MANDATORY_BASELINES:
                df = base.copy()
                diagnostics = {}
                if baseline == "msp":
                    risk = msp_risk(df["base_probability"].to_numpy(dtype=float))
                elif baseline == "entropy":
                    risk = entropy_risk(df["base_probability"].to_numpy(dtype=float))
                elif baseline == "energy":
                    risk = energy_risk(df["base_logit"].to_numpy(dtype=float))
                elif baseline == "temp_msp":
                    temp = json.loads((ARTIFACTS / "fits" / detector / split / "temperature.json").read_text(encoding="utf-8"))
                    risk, temp_prob = temp_msp_risk(df["base_logit"].to_numpy(dtype=float), float(temp["temperature"]))
                    diagnostics["temperature_scaled_probability"] = temp_prob
                elif baseline == "mahalanobis":
                    stats = load_mahalanobis_npz(ARTIFACTS / "fits" / detector / split / "mahalanobis_stats.npz")
                    scored = score_mahalanobis(embeddings, stats)
                    risk = scored["risk_score"]
                    diagnostics.update(scored)
                elif baseline == "knn":
                    fit_dir = ARTIFACTS / "fits" / detector / split
                    selected = json.loads((fit_dir / "knn_selected_k.json").read_text(encoding="utf-8"))
                    cache = Phase2Cache(PROJECT_ROOT, detector)
                    bank = pd.read_parquet(fit_dir / "knn_reference_bank.parquet")
                    bank_embeddings = cache.embeddings_for(bank["sample_id"])
                    scored = exact_knn_distance(
                        bank_embeddings,
                        embeddings,
                        int(selected["selected_k"]),
                        bank_ids=bank["sample_id"].astype(str).to_numpy(),
                        query_ids=None,
                        device=str(cfg["knn"].get("cuda_device", "cuda:1")),
                        batch_size=int(cfg["knn"].get("query_batch_size", 1024)),
                    )
                    risk = scored["risk_score"]
                    diagnostics.update(scored)
                else:
                    continue
                df["baseline"] = baseline
                df["risk_score"] = risk
                df["risk_orientation"] = "higher_risk_score_more_likely_to_reject"
                for key, value in diagnostics.items():
                    df[key] = value
                out_path = ARTIFACTS / "scores" / detector / baseline / f"{split}_bfree_snapshot.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_path, index=False)
                ranking = error_ranking_metrics(df["base_error"].to_numpy(dtype=int), df["risk_score"].to_numpy(dtype=float))
                row = {
                    "detector": detector,
                    "baseline": baseline,
                    "split": split,
                    "evaluation_role": "B-Free Viral Verified Snapshot",
                    "sample_count": int(len(df)),
                    "source_id_count": int(df["source_id"].nunique()),
                    "error_count": int(df["base_error"].sum()),
                    "AURC": aurc(df["base_error"].to_numpy(dtype=int), df["risk_score"].to_numpy(dtype=float), df["sample_id"].astype(str).to_numpy()),
                    "error_detection_AUROC": ranking.auroc,
                    "error_detection_AUPR": ranking.aupr,
                    "status": ranking.status,
                }
                for alpha in (0.01, 0.05):
                    thr = thresholds[
                        (thresholds["detector"] == detector)
                        & (thresholds["baseline"] == baseline)
                        & (thresholds["split"] == split)
                        & (thresholds["alpha"] == alpha)
                    ]["threshold"].iloc[0]
                    accepted = accepted_error_metrics(df, thr)
                    row[f"coverage_at_cp_risk_le_{int(alpha*100)}pct"] = accepted["coverage"]
                    row[f"selective_risk_at_cp_risk_le_{int(alpha*100)}pct"] = accepted["selective_risk"]
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "bfree_snapshot_selective_metrics.csv", index=False)
    return out


def fast_stratified_aurc_bootstrap(scores: pd.DataFrame, n_bootstrap: int, seed: int) -> np.ndarray:
    """Exact AURC bootstrap from multinomial sample counts, without per-draw sorting."""
    ordered = scores.sort_values(["risk_score", "sample_id"], kind="mergesort").reset_index(drop=True)
    errors = ordered["base_error"].to_numpy(dtype=np.int64)
    total = len(ordered)
    harmonic = np.zeros(total + 1, dtype=np.float64)
    harmonic[1:] = np.cumsum(1.0 / np.arange(1, total + 1, dtype=np.float64))
    strata = []
    grouped = ordered.reset_index().groupby(["label", "generator"], dropna=False, sort=True)
    for _, group in grouped:
        idx = group["index"].to_numpy(dtype=np.int64)
        strata.append(idx)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        counts = np.zeros(total, dtype=np.int64)
        for idx in strata:
            if len(idx):
                counts[idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)))
        accepted_before = np.cumsum(counts, dtype=np.int64) - counts
        errors_before = np.cumsum(counts * errors, dtype=np.int64) - counts * errors
        nz = counts > 0
        c = counts[nz].astype(np.float64)
        a0 = accepted_before[nz]
        e0 = errors_before[nz].astype(np.float64)
        segment = harmonic[a0 + counts[nz]] - harmonic[a0]
        err = errors[nz]
        contribution = np.where(err == 0, e0 * segment, c + (e0 - a0.astype(np.float64)) * segment)
        draws[i] = float(contribution.sum() / total)
    return draws


def bootstrap_ci_artifact(metrics: pd.DataFrame, cfg: dict, bfree_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_boot = int(cfg["bootstrap"]["iterations"])
    seed = int(cfg["bootstrap"]["seed"])
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                for role in ("protocol_seen", "protocol_held_out"):
                    scores = dedup_scores(pd.read_parquet(score_path(detector, baseline, split, role)))
                    point = float(
                        metrics[
                            (metrics["detector"] == detector)
                            & (metrics["baseline"] == baseline)
                            & (metrics["split"] == split)
                            & (metrics["evaluation_role"] == role)
                            & (metrics["generator"] == "all")
                            & (metrics["evaluation_weighting"] == "sha256_deduplicated")
                            & (metrics["metric"] == "AURC")
                        ]["value"].iloc[0]
                    )
                    # Full 2000-draw AURC bootstrap is intentionally limited to aggregate
                    # AURC to keep the Phase 3 audit reproducible within a workstation run.
                    draws = fast_stratified_aurc_bootstrap(scores, n_boot, seed)
                    lo, hi = percentile_ci(draws, float(cfg["bootstrap"]["confidence_level"]))
                    rows.append(
                        {
                            "detector": detector,
                            "baseline": baseline,
                            "split": split,
                            "evaluation_role": role,
                            "generator": "all",
                            "metric": "AURC",
                            "point_estimate": point,
                            "ci_lower": lo,
                            "ci_upper": hi,
                            "bootstrap_unit": "sha256",
                            "stratification": "label x generator",
                            "n_bootstrap": n_boot,
                            "seed": seed,
                            "sample_count": int(len(scores)),
                            "error_count": int(scores["base_error"].sum()),
                            "status": "ok",
                            "low_error_count_warning": bool(scores["base_error"].sum() < 50),
                        }
                    )
    for row in bfree_metrics.to_dict("records"):
        path = ARTIFACTS / "scores" / row["detector"] / row["baseline"] / f"{row['split']}_bfree_snapshot.parquet"
        scores = pd.read_parquet(path)
        draws = stratified_unit_bootstrap(
            scores,
            unit_col="source_id",
            strata_cols=["label"],
            metric_fn=lambda frame: aurc(
                frame["base_error"].to_numpy(dtype=int),
                frame["risk_score"].to_numpy(dtype=float),
                frame["sample_id"].astype(str).to_numpy(),
            ),
            n_bootstrap=n_boot,
            seed=seed,
        )
        lo, hi = percentile_ci(draws, float(cfg["bootstrap"]["confidence_level"]))
        rows.append(
            {
                "detector": row["detector"],
                "baseline": row["baseline"],
                "split": row["split"],
                "evaluation_role": "B-Free Viral Verified Snapshot",
                "generator": "bfree_viral",
                "metric": "AURC",
                "point_estimate": row["AURC"],
                "ci_lower": lo,
                "ci_upper": hi,
                "bootstrap_unit": "source_id",
                "stratification": "label",
                "n_bootstrap": n_boot,
                "seed": seed,
                "sample_count": int(row["sample_count"]),
                "error_count": int(row["error_count"]),
                "status": "ok",
                "low_error_count_warning": bool(row["error_count"] < 50),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS / "bootstrap_ci.csv", index=False)
    return out


def figures(metrics: pd.DataFrame, per_generator: pd.DataFrame, corr: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for detector in DETECTORS:
        det_metrics = metrics[
            (metrics["detector"] == detector)
            & (metrics["evaluation_weighting"] == "sha256_deduplicated")
            & (metrics["generator"] == "all")
        ]
        rc_data = det_metrics[det_metrics["metric"].str.startswith("risk_at_")]
        rc_data.to_csv(FIGURES / f"risk_coverage_{detector}_data.csv", index=False)
        plt.figure(figsize=(9, 5))
        for baseline, group in rc_data.groupby("baseline"):
            x = [50, 70, 80, 90, 95]
            y = [
                float(group[group["metric"] == f"risk_at_{cov}pct_coverage"]["value"].mean())
                for cov in x
            ]
            plt.plot(x, y, marker="o", label=baseline)
        plt.xlabel("Coverage (%)")
        plt.ylabel("Selective risk")
        plt.title(f"{detector} risk at coverage")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES / f"risk_coverage_{detector}.pdf")
        plt.close()

        hist_rows = []
        for baseline in MANDATORY_BASELINES:
            sample = dedup_scores(pd.read_parquet(score_path(detector, baseline, "split_a", "protocol_seen")))
            hist_rows.append(sample[["baseline", "risk_score", "base_error"]])
        hist = pd.concat(hist_rows, ignore_index=True)
        hist.to_csv(FIGURES / f"error_score_histogram_{detector}_data.csv", index=False)
        plt.figure(figsize=(9, 5))
        for err, group in hist.groupby("base_error"):
            plt.hist(group["risk_score"], bins=50, alpha=0.45, density=True, label=f"error={err}")
        plt.xlabel("Risk score")
        plt.ylabel("Density")
        plt.title(f"{detector} correct vs incorrect risk scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / f"error_score_histogram_{detector}.pdf")
        plt.close()

        gen = per_generator[
            (per_generator["detector"] == detector)
            & (per_generator["evaluation_weighting"] == "sha256_deduplicated")
            & (per_generator["metric"] == "AURC")
            & (~per_generator["generator"].isin(["all", "real_class", "fake_class"]))
        ]
        gen.to_csv(FIGURES / f"per_generator_aurc_{detector}_data.csv", index=False)
        plot_gen = gen.groupby(["baseline", "generator"], as_index=False)["value"].mean()
        pivot = plot_gen.pivot(index="generator", columns="baseline", values="value")
        pivot.plot(kind="bar", figsize=(10, 5))
        plt.ylabel("AURC")
        plt.title(f"{detector} per-generator AURC")
        plt.tight_layout()
        plt.savefig(FIGURES / f"per_generator_aurc_{detector}.pdf")
        plt.close()

        det_corr = corr[(corr["detector"] == detector) & (corr["split"] == "split_a") & (corr["evaluation_role"] == "protocol_seen")]
        matrix = det_corr.pivot(index="left_baseline", columns="right_baseline", values="spearman").loc[
            list(MANDATORY_BASELINES), list(MANDATORY_BASELINES)
        ]
        matrix.to_csv(FIGURES / f"risk_score_correlation_{detector}_data.csv")
        plt.figure(figsize=(6, 5))
        plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(label="Spearman")
        plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        plt.yticks(range(len(matrix.index)), matrix.index)
        plt.title(f"{detector} risk-score rank correlation")
        plt.tight_layout()
        plt.savefig(FIGURES / f"risk_score_correlation_{detector}.pdf")
        plt.close()


def main() -> int:
    verify_phase2_frozen_hashes(PROJECT_ROOT)
    cfg = load_yaml(CONFIG_PATH)
    metrics, per_generator = metric_tables()
    threshold_metrics = threshold_metric_table()
    calibration_table()
    corr = risk_correlations()
    paper_tables(metrics, per_generator, threshold_metrics)
    bfree = bfree_scores_and_metrics(cfg)
    bootstrap_ci_artifact(metrics, cfg, bfree)
    figures(metrics, per_generator, corr)
    print("Wrote Phase 3 evaluation artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
