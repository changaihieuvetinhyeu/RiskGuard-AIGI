#!/usr/bin/env python3
"""Evaluate Phase 5 OOF and threshold-cal calibration diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from selective_detection.error_probability_calibrator import PRIMARY_FEATURES, TRANSFORMED_FEATURE_NAMES, load_riskguard_json, transform_features
from selective_detection.calibration_metrics import calibrator_metrics, reliability_bins, score_distribution
from selective_detection.calibrator_artifact_io import DETECTORS, SPLITS, combo_slug, verify_frozen_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=[*DETECTORS, "all"], default="all")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase5" / "riskguard_calibrator.yaml"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / "phase5"))
    return parser.parse_args()


def selected(values: tuple[str, ...], requested: str) -> tuple[str, ...]:
    return values if requested == "all" else (requested,)


def score_path(output_root: Path, detector: str, split: str, artifact: str) -> Path:
    return output_root / "scores" / detector / split / f"{artifact}.parquet"


def metric_row(df: pd.DataFrame, detector: str, split: str, source: str, selected_c: float | None = None) -> dict[str, Any]:
    metrics = calibrator_metrics(
        df["base_error"].to_numpy(dtype=np.int64),
        df["risk_probability"].to_numpy(dtype=np.float64),
        sample_ids=df["sample_id"].astype(str).to_numpy(),
        n_bins=15,
    )
    row = {"detector": detector, "split": split, "score_source": source, **metrics}
    if selected_c is not None:
        row["selected_C"] = selected_c
    return row


def group_metric_rows(df: pd.DataFrame, detector: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_col in ("label", "base_prediction", "generator"):
        for group_value, part in df.groupby(group_col, dropna=False):
            metrics = calibrator_metrics(
                part["base_error"].to_numpy(dtype=np.int64),
                part["risk_probability"].to_numpy(dtype=np.float64),
                sample_ids=part["sample_id"].astype(str).to_numpy(),
                n_bins=15,
            )
            status = "undefined_single_error_class" if len(part["base_error"].unique()) < 2 else "ok"
            rows.append(
                {
                    "detector": detector,
                    "split": split,
                    "group_field": group_col,
                    "group_value": str(group_value),
                    "status": status,
                    **metrics,
                }
            )
    return rows


def bin_rows(df: pd.DataFrame, detector: str, split: str, partition: str) -> pd.DataFrame:
    bins = reliability_bins(df["base_error"].to_numpy(dtype=np.int64), df["risk_probability"].to_numpy(dtype=np.float64), n_bins=15)
    bins.insert(0, "partition", partition)
    bins.insert(0, "split", split)
    bins.insert(0, "detector", detector)
    return bins


def coefficient_rows(output_root: Path, detector: str, split: str, risk_fit_full: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = load_riskguard_json(output_root / "models" / f"{combo_slug(detector, split)}_riskguard.json")
    transformed = transform_features(risk_fit_full.loc[:, list(PRIMARY_FEATURES)], PRIMARY_FEATURES, as_frame=False)
    z = (transformed - np.asarray(model["scaler_means"], dtype=np.float64)) / np.asarray(model["scaler_scales"], dtype=np.float64)
    coef = np.asarray(model["coefficient_vector"], dtype=np.float64)
    coef_rows: list[dict[str, Any]] = []
    contrib_rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(PRIMARY_FEATURES):
        sign = "positive" if coef[idx] > 0.0 else "negative" if coef[idx] < 0.0 else "zero"
        coef_rows.append(
            {
                "detector": detector,
                "split": split,
                "feature": feature,
                "transformed_feature": TRANSFORMED_FEATURE_NAMES[feature],
                "coefficient": float(coef[idx]),
                "absolute_coefficient": float(abs(coef[idx])),
                "coefficient_sign": sign,
                "scaler_mean": float(model["scaler_means"][idx]),
                "scaler_scale": float(model["scaler_scales"][idx]),
                "selected_C": float(model["selected_C"]),
            }
        )
        contrib_rows.append(
            {
                "detector": detector,
                "split": split,
                "feature": feature,
                "coefficient": float(coef[idx]),
                "scaler_mean": float(model["scaler_means"][idx]),
                "scaler_scale": float(model["scaler_scales"][idx]),
                "mean_absolute_logit_contribution": float(np.mean(np.abs(z[:, idx] * coef[idx]))),
            }
        )
    return coef_rows, contrib_rows


def save_reliability_plot(data: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="ideal")
    for key, part in data.groupby([col for col in ("detector", "split") if col in data.columns]):
        label = "_".join(key) if isinstance(key, tuple) else str(key)
        valid = part[part["sample_count"] > 0]
        ax.plot(valid["mean_predicted_risk"], valid["observed_error_rate"], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed error rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_histogram_plot(frames: list[pd.DataFrame], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for frame in frames:
        label = f"{frame['split'].iloc[0]} errors"
        ax.hist(frame.loc[frame["base_error"].eq(1), "risk_probability"], bins=40, alpha=0.45, density=True, label=label)
        label = f"{frame['split'].iloc[0]} correct"
        ax.hist(frame.loc[frame["base_error"].eq(0), "risk_probability"], bins=40, alpha=0.35, density=True, label=label)
    ax.set_xlabel("Risk probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_bar_plot(data: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = data["detector"].astype(str) + "_" + data["split"].astype(str) + "_" + data["ablation"].astype(str)
    order = np.arange(len(data))
    ax.bar(order, data[metric].to_numpy(dtype=np.float64))
    ax.set_xticks(order)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    reports_dir = PROJECT_ROOT / "reports" / "phase5"
    figure_dir = reports_dir / "figures"
    figure_data_dir = output_root / "figure_data"
    figure_data_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen_inputs(PROJECT_ROOT, output_root, raise_on_fail=True)

    oof_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    oof_bin_frames: list[pd.DataFrame] = []
    threshold_bin_frames: list[pd.DataFrame] = []
    coef_rows: list[dict[str, Any]] = []
    contrib_rows: list[dict[str, Any]] = []
    dist_rows: list[dict[str, Any]] = []
    detector_oof_frames: dict[str, list[pd.DataFrame]] = {detector: [] for detector in DETECTORS}

    for detector in selected(DETECTORS, args.detector):
        for split in selected(SPLITS, args.split):
            slug = combo_slug(detector, split)
            model = load_riskguard_json(output_root / "models" / f"{slug}_riskguard.json")
            selected_c = float(model["selected_C"])
            oof = pd.read_parquet(score_path(output_root, detector, split, "risk_fit_oof"))
            threshold = pd.read_parquet(score_path(output_root, detector, split, "threshold_cal"))
            risk_fit_full = pd.read_parquet(score_path(output_root, detector, split, "risk_fit_fullfit"))
            oof_rows.append(metric_row(oof, detector, split, "risk_fit_oof", selected_c))
            threshold_rows.append(metric_row(threshold, detector, split, "threshold_cal", selected_c))
            group_rows.extend(group_metric_rows(oof, detector, split))
            oof_bins = bin_rows(oof, detector, split, "risk_fit_oof")
            threshold_bins = bin_rows(threshold, detector, split, "threshold_cal")
            oof_bin_frames.append(oof_bins)
            threshold_bin_frames.append(threshold_bins)
            c_rows, f_rows = coefficient_rows(output_root, detector, split, risk_fit_full)
            coef_rows.extend(c_rows)
            contrib_rows.extend(f_rows)
            detector_oof_frames[detector].append(oof)
            for artifact in ("threshold_cal", "protocol_seen", "protocol_held_out", "bfree_snapshot"):
                scored = pd.read_parquet(score_path(output_root, detector, split, artifact))
                summary = score_distribution(scored["risk_probability"].to_numpy(dtype=np.float64))
                dist_rows.append({"detector": detector, "split": split, "partition": artifact, "score": "risk_probability", **summary})

            oof_bins.to_csv(figure_data_dir / f"oof_reliability_{detector}_{split}.csv", index=False)
            save_reliability_plot(oof_bins, figure_dir / f"oof_reliability_{detector}_{split}.pdf", f"OOF reliability {detector} {split}")

    oof_metrics = pd.DataFrame(oof_rows)
    threshold_metrics = pd.DataFrame(threshold_rows)
    group_metrics = pd.DataFrame(group_rows)
    oof_bins_all = pd.concat(oof_bin_frames, ignore_index=True)
    threshold_bins_all = pd.concat(threshold_bin_frames, ignore_index=True)
    oof_metrics.to_csv(output_root / "oof_calibrator_metrics.csv", index=False)
    group_metrics.to_csv(output_root / "oof_calibrator_group_metrics.csv", index=False)
    oof_bins_all.to_csv(output_root / "oof_reliability_bins.csv", index=False)
    threshold_bins_all.to_csv(output_root / "threshold_cal_reliability_bins.csv", index=False)
    threshold_metrics.to_csv(output_root / "threshold_cal_calibrator_metrics.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(output_root / "calibrator_coefficients.csv", index=False)
    pd.DataFrame(contrib_rows).to_csv(output_root / "feature_contribution_summary.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(output_root / "frozen_score_distribution_summary.csv", index=False)
    oof_bins_all.to_csv(figure_data_dir / "oof_reliability_bins.csv", index=False)
    threshold_bins_all.to_csv(figure_data_dir / "threshold_cal_reliability_bins.csv", index=False)

    for detector in selected(DETECTORS, args.detector):
        detector_threshold = threshold_bins_all[threshold_bins_all["detector"].eq(detector)]
        detector_threshold.to_csv(figure_data_dir / f"threshold_cal_reliability_{detector}.csv", index=False)
        save_reliability_plot(detector_threshold, figure_dir / f"threshold_cal_reliability_{detector}.pdf", f"Threshold-cal reliability {detector}")
        if detector_oof_frames[detector]:
            hist_data = pd.concat(detector_oof_frames[detector], ignore_index=True)
            hist_data[["detector", "split", "base_error", "risk_probability"]].to_csv(
                figure_data_dir / f"oof_risk_histograms_{detector}.csv", index=False
            )
            save_histogram_plot(detector_oof_frames[detector], figure_dir / f"oof_risk_histograms_{detector}.pdf", f"OOF risk histograms {detector}")

    ablation_path = output_root / "ablation_oof_metrics.csv"
    if ablation_path.exists():
        ablations = pd.read_csv(ablation_path)
        ablations.to_csv(figure_data_dir / "ablation_oof_metrics.csv", index=False)
        save_bar_plot(ablations, "AURC", figure_dir / "ablation_aurc.pdf", "Ablation AURC")
        save_bar_plot(ablations, "NLL", figure_dir / "ablation_nll.pdf", "Ablation NLL")
    coefs = pd.DataFrame(coef_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = coefs["detector"] + "_" + coefs["split"] + "_" + coefs["feature"]
    ax.bar(np.arange(len(coefs)), coefs["coefficient"].to_numpy(dtype=np.float64))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(coefs)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("Coefficient")
    ax.set_title("RiskGuard coefficients")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "calibrator_coefficients.pdf")
    plt.close(fig)
    coefs.to_csv(figure_data_dir / "calibrator_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
