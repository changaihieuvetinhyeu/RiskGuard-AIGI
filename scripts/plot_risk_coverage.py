#!/usr/bin/env python3
"""Create Figure 2 from the current official SAFE held-out artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHASE3 = ROOT / "artifacts" / "phase3"
PHASE5 = ROOT / "artifacts" / "phase5"
PHASE6 = ROOT / "artifacts" / "phase6"
FIG_DIR = ROOT / "reports" / "phase8" / "figures"

THRESHOLD_REGISTRY = PHASE6 / "certified_threshold_registry.csv"
FINAL_METRICS = PHASE6 / "final_selective_metrics.csv"

PDF_OUT = FIG_DIR / "figure2_safe_heldout_risk_coverage.pdf"
PNG_OUT = FIG_DIR / "figure2_safe_heldout_risk_coverage.png"
PLOT_DATA_OUT = FIG_DIR / "figure2_safe_heldout_plot_data.csv"
AUDIT_OUT = FIG_DIR / "figure2_safe_heldout_audit.md"

DETECTOR = "safe"
DATASET = "protocol_held_out"
METHODS = ("riskguard", "msp", "knn")
SPLITS = ("split_a", "split_b")
POLICY = "source_group_cp"
OFFICIAL_METHOD = "riskguard_logit_trajectory"
ALPHA = 0.05
DELTA = 0.05

METHOD_LABELS = {
    "riskguard": "RiskGuard",
    "msp": "MSP",
    "knn": "Cosine kNN",
}
PANEL_LABELS = {
    "split_a": "(a) Split A: held-out generators",
    "split_b": "(b) Split B: held-out generators",
}
COLORS = {
    "riskguard": "#0072B2",
    "msp": "#D55E00",
    "knn": "#009E73",
}
LINESTYLES = {
    "riskguard": "-",
    "msp": "--",
    "knn": "-.",
}


def _require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing))


def _as_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {field}, got {value!r}") from exc


def load_curve_data() -> pd.DataFrame:
    """Recompute exact curves from the official scorer and matched baselines."""

    frames: list[pd.DataFrame] = []
    for split in SPLITS:
        sources = {
            "riskguard": (
                PHASE5 / "scores" / DETECTOR / split / f"{DATASET}.parquet",
                "risk_probability",
            ),
            "msp": (
                PHASE3 / "scores" / DETECTOR / "msp" / f"{split}_{DATASET}.parquet",
                "risk_score",
            ),
            "knn": (
                PHASE3 / "scores" / DETECTOR / "knn" / f"{split}_{DATASET}.parquet",
                "risk_score",
            ),
        }
        _require([path for path, _ in sources.values()])

        reference_ids: set[str] | None = None
        for method in METHODS:
            path, score_col = sources[method]
            scores = pd.read_parquet(path, columns=["sample_id", "base_error", score_col])
            if scores.empty:
                raise RuntimeError(f"No scores found for {method} {split} at {path}.")
            if scores[["sample_id", "base_error", score_col]].isna().any().any():
                raise RuntimeError(f"Missing Figure 2 score values for {method} {split}.")

            sample_ids = set(scores["sample_id"].astype(str))
            if reference_ids is None:
                reference_ids = sample_ids
            elif sample_ids != reference_ids:
                raise RuntimeError(f"Sample IDs differ across Figure 2 methods for {split}.")

            ordered = scores.sort_values([score_col, "sample_id"], kind="mergesort").reset_index(drop=True)
            accepted_count = np.arange(1, len(ordered) + 1, dtype=np.int64)
            curve = pd.DataFrame(
                {
                    "detector": DETECTOR,
                    "split": split,
                    "method": method,
                    "dataset": DATASET,
                    "coverage": accepted_count / len(ordered),
                    "selective_risk": np.cumsum(ordered["base_error"].to_numpy(dtype=np.int64)) / accepted_count,
                    "threshold": ordered[score_col].to_numpy(dtype=float),
                    "accepted_count": accepted_count,
                    "method_label": METHOD_LABELS[method],
                }
            )
            frames.append(curve)

    curves = pd.concat(frames, ignore_index=True)
    return curves.sort_values(["split", "method", "coverage", "threshold"], kind="mergesort").reset_index(drop=True)


def load_operating_points(curves: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    registry = pd.read_csv(THRESHOLD_REGISTRY)
    metrics = pd.read_csv(FINAL_METRICS)

    reg = registry[
        (registry["detector"].astype(str).str.lower() == DETECTOR)
        & (registry["method"].astype(str) == OFFICIAL_METHOD)
        & (registry["policy"].astype(str) == POLICY)
        & (registry["alpha"].astype(float) == ALPHA)
        & (registry["split"].isin(SPLITS))
    ].copy()
    met = metrics[
        (metrics["detector"].astype(str).str.lower() == DETECTOR)
        & (metrics["method"].astype(str) == OFFICIAL_METHOD)
        & (metrics["policy"].astype(str) == POLICY)
        & (metrics["partition"].astype(str) == DATASET)
        & (metrics["split"].isin(SPLITS))
    ].copy()

    if len(reg) != len(SPLITS):
        raise RuntimeError(f"Expected {len(SPLITS)} threshold registry rows, found {len(reg)}.")
    if len(met) != len(SPLITS):
        raise RuntimeError(f"Expected {len(SPLITS)} final metric rows, found {len(met)}.")

    points: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        r = reg[reg["split"].eq(split)].iloc[0]
        m = met[met["split"].eq(split)].iloc[0]
        if str(r["certification_status"]) != "CERTIFIED":
            raise RuntimeError(f"RiskGuard operating threshold for {split} is not CERTIFIED in registry.")
        tau_registry = _as_float(r["selected_threshold"], field=f"{split} registry selected_threshold")

        coverage = _as_float(m["coverage"], field=f"{split} coverage")
        risk = _as_float(m["selective_risk"], field=f"{split} selective_risk")
        accepted_samples = int(m["accepted_samples"])
        total_samples = int(m["total_samples"])
        accepted_errors = int(m["accepted_errors"])

        rg_curve = curves[(curves["split"].eq(split)) & (curves["method"].eq("riskguard"))].copy()
        eligible = rg_curve[rg_curve["threshold"] <= tau_registry]
        if eligible.empty:
            raise RuntimeError(f"No RiskGuard curve threshold <= tau for {split}.")
        curve_row = eligible.sort_values("threshold", kind="mergesort").iloc[-1]
        curve_matches_metrics = bool(
            np.isclose(float(curve_row["coverage"]), coverage, rtol=0.0, atol=1e-12)
            and np.isclose(float(curve_row["selective_risk"]), risk, rtol=0.0, atol=1e-12)
            and int(curve_row["accepted_count"]) == accepted_samples
        )
        if not curve_matches_metrics:
            raise RuntimeError(f"Operating point does not match curve/final metric records for {split}.")

        points.append(
            {
                "panel": "a" if split == "split_a" else "b",
                "detector": DETECTOR,
                "split": split,
                "dataset": DATASET,
                "method": "riskguard",
                "method_label": METHOD_LABELS["riskguard"],
                "policy": POLICY,
                "alpha": ALPHA,
                "delta": DELTA,
                "threshold": tau_registry,
                "curve_threshold_at_marker": float(curve_row["threshold"]),
                "coverage": coverage,
                "selective_risk": risk,
                "accepted_count": accepted_samples,
                "accepted_errors": accepted_errors,
                "total_samples": total_samples,
                "policy_status": str(r["certification_status"]),
                "point_role": "riskguard_tau_star_marker",
            }
        )
        audit_rows.append(
            {
                "split": split,
                "policy_status": str(r["certification_status"]),
                "threshold": tau_registry,
                "curve_threshold_at_marker": float(curve_row["threshold"]),
                "coverage": coverage,
                "selective_risk": risk,
                "accepted_samples": accepted_samples,
                "accepted_errors": accepted_errors,
                "total_samples": total_samples,
                "certification_coverage": _as_float(r["certification_coverage"], field="certification_coverage"),
                "max_group_cp_upper": _as_float(r["max_group_cp_upper"], field="max_group_cp_upper"),
                "delta_cell": _as_float(r["delta_cell"], field="delta_cell"),
                "curve_matches_final_metric": curve_matches_metrics,
            }
        )

    return pd.DataFrame(points), audit_rows


def write_plot_data(curves: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    curve_out = curves.copy()
    curve_out["panel"] = np.where(curve_out["split"].eq("split_a"), "a", "b")
    curve_out["policy"] = ""
    curve_out["alpha"] = ""
    curve_out["delta"] = ""
    curve_out["curve_threshold_at_marker"] = ""
    curve_out["accepted_errors"] = ""
    curve_out["total_samples"] = ""
    curve_out["policy_status"] = ""
    curve_out["point_role"] = "curve"
    curve_out = curve_out[
        [
            "panel",
            "detector",
            "split",
            "dataset",
            "method",
            "method_label",
            "policy",
            "alpha",
            "delta",
            "threshold",
            "curve_threshold_at_marker",
            "coverage",
            "selective_risk",
            "accepted_count",
            "accepted_errors",
            "total_samples",
            "policy_status",
            "point_role",
        ]
    ]
    points_out = points[curve_out.columns].copy()
    all_rows = pd.concat([curve_out, points_out], ignore_index=True)
    all_rows.to_csv(PLOT_DATA_OUT, index=False)
    return all_rows


def make_figure(curves: pd.DataFrame, points: pd.DataFrame) -> dict[str, Any]:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )
    max_risk = max(float(curves["selective_risk"].max()), ALPHA)
    y_max = min(1.0, max_risk * 1.12)
    if y_max < ALPHA + 0.01:
        y_max = ALPHA + 0.01
    y_limits = (0.0, float(y_max))
    x_limits = (0.0, 1.0)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(4.803, 2.44),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    handles = {}
    for ax, split in zip(axes, SPLITS, strict=True):
        split_curves = curves[curves["split"].eq(split)]
        for method in METHODS:
            sub = split_curves[split_curves["method"].eq(method)].sort_values("coverage", kind="mergesort")
            line = ax.plot(
                sub["coverage"].to_numpy(),
                sub["selective_risk"].to_numpy(),
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.0 if method == "riskguard" else 1.5,
                label=METHOD_LABELS[method],
            )[0]
            handles.setdefault(method, line)
        op = points[points["split"].eq(split)].iloc[0]
        ax.scatter(
            [float(op["coverage"])],
            [float(op["selective_risk"])],
            color=COLORS["riskguard"],
            marker="o",
            s=36,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            clip_on=False,
            label=r"Frozen $\tau^\ast$",
        )
        ax.axhline(ALPHA, color="#7A7A7A", linestyle=(0, (3, 2)), linewidth=0.7, alpha=0.9)
        ax.text(
            0.02,
            ALPHA,
            f"{ALPHA:.2f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=8,
            color="#666666",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
            zorder=6,
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(PANEL_LABELS[split], loc="left", pad=3)
        ax.grid(True, color="#DDDDDD", linewidth=0.4, alpha=0.8)
        ax.set_xlabel("Coverage")
    axes[0].set_ylabel("Selective risk")

    inset = axes[0].inset_axes([0.48, 0.39, 0.48, 0.48])
    inset_curves = curves[curves["split"].eq("split_a")]
    for method in METHODS:
        sub = inset_curves[inset_curves["method"].eq(method)].sort_values("coverage", kind="mergesort")
        inset.plot(
            sub["coverage"].to_numpy(),
            sub["selective_risk"].to_numpy(),
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.35 if method == "riskguard" else 1.0,
        )
    op_a = points[points["split"].eq("split_a")].iloc[0]
    inset.scatter(
        [float(op_a["coverage"])],
        [float(op_a["selective_risk"])],
        color=COLORS["riskguard"],
        marker="o",
        s=24,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 0.015)
    inset.tick_params(labelsize=7, length=2, pad=1)
    inset.tick_params(axis="y", pad=2.5)
    for label in inset.get_yticklabels():
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.4})
        label.set_zorder(10)
    inset.get_xticklabels()[0].set_horizontalalignment("left")
    inset.grid(True, color="#E6E6E6", linewidth=0.35, alpha=0.85)
    inset.set_title("Split A detail", fontsize=8, pad=1)

    tau_handle = axes[0].scatter([], [], color=COLORS["riskguard"], marker="o", s=36, edgecolor="white", linewidth=0.7, label=r"Frozen $\tau^\ast$")
    alpha_handle = axes[0].plot([], [], color="#7A7A7A", linestyle=(0, (3, 2)), linewidth=0.7, alpha=0.9, label=r"Target $\alpha = 0.05$")[0]
    legend_handles = [handles[m] for m in METHODS] + [tau_handle, alpha_handle]
    legend_labels = [METHOD_LABELS[m] for m in METHODS] + [r"Frozen $\tau^\ast$", r"Target $\alpha = 0.05$"]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        handlelength=1.7,
        handletextpad=0.4,
        columnspacing=0.7,
    )
    fig.subplots_adjust(top=0.76, left=0.11, right=0.98, bottom=0.19, wspace=0.22)
    fig.savefig(PDF_OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PNG_OUT, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {"x_limits": x_limits, "y_limits": y_limits, "linear_scale": True}


def curve_audit(curves: pd.DataFrame, axes: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    x0, x1 = axes["x_limits"]
    y0, y1 = axes["y_limits"]
    for (split, method), sub in curves.groupby(["split", "method"], sort=True):
        missing = sub[["coverage", "selective_risk", "threshold", "accepted_count"]].isna().any().to_dict()
        duplicate_coverages = int(sub["coverage"].duplicated().sum())
        clipped = bool(
            (sub["coverage"].lt(x0).any())
            or (sub["coverage"].gt(x1).any())
            or (sub["selective_risk"].lt(y0).any())
            or (sub["selective_risk"].gt(y1).any())
        )
        rows.append(
            {
                "split": split,
                "method": method,
                "plotted_points": int(len(sub)),
                "coverage_min": float(sub["coverage"].min()),
                "coverage_max": float(sub["coverage"].max()),
                "selective_risk_min": float(sub["selective_risk"].min()),
                "selective_risk_max": float(sub["selective_risk"].max()),
                "duplicate_coverage_count": duplicate_coverages,
                "missing_values": bool(any(missing.values())),
                "missing_value_columns": ",".join([col for col, value in missing.items() if value]),
                "axis_clipping_or_truncation": clipped,
            }
        )
    return rows


def low_coverage_audit(curves: pd.DataFrame, coverage_cutoff: float = 0.01) -> list[dict[str, Any]]:
    rows = []
    for (split, method), sub in curves.groupby(["split", "method"], sort=True):
        low = sub[sub["coverage"] <= coverage_cutoff].copy()
        rows.append(
            {
                "split": split,
                "method": method,
                "coverage_cutoff": coverage_cutoff,
                "low_coverage_points": int(len(low)),
                "accepted_count_min": int(low["accepted_count"].min()) if len(low) else 0,
                "accepted_count_max": int(low["accepted_count"].max()) if len(low) else 0,
                "selective_risk_min": float(low["selective_risk"].min()) if len(low) else float("nan"),
                "selective_risk_max": float(low["selective_risk"].max()) if len(low) else float("nan"),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.12g}")
            else:
                values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def write_audit(
    curves: pd.DataFrame,
    op_audit: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    low_coverage_rows: list[dict[str, Any]],
    axes: dict[str, Any],
) -> None:
    source_values_verified = all(row["curve_matches_final_metric"] for row in op_audit)
    no_axis_truncation = not any(row["axis_clipping_or_truncation"] for row in curve_rows)
    no_missing = not any(row["missing_values"] for row in curve_rows)
    safe_for_main = bool(source_values_verified and no_axis_truncation and no_missing and PDF_OUT.exists() and PNG_OUT.exists())

    filters = {
        "official_riskguard_scores": {
            "paths": [f"artifacts/phase5/scores/{DETECTOR}/{split}/{DATASET}.parquet" for split in SPLITS],
            "method": OFFICIAL_METHOD,
            "risk_column": "risk_probability",
        },
        "baseline_scores": {
            "paths": [
                f"artifacts/phase3/scores/{DETECTOR}/{method}/{split}_{DATASET}.parquet"
                for method in ("msp", "knn")
                for split in SPLITS
            ],
            "risk_column": "risk_score",
        },
        "operating_point": {
            "threshold_registry": str(THRESHOLD_REGISTRY.relative_to(ROOT)),
            "final_metrics": str(FINAL_METRICS.relative_to(ROOT)),
            "detector": DETECTOR,
            "method": OFFICIAL_METHOD,
            "policy": POLICY,
            "alpha": ALPHA,
            "partition": DATASET,
            "split": list(SPLITS),
        },
    }

    caption = (
        "Risk--coverage performance on held-out generators. Curves compare\n"
        "the official logit-trajectory RiskGuard scorer with MSP and cosine kNN\n"
        "for SAFE under the two complementary\n"
        "protocol directions. Markers show the operating threshold\n"
        "$\\tau^\\star$ selected and certified on the independent certification\n"
        "subset, then frozen and evaluated empirically on the corresponding\n"
        "held-out partition. The horizontal dashed line denotes the target risk\n"
        "$\\alpha=0.05$. The finite-sample certificate does not extend to\n"
        "held-out generators."
    )

    audit = f"""# Figure 2 SAFE Held-Out Risk-Coverage Audit

Generated by `scripts/plot_risk_coverage.py`.

## Outputs

- PDF: `reports/phase8/figures/figure2_safe_heldout_risk_coverage.pdf`
- PNG: `reports/phase8/figures/figure2_safe_heldout_risk_coverage.png`
- Plot data: `reports/phase8/figures/figure2_safe_heldout_plot_data.csv`

## Caption

\"{caption}\"

## Source Artifacts And Filters

Official RiskGuard curve sources: `artifacts/phase5/scores/safe/{{split}}/protocol_held_out.parquet`

Baseline curve sources: `artifacts/phase3/scores/safe/{{msp,knn}}/{{split}}_protocol_held_out.parquet`

Threshold registry: `{THRESHOLD_REGISTRY.relative_to(ROOT)}`

Final evaluation metrics: `{FINAL_METRICS.relative_to(ROOT)}`

Filters:

```json
{json.dumps(filters, indent=2, sort_keys=True)}
```

All curves were recomputed exactly by sorting each matched score artifact from lowest
to highest estimated risk, with `sample_id` as the deterministic tie-breaker. The
three methods use the same 110,003 sample IDs within each split.

## Figure Settings

- Panels: `(a) Split A: held-out generators`; `(b) Split B: held-out generators`
- Methods: {", ".join(METHOD_LABELS[method] for method in METHODS)}
- Excluded from main figure: Mahalanobis
- Selective-risk scale: linear
- Shared x-limits: `{axes["x_limits"]}`
- Shared y-limits: `{axes["y_limits"]}`
- Horizontal target line: alpha = `{ALPHA}`, labeled `0.05` in both panels
- Panel A inset: x-range `(0.0, 1.0)`, y-range `(0.0, 0.015)`, no alpha line
- Smoothing/interpolation: none
- Typography: 9 pt axis labels/panel titles; 8 pt legend/main ticks/inset title; 7 pt inset ticks
- Final physical canvas: 4.803 x 2.44 inches (12.2 x 6.2 cm)
- Export: vector PDF and 400 dpi PNG with tight bounding box and 0.02-inch padding
- Intended LaTeX placement: `width=\\linewidth` where `\\linewidth=12.2 cm` (scale factor approximately 1)

## Operating Points

The marker is the RiskGuard source-group operating point selected/certified on the independent certification subset, then evaluated empirically on held-out data. It is not labeled as a held-out certificate.

{markdown_table(op_audit, ["split", "policy_status", "threshold", "curve_threshold_at_marker", "coverage", "selective_risk", "accepted_samples", "accepted_errors", "total_samples", "certification_coverage", "max_group_cp_upper", "delta_cell", "curve_matches_final_metric"])}

Expected reference check:

- SAFE Split A held-out coverage ~= 0.861886 and selective risk ~= 0.002162.
- SAFE Split B held-out coverage ~= 0.999836 and selective risk ~= 0.055326.

## Plotted Curve Audit

{markdown_table(curve_rows, ["split", "method", "plotted_points", "coverage_min", "coverage_max", "selective_risk_min", "selective_risk_max", "duplicate_coverage_count", "missing_values", "missing_value_columns", "axis_clipping_or_truncation"])}

Duplicate coverages are counted within each plotted curve. Missing-value checks cover `coverage`, `selective_risk`, `threshold`, and `accepted_count`. Axis clipping/truncation is true only if stored plotted values fall outside the displayed shared axes.

## Low-Coverage Segment Audit

The low-coverage MSP segment is preserved without hiding, smoothing, or interpolation. The table below records the small accepted-sample regime for coverage <= 1%.

{markdown_table(low_coverage_rows, ["split", "method", "coverage_cutoff", "low_coverage_points", "accepted_count_min", "accepted_count_max", "selective_risk_min", "selective_risk_max"])}

FIGURE_2_STATUS = COMPLETE
SOURCE_VALUES_VERIFIED = {str(source_values_verified).upper()}
SAFE_FOR_MAIN_PAPER = {str(safe_for_main).upper()}
"""
    AUDIT_OUT.write_text(audit, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _require([THRESHOLD_REGISTRY, FINAL_METRICS])
    curves = load_curve_data()
    points, op_audit = load_operating_points(curves)
    write_plot_data(curves, points)
    axes = make_figure(curves, points)
    curve_rows = curve_audit(curves, axes)
    low_coverage_rows = low_coverage_audit(curves)
    write_audit(curves, op_audit, curve_rows, low_coverage_rows, axes)
    print(f"Wrote {PDF_OUT.relative_to(ROOT)}")
    print(f"Wrote {PNG_OUT.relative_to(ROOT)}")
    print(f"Wrote {PLOT_DATA_OUT.relative_to(ROOT)}")
    print(f"Wrote {AUDIT_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
