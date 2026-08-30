"""Selective-classification metrics used by Phase 3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from selective_detection.selective_thresholds import clopper_pearson_upper
from scipy.stats import beta


@dataclass(frozen=True)
class RankingMetrics:
    auroc: float
    aupr: float
    status: str


def error_ranking_metrics(errors: np.ndarray, risks: np.ndarray) -> RankingMetrics:
    """AUROC/AUPR for detecting frozen detector errors as the positive class."""
    errors = np.asarray(errors, dtype=np.int64)
    risks = np.asarray(risks, dtype=np.float64)
    if errors.size == 0 or len(np.unique(errors)) < 2:
        return RankingMetrics(float("nan"), float("nan"), "undefined_due_to_single_error_class")
    return RankingMetrics(
        auroc=float(roc_auc_score(errors, risks)),
        aupr=float(average_precision_score(errors, risks)),
        status="ok",
    )


def ordered_errors(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray | None = None) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.int64)
    risks = np.asarray(risks, dtype=np.float64)
    if errors.shape != risks.shape:
        raise ValueError("errors and risks must have the same shape")
    if not np.isfinite(risks).all():
        raise ValueError("risks contain NaN or Inf")
    if sample_ids is None:
        sample_ids = np.arange(len(errors)).astype(str)
    order = np.lexsort((np.asarray(sample_ids).astype(str), risks))
    return errors[order]


def aurc(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray | None = None) -> float:
    """Discrete area under the risk-coverage curve."""
    sorted_errors = ordered_errors(errors, risks, sample_ids)
    n = len(sorted_errors)
    if n == 0:
        return float("nan")
    prefix_risk = np.cumsum(sorted_errors, dtype=np.float64) / np.arange(1, n + 1, dtype=np.float64)
    return float(prefix_risk.mean())


def optimal_aurc(errors: np.ndarray) -> float:
    errors = np.asarray(errors, dtype=np.int64)
    n = len(errors)
    if n == 0:
        return float("nan")
    sorted_errors = np.sort(errors)
    prefix_risk = np.cumsum(sorted_errors, dtype=np.float64) / np.arange(1, n + 1, dtype=np.float64)
    return float(prefix_risk.mean())


def eaurc(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray | None = None) -> float:
    value = aurc(errors, risks, sample_ids)
    optimum = optimal_aurc(errors)
    return float(value - optimum)


def risk_at_coverage(
    errors: np.ndarray,
    risks: np.ndarray,
    coverage: float,
    sample_ids: np.ndarray | None = None,
) -> float:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    sorted_errors = ordered_errors(errors, risks, sample_ids)
    if len(sorted_errors) == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(coverage) * len(sorted_errors))))
    return float(sorted_errors[:k].mean())


def coverage_under_empirical_risk(errors: np.ndarray, risks: np.ndarray, alpha: float, sample_ids: np.ndarray | None = None) -> float:
    sorted_errors = ordered_errors(errors, risks, sample_ids)
    if len(sorted_errors) == 0:
        return 0.0
    prefix_risk = np.cumsum(sorted_errors, dtype=np.float64) / np.arange(1, len(sorted_errors) + 1, dtype=np.float64)
    feasible = np.where(prefix_risk <= alpha)[0]
    return float((feasible[-1] + 1) / len(sorted_errors)) if len(feasible) else 0.0


def coverage_under_cp_risk(
    errors: np.ndarray,
    risks: np.ndarray,
    alpha: float,
    delta: float,
    sample_ids: np.ndarray | None = None,
) -> float:
    sorted_errors = ordered_errors(errors, risks, sample_ids)
    if len(sorted_errors) == 0:
        return 0.0
    cumulative_errors = np.cumsum(sorted_errors, dtype=np.int64)
    accepted = np.arange(1, len(sorted_errors) + 1, dtype=np.int64)
    cp = np.ones(len(sorted_errors), dtype=np.float64)
    zero = cumulative_errors == 0
    cp[zero] = 1.0 - float(delta) ** (1.0 / accepted[zero])
    middle = (~zero) & (cumulative_errors < accepted)
    cp[middle] = beta.ppf(1.0 - float(delta), cumulative_errors[middle] + 1, accepted[middle] - cumulative_errors[middle])
    feasible = np.where(cp <= alpha)[0]
    return float((feasible[-1] + 1) / len(sorted_errors)) if len(feasible) else 0.0


def accepted_error_metrics(df: pd.DataFrame, threshold: float | None) -> dict[str, float | str | int]:
    """Class-conditional accepted forensic error metrics at a frozen threshold."""
    if threshold is None or pd.isna(threshold):
        return {
            "accepted_count": 0,
            "coverage": 0.0,
            "selective_risk": float("nan"),
            "far_accepted": float("nan"),
            "fnr_accepted": float("nan"),
            "real_coverage": 0.0,
            "fake_coverage": 0.0,
            "balanced_selective_risk": float("nan"),
            "minimum_class_coverage": 0.0,
            "status": "no_feasible_nonempty_threshold",
        }
    accepted = df["risk_score"].to_numpy(dtype=np.float64) <= float(threshold)
    labels = df["label"].to_numpy(dtype=np.int64)
    predictions = df["base_prediction"].to_numpy(dtype=np.int64)
    errors = df["base_error"].to_numpy(dtype=np.int64)
    accepted_count = int(accepted.sum())
    if accepted_count == 0:
        return {
            "accepted_count": 0,
            "coverage": 0.0,
            "selective_risk": float("nan"),
            "far_accepted": float("nan"),
            "fnr_accepted": float("nan"),
            "real_coverage": 0.0,
            "fake_coverage": 0.0,
            "balanced_selective_risk": float("nan"),
            "minimum_class_coverage": 0.0,
            "status": "empty_acceptance",
        }
    real = labels == 0
    fake = labels == 1
    accepted_real = accepted & real
    accepted_fake = accepted & fake
    far = float(((predictions == 1) & accepted_real).sum() / accepted_real.sum()) if accepted_real.sum() else float("nan")
    fnr = float(((predictions == 0) & accepted_fake).sum() / accepted_fake.sum()) if accepted_fake.sum() else float("nan")
    real_cov = float(accepted_real.sum() / real.sum()) if real.sum() else float("nan")
    fake_cov = float(accepted_fake.sum() / fake.sum()) if fake.sum() else float("nan")
    return {
        "accepted_count": accepted_count,
        "coverage": float(accepted_count / len(df)),
        "selective_risk": float(errors[accepted].mean()),
        "far_accepted": far,
        "fnr_accepted": fnr,
        "real_coverage": real_cov,
        "fake_coverage": fake_cov,
        "balanced_selective_risk": float(np.nanmean([far, fnr])),
        "minimum_class_coverage": float(np.nanmin([real_cov, fake_cov])),
        "status": "ok",
    }


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return {
        "NLL": float(log_loss(labels, probabilities, labels=[0, 1])),
        "Brier": float(brier_score_loss(labels, probabilities)),
        "ECE": float(ece),
    }


def sha256_deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the lexicographically smallest sample_id per SHA-256."""
    required = {"sha256", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"deduplication requires columns: {sorted(missing)}")
    ordered = df.sort_values(["sha256", "sample_id"], kind="mergesort").copy()
    keep_mask = ~ordered["sha256"].duplicated(keep="first")
    kept = ordered[keep_mask].copy()
    canonical = kept[["sha256", "sample_id"]].rename(columns={"sample_id": "canonical_sample_id"})
    mapped = ordered.merge(canonical, on="sha256", how="left", validate="many_to_one")
    mapped["is_canonical"] = mapped["sample_id"].astype(str) == mapped["canonical_sample_id"].astype(str)
    dedup_map = mapped[["sha256", "canonical_sample_id", "sample_id", "is_canonical"]].rename(
        columns={"sample_id": "alias_sample_id"}
    )
    kept = kept.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return kept, dedup_map.reset_index(drop=True)


def summarize_selective_metrics(
    df: pd.DataFrame,
    evaluation_weighting: str,
    split: str,
    evaluation_role: str,
    generator: str = "all",
) -> list[dict[str, object]]:
    errors = df["base_error"].to_numpy(dtype=np.int64)
    risks = df["risk_score"].to_numpy(dtype=np.float64)
    sample_ids = df["sample_id"].astype(str).to_numpy()
    ranking = error_ranking_metrics(errors, risks)
    rows: list[dict[str, object]] = []
    base = {
        "detector": df["detector"].iloc[0],
        "baseline": df["baseline"].iloc[0],
        "split": split,
        "evaluation_role": evaluation_role,
        "generator": generator,
        "evaluation_weighting": evaluation_weighting,
        "sample_count": int(len(df)),
        "error_count": int(errors.sum()),
    }
    metric_values = {
        "error_detection_AUROC": ranking.auroc,
        "error_detection_AUPR": ranking.aupr,
        "AURC": aurc(errors, risks, sample_ids),
        "E_AURC": eaurc(errors, risks, sample_ids),
        "risk_at_50pct_coverage": risk_at_coverage(errors, risks, 0.50, sample_ids),
        "risk_at_70pct_coverage": risk_at_coverage(errors, risks, 0.70, sample_ids),
        "risk_at_80pct_coverage": risk_at_coverage(errors, risks, 0.80, sample_ids),
        "risk_at_90pct_coverage": risk_at_coverage(errors, risks, 0.90, sample_ids),
        "risk_at_95pct_coverage": risk_at_coverage(errors, risks, 0.95, sample_ids),
        "coverage_empirical_risk_le_1pct": coverage_under_empirical_risk(errors, risks, 0.01, sample_ids),
        "coverage_empirical_risk_le_5pct": coverage_under_empirical_risk(errors, risks, 0.05, sample_ids),
        "coverage_cp_risk_le_1pct": coverage_under_cp_risk(errors, risks, 0.01, 0.05, sample_ids),
        "coverage_cp_risk_le_5pct": coverage_under_cp_risk(errors, risks, 0.05, 0.05, sample_ids),
    }
    for metric, value in metric_values.items():
        status = ranking.status if metric.startswith("error_detection") else "ok"
        rows.append({**base, "metric": metric, "value": value, "status": status})
    return rows
