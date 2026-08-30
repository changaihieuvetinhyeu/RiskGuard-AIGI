"""RiskGuard Phase 5 calibration and ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


EPS = 1.0e-15


def clipped_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain NaN or Inf")
    return np.clip(probs, EPS, 1.0 - EPS)


def binary_nll(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    return float(log_loss(y, clipped_probabilities(probabilities), labels=[0, 1]))


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    return float(brier_score_loss(y, clipped_probabilities(probabilities)))


def reliability_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> pd.DataFrame:
    y = np.asarray(labels, dtype=np.int64)
    probs = clipped_probabilities(probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    for idx, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if idx == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        mean_pred = float(probs[mask].mean()) if count else float("nan")
        observed = float(y[mask].mean()) if count else float("nan")
        gap = abs(mean_pred - observed) if count else float("nan")
        rows.append(
            {
                "bin_index": idx,
                "lower_bound": float(lo),
                "upper_bound": float(hi),
                "sample_count": count,
                "mean_predicted_risk": mean_pred,
                "observed_error_rate": observed,
                "absolute_calibration_gap": float(gap) if count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ece_from_bins(bins: pd.DataFrame, total_count: int) -> float:
    if total_count <= 0:
        return float("nan")
    valid = bins["sample_count"].to_numpy(dtype=np.float64) > 0
    weights = bins.loc[valid, "sample_count"].to_numpy(dtype=np.float64) / float(total_count)
    gaps = bins.loc[valid, "absolute_calibration_gap"].to_numpy(dtype=np.float64)
    return float(np.sum(weights * gaps))


def maximum_calibration_error(bins: pd.DataFrame) -> float:
    values = bins["absolute_calibration_gap"].dropna().to_numpy(dtype=np.float64)
    return float(values.max()) if len(values) else float("nan")


def ordered_errors(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray | None = None) -> np.ndarray:
    y = np.asarray(errors, dtype=np.int64)
    r = np.asarray(risks, dtype=np.float64)
    if y.shape != r.shape:
        raise ValueError("errors and risks must have the same shape")
    if not np.isfinite(r).all():
        raise ValueError("risks contain NaN or Inf")
    if sample_ids is None:
        sample_ids = np.arange(len(y)).astype(str)
    order = np.lexsort((np.asarray(sample_ids).astype(str), r))
    return y[order]


def aurc(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray | None = None) -> float:
    sorted_errors = ordered_errors(errors, risks, sample_ids)
    n = len(sorted_errors)
    if n == 0:
        return float("nan")
    prefix_risk = np.cumsum(sorted_errors, dtype=np.float64) / np.arange(1, n + 1, dtype=np.float64)
    return float(prefix_risk.mean())


def optimal_aurc(errors: np.ndarray) -> float:
    y = np.asarray(errors, dtype=np.int64)
    n = len(y)
    if n == 0:
        return float("nan")
    sorted_errors = np.sort(y)
    prefix_risk = np.cumsum(sorted_errors, dtype=np.float64) / np.arange(1, n + 1, dtype=np.float64)
    return float(prefix_risk.mean())


def eaurc(errors: np.ndarray, risks: np.ndarray, sample_ids: np.ndarray | None = None) -> float:
    return float(aurc(errors, risks, sample_ids) - optimal_aurc(errors))


def error_detection_metrics(errors: np.ndarray, probabilities: np.ndarray) -> dict[str, float | str]:
    y = np.asarray(errors, dtype=np.int64)
    probs = clipped_probabilities(probabilities)
    if y.size == 0 or len(np.unique(y)) < 2:
        return {
            "error_detection_AUROC": float("nan"),
            "error_detection_AUPR": float("nan"),
            "ranking_status": "undefined_single_error_class",
        }
    return {
        "error_detection_AUROC": float(roc_auc_score(y, probs)),
        "error_detection_AUPR": float(average_precision_score(y, probs)),
        "ranking_status": "ok",
    }


def calibrator_metrics(
    errors: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_ids: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict[str, float | str | int]:
    y = np.asarray(errors, dtype=np.int64)
    probs = clipped_probabilities(probabilities)
    bins = reliability_bins(y, probs, n_bins=n_bins)
    ranking = error_detection_metrics(y, probs)
    return {
        "row_count": int(len(y)),
        "error_count": int(y.sum()),
        "error_prevalence": float(y.mean()) if len(y) else float("nan"),
        "binary_nll": binary_nll(y, probs) if len(y) else float("nan"),
        "brier_score": brier_score(y, probs) if len(y) else float("nan"),
        "ece": ece_from_bins(bins, len(y)),
        "maximum_calibration_error": maximum_calibration_error(bins),
        "error_detection_AUROC": ranking["error_detection_AUROC"],
        "error_detection_AUPR": ranking["error_detection_AUPR"],
        "AURC": aurc(y, probs, sample_ids),
        "E_AURC": eaurc(y, probs, sample_ids),
        "mean_predicted_risk": float(probs.mean()) if len(probs) else float("nan"),
        "observed_error_prevalence": float(y.mean()) if len(y) else float("nan"),
        "ranking_status": ranking["ranking_status"],
    }


def score_distribution(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    q25, q50, q75 = (np.nan, np.nan, np.nan) if len(finite) == 0 else np.percentile(finite, [25, 50, 75])
    return {
        "count": int(len(arr)),
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "standard_deviation": float(finite.std(ddof=0)) if len(finite) else float("nan"),
        "median": float(q50) if len(finite) else float("nan"),
        "IQR": float(q75 - q25) if len(finite) else float("nan"),
        "minimum": float(finite.min()) if len(finite) else float("nan"),
        "maximum": float(finite.max()) if len(finite) else float("nan"),
        "p01": float(np.percentile(finite, 1)) if len(finite) else float("nan"),
        "p99": float(np.percentile(finite, 99)) if len(finite) else float("nan"),
        "nonfinite_count": int((~np.isfinite(arr)).sum()),
    }
