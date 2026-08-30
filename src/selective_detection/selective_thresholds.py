"""Global selective-risk threshold selection for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta


@dataclass(frozen=True)
class GlobalThresholdResult:
    threshold: float | None
    accepted_count: int
    total_count: int
    coverage: float
    accepted_errors: int
    empirical_risk: float
    cp_upper: float
    selection_status: str


def clopper_pearson_upper(errors: int, accepted: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound for a binomial error rate."""
    errors = int(errors)
    accepted = int(accepted)
    if accepted <= 0:
        return 1.0
    if errors < 0 or errors > accepted:
        raise ValueError("errors must satisfy 0 <= errors <= accepted")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if errors == accepted:
        return 1.0
    if errors == 0:
        return float(1.0 - delta ** (1.0 / accepted))
    return float(beta.ppf(1.0 - delta, errors + 1, accepted - errors))


def threshold_search_curve(
    risks: np.ndarray,
    errors: np.ndarray,
    sample_ids: np.ndarray | None,
    alpha: float,
    delta: float,
) -> pd.DataFrame:
    """Evaluate every unique risk threshold with deterministic stable ordering."""
    risks = np.asarray(risks, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.int64)
    if risks.shape != errors.shape:
        raise ValueError("risks and errors must have the same shape")
    if not np.isfinite(risks).all():
        raise ValueError("risks contain NaN or Inf")
    if not np.isin(errors, [0, 1]).all():
        raise ValueError("errors must be binary")
    if sample_ids is None:
        sample_ids = np.arange(len(risks)).astype(str)
    sample_ids = np.asarray(sample_ids).astype(str)
    order = np.lexsort((sample_ids, risks))
    sorted_risks = risks[order]
    sorted_errors = errors[order]
    rows = []
    total = int(len(risks))
    if total == 0:
        return pd.DataFrame(
            columns=[
                "threshold",
                "accepted_count",
                "total_count",
                "coverage",
                "accepted_errors",
                "empirical_risk",
                "cp_upper",
                "is_feasible",
            ]
        )
    cumulative_errors = np.cumsum(sorted_errors)
    unique_values, last_indexes = np.unique(sorted_risks, return_index=False, return_counts=False), None
    _, last_indexes = np.unique(sorted_risks, return_index=True)
    next_starts = np.r_[last_indexes[1:], total]
    for tau, end in zip(unique_values, next_starts):
        accepted = int(end)
        accepted_errors = int(cumulative_errors[end - 1])
        empirical = accepted_errors / accepted if accepted else 1.0
        upper = clopper_pearson_upper(accepted_errors, accepted, delta)
        rows.append(
            {
                "threshold": float(tau),
                "accepted_count": accepted,
                "total_count": total,
                "coverage": accepted / total,
                "accepted_errors": accepted_errors,
                "empirical_risk": float(empirical),
                "cp_upper": upper,
                "is_feasible": bool(accepted > 0 and upper <= alpha),
            }
        )
    return pd.DataFrame(rows)


def select_global_threshold(
    risks: np.ndarray,
    errors: np.ndarray,
    sample_ids: np.ndarray | None = None,
    alpha: float = 0.05,
    delta: float = 0.05,
) -> tuple[GlobalThresholdResult, pd.DataFrame]:
    """Select the highest-coverage non-empty threshold satisfying CP <= alpha."""
    curve = threshold_search_curve(risks, errors, sample_ids, alpha, delta)
    if curve.empty:
        return (
            GlobalThresholdResult(None, 0, 0, 0.0, 0, 1.0, 1.0, "no_samples"),
            curve,
        )
    feasible = curve[curve["is_feasible"]].copy()
    if feasible.empty:
        return (
            GlobalThresholdResult(
                None,
                0,
                int(curve["total_count"].iloc[0]),
                0.0,
                0,
                1.0,
                1.0,
                "no_feasible_nonempty_threshold",
            ),
            curve,
        )
    feasible = feasible.sort_values(
        by=["accepted_count", "cp_upper", "empirical_risk", "threshold"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    row = feasible.iloc[0]
    return (
        GlobalThresholdResult(
            threshold=float(row["threshold"]),
            accepted_count=int(row["accepted_count"]),
            total_count=int(row["total_count"]),
            coverage=float(row["coverage"]),
            accepted_errors=int(row["accepted_errors"]),
            empirical_risk=float(row["empirical_risk"]),
            cp_upper=float(row["cp_upper"]),
            selection_status="selected",
        ),
        curve,
    )
