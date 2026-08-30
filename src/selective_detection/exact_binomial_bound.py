"""Exact one-sided Clopper-Pearson bounds for Phase 6 risk control."""

from __future__ import annotations

from scipy.stats import beta


def clopper_pearson_upper(errors: int, accepted: int, delta: float) -> float:
    """Return the exact one-sided upper Clopper-Pearson bound.

    Phase 6 uses the finite-sample certificate
    BetaPPF(1 - delta; k + 1, n - k), with explicit handling of
    n=0 and k=n where the upper bound is one.
    """

    k = int(errors)
    n = int(accepted)
    d = float(delta)
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError("errors must satisfy 0 <= errors <= accepted")
    if not 0.0 < d < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - d, k + 1, n - k))

