"""Group-controlled selective-risk threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class GroupBound:
    group: str
    accepted: int
    errors: int
    empirical_risk: float
    cp_upper: float
    coverage: float


@dataclass(frozen=True)
class ThresholdSelection:
    tau: float | None
    coverage: float
    worst_cp_upper: float
    group_bounds: tuple[GroupBound, ...]


def binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    p = min(max(float(p), 0.0), 1.0)
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0 if k < n else 1.0
    total = 0.0
    for i in range(k + 1):
        total += comb(n, i) * (p**i) * ((1.0 - p) ** (n - i))
    return min(max(total, 0.0), 1.0)


def clopper_pearson_upper(errors: int, accepted: int, delta: float) -> float:
    """Upper one-sided Clopper-Pearson bound for a binomial error rate."""
    if accepted <= 0:
        return 1.0
    if errors < 0 or errors > accepted:
        raise ValueError("errors must satisfy 0 <= errors <= accepted")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if errors == accepted:
        return 1.0
    if errors == 0:
        return 1.0 - delta ** (1.0 / accepted)

    lo, hi = 0.0, 1.0
    target = delta
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if binomial_cdf(errors, accepted, mid) >= target:
            lo = mid
        else:
            hi = mid
    return hi


def group_bounds_at_threshold(
    risks: list[float],
    errors: list[int],
    groups: list[str],
    tau: float,
    delta: float,
) -> tuple[GroupBound, ...]:
    if not (len(risks) == len(errors) == len(groups)):
        raise ValueError("risks, errors, and groups must have the same length")
    unique_groups = sorted(set(groups))
    bounds = []
    for group in unique_groups:
        group_mask = [g == group for g in groups]
        group_total = sum(group_mask)
        accepted_mask = [m and r <= tau for m, r in zip(group_mask, risks)]
        accepted = sum(accepted_mask)
        err = sum(int(e) for e, m in zip(errors, accepted_mask) if m)
        empirical = err / accepted if accepted else 1.0
        bounds.append(
            GroupBound(
                group=group,
                accepted=accepted,
                errors=err,
                empirical_risk=empirical,
                cp_upper=clopper_pearson_upper(err, accepted, delta),
                coverage=accepted / group_total if group_total else 0.0,
            )
        )
    return tuple(bounds)


def select_group_controlled_threshold(
    risks: list[float],
    errors: list[int],
    groups: list[str],
    alpha: float = 0.05,
    delta: float = 0.05,
    bonferroni: bool = True,
) -> ThresholdSelection:
    """Maximize coverage subject to every calibration group satisfying CP <= alpha."""
    if not risks:
        return ThresholdSelection(None, 0.0, 1.0, tuple())
    if not (len(risks) == len(errors) == len(groups)):
        raise ValueError("risks, errors, and groups must have the same length")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    unique_groups = sorted(set(groups))
    group_delta = delta / len(unique_groups) if bonferroni and unique_groups else delta
    candidates = sorted(set(float(r) for r in risks))
    best: ThresholdSelection | None = None

    for tau in candidates:
        bounds = group_bounds_at_threshold(risks, errors, groups, tau, group_delta)
        worst = max((bound.cp_upper for bound in bounds), default=1.0)
        if worst <= alpha:
            coverage = sum(1 for risk in risks if risk <= tau) / len(risks)
            if best is None or coverage > best.coverage:
                best = ThresholdSelection(tau, coverage, worst, bounds)

    if best is not None:
        return best
    return ThresholdSelection(None, 0.0, 1.0, tuple())
