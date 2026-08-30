"""Bootstrap utilities used by the Phase 6 pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
        int(values.size),
    )


def stratified_count_bootstrap(
    df: pd.DataFrame,
    strata_cols: list[str],
    category_col: str,
    category_values: list[str],
    metric_fn: Callable[[dict[str, int]], float],
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Fast stratified bootstrap for metrics expressible by category counts."""

    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    if df.empty:
        draws.fill(float("nan"))
        return draws

    if not strata_cols:
        grouped = [((), df)]
    else:
        grouped = list(df.groupby(strata_cols, dropna=False, sort=True))
    strata = []
    for _, group in grouped:
        counts = group[category_col].value_counts().reindex(category_values, fill_value=0).to_numpy(dtype=np.float64)
        n = int(counts.sum())
        probs = counts / counts.sum() if counts.sum() else np.ones(len(category_values)) / len(category_values)
        strata.append((n, probs))

    for i in range(int(n_bootstrap)):
        aggregate = dict.fromkeys(category_values, 0)
        for n, probs in strata:
            sampled = rng.multinomial(n, probs)
            for value, count in zip(category_values, sampled):
                aggregate[value] += int(count)
        draws[i] = float(metric_fn(aggregate))
    return draws

