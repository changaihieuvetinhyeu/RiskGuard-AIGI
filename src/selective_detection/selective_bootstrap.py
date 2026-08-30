"""Bootstrap confidence intervals for Phase 3 selective baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def stratified_unit_bootstrap(
    df: pd.DataFrame,
    unit_col: str,
    strata_cols: list[str],
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap units within each stratum and return metric draws."""
    rng = np.random.default_rng(seed)
    if not strata_cols:
        grouped = [((), df)]
    else:
        grouped = list(df.groupby(strata_cols, dropna=False, sort=True))
    df = df.reset_index(drop=True)
    strata = []
    for _, group in grouped:
        group_index = group.index.to_numpy(dtype=np.int64)
        unit_series = group[unit_col].astype(str)
        if unit_series.is_unique:
            unit_values = unit_series.to_numpy()
            unit_to_indexes = {unit: np.array([idx], dtype=np.int64) for unit, idx in zip(unit_values, group_index)}
        else:
            unit_values = unit_series.drop_duplicates().to_numpy()
            unit_to_indexes = {
                unit: group_index[unit_series.to_numpy() == unit]
                for unit in unit_values
            }
        strata.append((unit_values, unit_to_indexes))
    draws = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        pieces = []
        for unit_values, unit_to_indexes in strata:
            if len(unit_values) == 0:
                continue
            sampled_units = rng.choice(unit_values, size=len(unit_values), replace=True)
            pieces.extend(unit_to_indexes[unit] for unit in sampled_units)
        if not pieces:
            draws[i] = float("nan")
        else:
            sampled = df.iloc[np.concatenate(pieces)]
            draws[i] = float(metric_fn(sampled))
    return draws
