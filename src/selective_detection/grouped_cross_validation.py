"""Deterministic SHA-grouped cross-validation for RiskGuard Phase 5."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_FOLD_COLUMNS = ("sample_id", "sha256", "base_error", "label", "generator")


def make_strata(df: pd.DataFrame) -> np.ndarray:
    missing = [column for column in REQUIRED_FOLD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing required fold column(s): {missing}")
    generator = df["generator"].astype(str).fillna("unknown")
    return (
        df["base_error"].astype(int).astype(str)
        + "|"
        + df["label"].astype(int).astype(str)
        + "|"
        + generator
    ).to_numpy()


def assign_sha_grouped_folds(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 20260916,
) -> pd.DataFrame:
    """Assign deterministic folds while keeping identical SHA-256 rows together."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = df["sha256"].astype(str).to_numpy()
    y = make_strata(df)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=np.int64)
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(df)), y, groups=groups)):
        folds[val_idx] = fold
    if (folds < 0).any():
        raise RuntimeError("some rows were not assigned to a CV fold")
    out = df[["sample_id", "sha256"]].copy()
    out["cv_fold"] = folds
    return out


def assert_no_sha_overlap(folded: pd.DataFrame) -> None:
    per_sha = folded.groupby("sha256")["cv_fold"].nunique()
    leaked = per_sha[per_sha > 1]
    if len(leaked):
        raise RuntimeError(f"SHA crossed folds: {leaked.index[:5].tolist()}")


def fold_audit_rows(df: pd.DataFrame, *, detector: str, split: str) -> list[dict[str, object]]:
    required = {"cv_fold", "sha256", "base_error", "label", "generator"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fold audit missing column(s): {sorted(missing)}")
    assert_no_sha_overlap(df)
    rows: list[dict[str, object]] = []
    all_sha = set(df["sha256"].astype(str))
    for fold in sorted(df["cv_fold"].unique()):
        part = df[df["cv_fold"].eq(fold)]
        other_sha = set(df.loc[~df["cv_fold"].eq(fold), "sha256"].astype(str))
        overlap = len(set(part["sha256"].astype(str)) & other_sha)
        correct_count = int((part["base_error"].astype(int) == 0).sum())
        error_count = int((part["base_error"].astype(int) == 1).sum())
        real_count = int((part["label"].astype(int) == 0).sum())
        fake_count = int((part["label"].astype(int) == 1).sum())
        status = "pass" if overlap == 0 and min(correct_count, error_count, real_count, fake_count) > 0 else "fail"
        rows.append(
            {
                "detector": detector,
                "split": split,
                "fold": int(fold),
                "row_count": int(len(part)),
                "unique_sha256": int(part["sha256"].nunique()),
                "correct_count": correct_count,
                "error_count": error_count,
                "real_count": real_count,
                "fake_count": fake_count,
                "generator_counts_json": json.dumps(part["generator"].astype(str).value_counts().sort_index().to_dict(), sort_keys=True),
                "sha_overlap_with_other_folds": int(overlap),
                "status": status,
            }
        )
    if set(df["sha256"].astype(str)) != all_sha:
        raise RuntimeError("internal SHA accounting error")
    return rows
