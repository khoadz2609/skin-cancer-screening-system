"""Feature selection utilities using correlation and null importance methods."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _absolute_feature_target_corr(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute absolute Pearson correlation between features and target.

    Args:
        matrix: Feature matrix of shape (n_samples, n_features), dtype float32.
        target: Target vector of shape (n_samples,), dtype float32.

    Returns:
        Array of absolute correlation coefficients for each feature, shape (n_features,).
    """
    centered_x = matrix - matrix.mean(axis=0, keepdims=True)
    centered_y = target - target.mean()

    numerator = np.abs(np.sum(centered_x * centered_y[:, None], axis=0))
    denominator = np.sqrt(
        np.sum(centered_x**2, axis=0) * np.sum(centered_y**2, axis=0) + 1e-8
    )
    corr = numerator / denominator
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


def _select_numeric_by_null_importance(
    numeric_frame: pd.DataFrame,
    target: pd.Series,
    repeats: int,
    quantile: float,
    min_kept: int,
    random_state: int,
) -> list[str]:
    """Filter numeric features using null importance permutation method.

    Compares feature-target correlation under actual target vs shuffled target.
    Keeps features where the margin (real_score - null_threshold) > 0, ensuring
    at least min_kept features are retained.

    Args:
        numeric_frame: DataFrame of numeric features.
        target: Target series for correlation computation.
        repeats: Number of random shuffles for null distribution.
        quantile: Quantile of null scores to use as threshold (0.5-0.999).
        min_kept: Minimum number of features to retain.
        random_state: Random seed for reproducibility.

    Returns:
        List of selected feature column names.
    """
    if numeric_frame.empty:
        return []

    matrix = numeric_frame.to_numpy(dtype=np.float32, copy=True)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    y = pd.to_numeric(target, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if len(y) < 16:
        return list(numeric_frame.columns)

    real_score = _absolute_feature_target_corr(matrix=matrix, target=y)

    n_repeats = max(1, int(repeats))
    clipped_quantile = float(np.clip(quantile, 0.5, 0.999))
    rng = np.random.default_rng(random_state)
    null_scores = np.zeros((n_repeats, matrix.shape[1]), dtype=np.float32)

    for i in range(n_repeats):
        shuffled = rng.permutation(y)
        null_scores[i] = _absolute_feature_target_corr(matrix=matrix, target=shuffled)

    null_threshold = np.quantile(null_scores, clipped_quantile, axis=0)
    margin = real_score - null_threshold
    keep_mask = margin > 0.0

    min_kept = max(1, min(int(min_kept), matrix.shape[1]))
    if int(keep_mask.sum()) < min_kept:
        keep_mask = np.zeros_like(keep_mask, dtype=bool)
        top_idx = np.argsort(margin)[-min_kept:]
        keep_mask[top_idx] = True

    return [
        col
        for col, keep in zip(numeric_frame.columns, keep_mask.tolist())
        if bool(keep)
    ]
