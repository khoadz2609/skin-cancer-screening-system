"""Preprocessing and encoding for tabular data using scalers and encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

try:
    from .tabular_selection_03 import _select_numeric_by_null_importance
except ImportError:
    from tabular_selection_03 import _select_numeric_by_null_importance


def _build_one_hot_encoder() -> OneHotEncoder:
    """Initialize OneHotEncoder with float32 output for compatibility.

    Returns:
        OneHotEncoder configured for handling unknown categories and float32 dtype.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def select_available_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_candidates: Iterable[str],
    categorical_candidates: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Select columns that exist in both train and test sets.

    Filters candidate columns to only those present in both datasets to ensure
    consistent preprocessing and feature availability.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        numeric_candidates: Candidate numeric column names.
        categorical_candidates: Candidate categorical column names.

    Returns:
        Tuple of (numeric_cols, categorical_cols) filtered to common columns.
    """
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    numeric_cols = [
        col for col in numeric_candidates if col in train_cols and col in test_cols
    ]
    categorical_cols = [
        col for col in categorical_candidates if col in train_cols and col in test_cols
    ]
    return numeric_cols, categorical_cols


@dataclass
class FoldTabularPreprocessor:
    """Preprocessing pipeline for tabular data: scaling, encoding, optional quantile transform.

    Attributes:
        numeric_cols: List of numeric column names to scale.
        categorical_cols: List of categorical column names to one-hot encode.
        use_quantile_transform: Whether to apply quantile transformer (default: False).
        quantile_noise_std: Standard deviation of Gaussian noise added before quantile transform.
        quantile_random_state: Random seed for quantile transformer.
        use_null_importance_filter: Whether to filter features by null importance (default: False).
        null_importance_repeats: Number of shuffle repeats for null importance (default: 8).
        null_importance_quantile: Quantile threshold for null importance (default: 0.95).
        null_importance_min_kept: Minimum features to keep after filtering (default: 64).
        target_col: Name of target column (required if use_null_importance_filter=True).
    """

    numeric_cols: list[str]
    categorical_cols: list[str]
    use_quantile_transform: bool = False
    quantile_noise_std: float = 1e-5
    quantile_random_state: int = 42
    use_null_importance_filter: bool = False
    null_importance_repeats: int = 8
    null_importance_quantile: float = 0.95
    null_importance_min_kept: int = 64
    target_col: str = "target"

    def __post_init__(self) -> None:
        """Initialize scalers, encoders, and preprocessing state."""
        self.numeric_cols = list(dict.fromkeys(self.numeric_cols))
        self.active_numeric_cols = list(self.numeric_cols)
        self.numeric_medians: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.encoder = _build_one_hot_encoder()
        self.quantile_transformer: QuantileTransformer | None = None
        self.is_fitted = False

    def _transform_base(self, df: pd.DataFrame) -> np.ndarray:
        """Transform numeric and categorical columns into concatenated feature matrix.

        Args:
            df: Input DataFrame.

        Returns:
            Feature matrix of shape (n_samples, n_scaled + n_encoded_features).

        Raises:
            ValueError: If no columns are available for preprocessing.
        """
        chunks: list[np.ndarray] = []

        if self.active_numeric_cols:
            numeric_frame = df[self.active_numeric_cols].copy()
            numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
            numeric_filled = numeric_frame.fillna(self.numeric_medians)
            numeric_scaled = self.scaler.transform(numeric_filled.values).astype(
                np.float32
            )
            chunks.append(numeric_scaled)

        if self.categorical_cols:
            cat_frame = df[self.categorical_cols].copy()
            cat_frame = cat_frame.astype("object").fillna("UNK").astype(str)
            cat_encoded = self.encoder.transform(cat_frame).astype(np.float32)
            chunks.append(cat_encoded)

        if not chunks:
            raise ValueError("No tabular columns are available after preprocessing.")

        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=1)

    def fit(self, df: pd.DataFrame) -> None:
        """Fit scalers, encoders, and optionally apply null importance filtering.

        Args:
            df: Training DataFrame. Must contain numeric, categorical, and optionally target columns.
        """
        if self.active_numeric_cols:
            numeric_frame = df[self.active_numeric_cols].copy()
            numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")

            if self.use_null_importance_filter and self.target_col in df.columns:
                selected_numeric = _select_numeric_by_null_importance(
                    numeric_frame=numeric_frame,
                    target=df[self.target_col],
                    repeats=self.null_importance_repeats,
                    quantile=self.null_importance_quantile,
                    min_kept=self.null_importance_min_kept,
                    random_state=self.quantile_random_state,
                )
                if selected_numeric and len(selected_numeric) < len(
                    self.active_numeric_cols
                ):
                    self.active_numeric_cols = selected_numeric
                    numeric_frame = numeric_frame[self.active_numeric_cols]
                    print(
                        "Null-importance filter kept "
                        f"{len(self.active_numeric_cols)}/{len(self.numeric_cols)} "
                        "numeric features."
                    )

            self.numeric_medians = {
                col: float(numeric_frame[col].median())
                for col in self.active_numeric_cols
            }
            numeric_filled = numeric_frame.fillna(self.numeric_medians)
            self.scaler.fit(numeric_filled.values)

        if self.categorical_cols:
            cat_frame = df[self.categorical_cols].copy()
            cat_frame = cat_frame.astype("object").fillna("UNK").astype(str)
            self.encoder.fit(cat_frame)

        if self.use_quantile_transform:
            base_matrix = self._transform_base(df)

            if len(base_matrix) >= 2:
                n_quantiles = max(min(len(base_matrix) // 30, 1000), 10)
                n_quantiles = min(n_quantiles, len(base_matrix))

                noise = (
                    np.random.default_rng(self.quantile_random_state)
                    .normal(
                        loc=0.0,
                        scale=self.quantile_noise_std,
                        size=base_matrix.shape,
                    )
                    .astype(np.float32)
                )

                self.quantile_transformer = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    subsample=10**9,
                    random_state=self.quantile_random_state,
                )
                self.quantile_transformer.fit(base_matrix + noise)
            else:
                self.quantile_transformer = None

        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame using fitted preprocessor.

        Args:
            df: DataFrame to transform. Must contain same columns as training data.

        Returns:
            Preprocessed feature matrix of shape (n_samples, n_features).

        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FoldTabularPreprocessor must be fitted before transform."
            )

        base_matrix = self._transform_base(df)
        if self.use_quantile_transform and self.quantile_transformer is not None:
            return self.quantile_transformer.transform(base_matrix).astype(np.float32)
        return base_matrix
