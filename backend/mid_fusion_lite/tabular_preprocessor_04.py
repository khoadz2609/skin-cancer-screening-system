"""Preprocessing and encoding for tabular data using scalers and encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

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
    categorical_as_codes: bool = False

    def __post_init__(self) -> None:
        """Initialize scalers, encoders, and preprocessing state."""
        self.numeric_cols = list(dict.fromkeys(self.numeric_cols))
        self.active_numeric_cols = list(self.numeric_cols)
        self.numeric_medians: dict[str, float] = {}
        self.encoder = _build_one_hot_encoder()
        self.quantile_transformer: QuantileTransformer | None = None
        # Label-encoding state (only used when categorical_as_codes=True). Code 0 is
        # reserved for unknown/unseen categories; known categories take codes 1..n.
        self.category_maps: dict[str, dict[str, int]] = {}
        self.cat_cardinalities: list[int] = []
        self.n_numeric_out: int = 0
        self.is_fitted = False

    def _transform_numeric(self, df: pd.DataFrame) -> np.ndarray | None:
        """Scale numeric columns. Returns None when there are no numeric columns."""
        if not self.active_numeric_cols:
            return None
        numeric_frame = df[self.active_numeric_cols].copy()
        numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
        numeric_filled = numeric_frame.fillna(self.numeric_medians)
        return numeric_filled.values.astype(np.float32)

    def _transform_categorical(self, df: pd.DataFrame) -> np.ndarray | None:
        """Encode categorical columns. Returns None when there are none.

        One-hot by default; integer codes when ``categorical_as_codes`` is set
        (one column per categorical feature, unknown/unseen -> 0).
        """
        if not self.categorical_cols:
            return None
        cat_frame = df[self.categorical_cols].copy()
        cat_frame = cat_frame.astype("object").fillna("UNK").astype(str)
        if self.categorical_as_codes:
            codes = np.zeros((len(cat_frame), len(self.categorical_cols)), dtype=np.float32)
            for j, col in enumerate(self.categorical_cols):
                mapping = self.category_maps[col]
                codes[:, j] = cat_frame[col].map(mapping).fillna(0).to_numpy(dtype=np.float32)
            return codes
        return self.encoder.transform(cat_frame).astype(np.float32)

    @staticmethod
    def _assemble(
        numeric_part: np.ndarray | None, cat_part: np.ndarray | None
    ) -> np.ndarray:
        """Concatenate numeric and categorical chunks into one feature matrix.

        Raises:
            ValueError: If both chunks are empty.
        """
        chunks = [c for c in (numeric_part, cat_part) if c is not None]
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

        # Number of numeric columns after optional null-importance filtering. Used by
        # callers to compute PLE bins on the numeric block only (codes come after).
        self.n_numeric_out = len(self.active_numeric_cols)

        if self.categorical_cols:
            cat_frame = df[self.categorical_cols].copy()
            cat_frame = cat_frame.astype("object").fillna("UNK").astype(str)
            if self.categorical_as_codes:
                # Label-encode: code 0 reserved for unknown/unseen; known categories 1..n.
                self.category_maps = {}
                self.cat_cardinalities = []
                for col in self.categorical_cols:
                    cats = sorted(cat_frame[col].unique().tolist())
                    self.category_maps[col] = {c: i + 1 for i, c in enumerate(cats)}
                    self.cat_cardinalities.append(len(cats) + 1)
            else:
                self.encoder.fit(cat_frame)

        if self.use_quantile_transform:
            # Official TabM recipe applies QuantileTransformer to the NUMERIC block
            # only (x_num), never to one-hot encoded categoricals — mirror that here.
            numeric_part = self._transform_numeric(df)

            if numeric_part is not None and len(numeric_part) >= 2:
                # Cap at 1000 quantiles to match the official TabM example; beyond
                # ~1000 landmarks the empirical CDF gains negligible resolution.
                n_quantiles = max(min(len(numeric_part) // 30, 1000), 10)
                n_quantiles = min(n_quantiles, len(numeric_part))

                rng = np.random.default_rng(self.quantile_random_state)
                # Generate noise directly in float32 and add in-place to avoid huge memory spikes
                noise = rng.standard_normal(size=numeric_part.shape, dtype=np.float32)
                noise *= np.float32(self.quantile_noise_std)
                numeric_part += noise
                
                self.quantile_transformer = QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    subsample=10**9,
                    random_state=self.quantile_random_state,
                )
                self.quantile_transformer.fit(numeric_part)
                del noise
                del numeric_part
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

        numeric_part = self._transform_numeric(df)
        if (
            numeric_part is not None
            and self.use_quantile_transform
            and self.quantile_transformer is not None
        ):
            numeric_part = self.quantile_transformer.transform(numeric_part).astype(
                np.float32
            )
        cat_part = self._transform_categorical(df)
        return self._assemble(numeric_part, cat_part)
