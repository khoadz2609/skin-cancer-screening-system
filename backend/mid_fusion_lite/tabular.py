from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_one_hot_encoder() -> OneHotEncoder:
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
    numeric_cols: list[str]
    categorical_cols: list[str]

    def __post_init__(self) -> None:
        self.numeric_medians: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.encoder = _build_one_hot_encoder()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        if self.numeric_cols:
            numeric_frame = df[self.numeric_cols].copy()
            numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
            self.numeric_medians = {
                col: float(numeric_frame[col].median()) for col in self.numeric_cols
            }
            numeric_filled = numeric_frame.fillna(self.numeric_medians)
            self.scaler.fit(numeric_filled.values)

        if self.categorical_cols:
            cat_frame = df[self.categorical_cols].copy()
            cat_frame = cat_frame.fillna("UNK").astype(str)
            self.encoder.fit(cat_frame)

        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "FoldTabularPreprocessor must be fitted before transform."
            )

        chunks: list[np.ndarray] = []

        if self.numeric_cols:
            numeric_frame = df[self.numeric_cols].copy()
            numeric_frame = numeric_frame.apply(pd.to_numeric, errors="coerce")
            numeric_filled = numeric_frame.fillna(self.numeric_medians)
            numeric_scaled = self.scaler.transform(numeric_filled.values).astype(
                np.float32
            )
            chunks.append(numeric_scaled)

        if self.categorical_cols:
            cat_frame = df[self.categorical_cols].copy()
            cat_frame = cat_frame.fillna("UNK").astype(str)
            cat_encoded = self.encoder.transform(cat_frame).astype(np.float32)
            chunks.append(cat_encoded)

        if not chunks:
            raise ValueError("No tabular columns are available after preprocessing.")

        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=1)
