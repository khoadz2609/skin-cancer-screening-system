"""Data preparation utilities for numeric column preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import BASE_NUMERIC_COLS
except ImportError:
    from config import BASE_NUMERIC_COLS


def _prepare_base_numeric(df: pd.DataFrame, err: float) -> pd.DataFrame:
    """Prepare numeric columns: convert types, handle missing values, and fix infinities.

    Args:
        df: Input DataFrame with potentially mixed types and missing values.
        err: Small epsilon value for numerical stability.

    Returns:
        DataFrame with numeric columns properly typed and cleaned (NaN-filled with medians,
        infinities replaced, converted to float32).
    """
    out = df.copy()

    if "age_approx" in out.columns:
        out["age_approx"] = (
            out["age_approx"].replace("NA", np.nan).pipe(pd.to_numeric, errors="coerce")
        )

    base_cols = [c for c in BASE_NUMERIC_COLS if c in out.columns]
    for col in base_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if base_cols:
        out[base_cols] = out[base_cols].replace([np.inf, -np.inf], np.nan)
        medians = out[base_cols].median(numeric_only=True)
        out[base_cols] = out[base_cols].fillna(medians)
        out[base_cols] = out[base_cols].astype(np.float32, copy=False)

    return out


def _check_required_columns(df: pd.DataFrame, group_col: str) -> None:
    """Validate that all required columns are present in DataFrame.

    Args:
        df: Input DataFrame to validate.
        group_col: Name of the grouping column (e.g., 'patient_id').

    Raises:
        KeyError: If any required columns are missing.
    """
    required = set(BASE_NUMERIC_COLS)
    required.add(group_col)
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise KeyError(
            "Missing required columns for feature engineering: " + ", ".join(missing)
        )
