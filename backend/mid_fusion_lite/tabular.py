from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

try:
    from .config import BASE_NUMERIC_COLS, ENGINEERED_NUMERIC_COLS
except ImportError:
    from config import BASE_NUMERIC_COLS, ENGINEERED_NUMERIC_COLS


def _prepare_base_numeric(df: pd.DataFrame, err: float) -> pd.DataFrame:
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
    required = set(BASE_NUMERIC_COLS)
    required.add(group_col)
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise KeyError(
            "Missing required columns for feature engineering: " + ", ".join(missing)
        )


def _add_engineered_columns(
    df: pd.DataFrame,
    group_col: str,
    categorical_cols: list[str],
    err: float,
) -> pd.DataFrame:
    out = _prepare_base_numeric(df=df, err=err)
    _check_required_columns(df=out, group_col=group_col)

    out["lesion_size_ratio"] = out["tbp_lv_minorAxisMM"] / (
        out["clin_size_long_diam_mm"] + err
    )
    out["lesion_shape_index"] = out["tbp_lv_areaMM2"] / (
        np.square(out["tbp_lv_perimeterMM"]) + err
    )
    out["hue_contrast"] = np.abs(out["tbp_lv_H"] - out["tbp_lv_Hext"])
    out["luminance_contrast"] = np.abs(out["tbp_lv_L"] - out["tbp_lv_Lext"])
    out["lesion_color_difference"] = np.sqrt(
        np.square(out["tbp_lv_deltaA"])
        + np.square(out["tbp_lv_deltaB"])
        + np.square(out["tbp_lv_deltaL"])
    )
    out["border_complexity"] = out["tbp_lv_norm_border"] + out["tbp_lv_symm_2axis"]
    out["color_uniformity"] = out["tbp_lv_color_std_mean"] / (
        out["tbp_lv_radial_color_std_max"] + err
    )
    out["position_distance_3d"] = np.sqrt(
        np.square(out["tbp_lv_x"])
        + np.square(out["tbp_lv_y"])
        + np.square(out["tbp_lv_z"])
    )
    out["perimeter_to_area_ratio"] = out["tbp_lv_perimeterMM"] / (
        out["tbp_lv_areaMM2"] + err
    )
    out["area_to_perimeter_ratio"] = out["tbp_lv_areaMM2"] / (
        out["tbp_lv_perimeterMM"] + err
    )
    out["lesion_visibility_score"] = (
        out["tbp_lv_deltaLBnorm"] + out["tbp_lv_norm_color"]
    )
    out["symmetry_border_consistency"] = (
        out["tbp_lv_symm_2axis"] * out["tbp_lv_norm_border"]
    )
    out["consistency_symmetry_border"] = (
        out["tbp_lv_symm_2axis"] * out["tbp_lv_norm_border"]
    ) / (out["tbp_lv_symm_2axis"] + out["tbp_lv_norm_border"] + err)

    out["color_consistency"] = out["tbp_lv_stdL"] / (out["tbp_lv_Lext"] + err)
    out["consistency_color"] = (out["tbp_lv_stdL"] * out["tbp_lv_Lext"]) / (
        out["tbp_lv_stdL"] + out["tbp_lv_Lext"] + err
    )
    out["size_age_interaction"] = out["clin_size_long_diam_mm"] * out["age_approx"]
    out["hue_color_std_interaction"] = out["tbp_lv_H"] * out["tbp_lv_color_std_mean"]
    out["lesion_severity_index"] = (
        out["tbp_lv_norm_border"]
        + out["tbp_lv_norm_color"]
        + out["tbp_lv_eccentricity"]
    ) / 3.0
    out["shape_complexity_index"] = out["border_complexity"] + out["lesion_shape_index"]
    out["color_contrast_index"] = (
        out["tbp_lv_deltaA"]
        + out["tbp_lv_deltaB"]
        + out["tbp_lv_deltaL"]
        + out["tbp_lv_deltaLBnorm"]
    )

    out["log_lesion_area"] = np.log(out["tbp_lv_areaMM2"] + 1.0)
    out["normalized_lesion_size"] = out["clin_size_long_diam_mm"] / (
        out["age_approx"] + err
    )
    out["mean_hue_difference"] = (out["tbp_lv_H"] + out["tbp_lv_Hext"]) / 2.0
    out["std_dev_contrast"] = np.sqrt(
        (
            np.square(out["tbp_lv_deltaA"])
            + np.square(out["tbp_lv_deltaB"])
            + np.square(out["tbp_lv_deltaL"])
        )
        / 3.0
    )
    out["color_shape_composite_index"] = (
        out["tbp_lv_color_std_mean"]
        + out["tbp_lv_area_perim_ratio"]
        + out["tbp_lv_symm_2axis"]
    ) / 3.0
    out["lesion_orientation_3d"] = np.arctan2(out["tbp_lv_y"], out["tbp_lv_x"])
    out["overall_color_difference"] = (
        out["tbp_lv_deltaA"] + out["tbp_lv_deltaB"] + out["tbp_lv_deltaL"]
    ) / 3.0

    out["symmetry_perimeter_interaction"] = (
        out["tbp_lv_symm_2axis"] * out["tbp_lv_perimeterMM"]
    )
    out["comprehensive_lesion_index"] = (
        out["tbp_lv_area_perim_ratio"]
        + out["tbp_lv_eccentricity"]
        + out["tbp_lv_norm_color"]
        + out["tbp_lv_symm_2axis"]
    ) / 4.0
    out["color_variance_ratio"] = out["tbp_lv_color_std_mean"] / (
        out["tbp_lv_stdLExt"] + err
    )
    out["border_color_interaction"] = (
        out["tbp_lv_norm_border"] * out["tbp_lv_norm_color"]
    )
    out["border_color_interaction_2"] = (
        out["tbp_lv_norm_border"] * out["tbp_lv_norm_color"]
    ) / (out["tbp_lv_norm_border"] + out["tbp_lv_norm_color"] + err)
    out["size_color_contrast_ratio"] = out["clin_size_long_diam_mm"] / (
        out["tbp_lv_deltaLBnorm"] + err
    )
    out["age_normalized_nevi_confidence"] = out["tbp_lv_nevi_confidence"] / (
        out["age_approx"] + err
    )
    out["age_normalized_nevi_confidence_2"] = np.sqrt(
        np.square(out["clin_size_long_diam_mm"]) + np.square(out["age_approx"])
    )
    out["color_asymmetry_index"] = (
        out["tbp_lv_radial_color_std_max"] * out["tbp_lv_symm_2axis"]
    )

    out["volume_approximation_3d"] = out["tbp_lv_areaMM2"] * np.sqrt(
        np.square(out["tbp_lv_x"])
        + np.square(out["tbp_lv_y"])
        + np.square(out["tbp_lv_z"])
    )
    out["color_range"] = (
        np.abs(out["tbp_lv_L"] - out["tbp_lv_Lext"])
        + np.abs(out["tbp_lv_A"] - out["tbp_lv_Aext"])
        + np.abs(out["tbp_lv_B"] - out["tbp_lv_Bext"])
    )
    out["shape_color_consistency"] = (
        out["tbp_lv_eccentricity"] * out["tbp_lv_color_std_mean"]
    )
    out["border_length_ratio"] = out["tbp_lv_perimeterMM"] / (
        2.0 * np.pi * np.sqrt(out["tbp_lv_areaMM2"] / np.pi + err)
    )
    out["age_size_symmetry_index"] = (
        out["age_approx"] * out["clin_size_long_diam_mm"] * out["tbp_lv_symm_2axis"]
    )
    out["index_age_size_symmetry"] = (
        out["age_approx"] * out["tbp_lv_areaMM2"] * out["tbp_lv_symm_2axis"]
    )

    if group_col in out.columns:
        out[group_col] = out[group_col].fillna("UNK").astype(str).astype("category")

    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].fillna("UNK").astype(str).astype("category")

    generated_features: dict[str, pd.Series] = {}
    patient_grouped = out.groupby(group_col, observed=False, sort=False)

    for col in BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS:
        group_mean = patient_grouped[col].transform("mean")
        group_std = patient_grouped[col].transform("std")
        generated_features[f"{col}_patient_norm"] = (out[col] - group_mean) / (
            group_std + err
        )

    for cat_col in categorical_cols:
        if cat_col not in out.columns:
            continue
        grouped = out.groupby([group_col, cat_col], observed=False, sort=False)
        for num_col in BASE_NUMERIC_COLS:
            group_mean = grouped[num_col].transform("mean")
            group_std = grouped[num_col].transform("std")
            generated_features[f"{num_col}_{cat_col}"] = (out[num_col] - group_mean) / (
                group_std + err
            )

    for col in categorical_cols:
        if col not in out.columns:
            continue
        generated_features[f"{col}_count"] = out.groupby(
            [group_col, col], observed=False, sort=False
        )[col].transform("count")

    generated_features["count_per_patient"] = patient_grouped[group_col].transform(
        "count"
    )

    if generated_features:
        out = pd.concat(
            [out, pd.DataFrame(generated_features, index=out.index)],
            axis=1,
            copy=False,
        )

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out[numeric_cols].fillna(out[numeric_cols].median())
    out[numeric_cols] = out[numeric_cols].astype(np.float32, copy=False)

    return out


def add_feature_engineering(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    categorical_cols: list[str],
    err: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_enhanced = _add_engineered_columns(
        df=train_df,
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
    )
    test_enhanced = _add_engineered_columns(
        df=test_df,
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
    )
    return train_enhanced, test_enhanced


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
    use_quantile_transform: bool = False
    quantile_noise_std: float = 1e-5
    quantile_random_state: int = 42

    def __post_init__(self) -> None:
        self.numeric_medians: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.encoder = _build_one_hot_encoder()
        self.quantile_transformer: QuantileTransformer | None = None
        self.is_fitted = False

    def _transform_base(self, df: pd.DataFrame) -> np.ndarray:
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
        if not self.is_fitted:
            raise RuntimeError(
                "FoldTabularPreprocessor must be fitted before transform."
            )

        base_matrix = self._transform_base(df)
        if self.use_quantile_transform and self.quantile_transformer is not None:
            return self.quantile_transformer.transform(base_matrix).astype(np.float32)
        return base_matrix
