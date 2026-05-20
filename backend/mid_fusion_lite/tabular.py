from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
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


def _compute_patient_lof_feature(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: list[str],
    min_group_size: int,
) -> pd.Series:
    scores = pd.Series(np.nan, index=df.index, dtype=np.float32)
    grouped = df.groupby(group_col, observed=False, sort=False).groups

    for _, group_index in grouped.items():
        if len(group_index) < max(3, int(min_group_size)):
            continue

        n_neighbors = min(20, len(group_index) - 1)
        if n_neighbors < 2:
            continue

        matrix = df.loc[group_index, feature_cols].to_numpy(dtype=np.float32, copy=True)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination="auto",
            )
            _ = lof.fit_predict(matrix)
            scores.loc[group_index] = -lof.negative_outlier_factor_.astype(np.float32)
        except ValueError:
            continue

    if scores.notna().any():
        fill_value = float(scores.median(skipna=True))
    else:
        fill_value = 1.0
    return scores.fillna(fill_value).astype(np.float32)


def _add_contextual_group_norm_features(
    out: pd.DataFrame,
    generated_features: dict[str, pd.Series],
    group_col: str,
    source_cols: list[str],
    err: float,
) -> None:
    context_groups = [
        ("patient_loc_norm", "tbp_lv_location"),
        ("patient_loc_sim_norm", "tbp_lv_location_simple"),
        ("patient_site_general_norm", "anatom_site_general"),
    ]

    for suffix, context_col in context_groups:
        if context_col not in out.columns:
            continue

        grouped = out.groupby([group_col, context_col], observed=False, sort=False)
        for col in source_cols:
            group_mean = grouped[col].transform("mean")
            group_std = grouped[col].transform("std")
            generated_features[f"{col}_{suffix}"] = (out[col] - group_mean) / (
                group_std + err
            )


def _add_ugly_duckling_features(
    out: pd.DataFrame,
    generated_features: dict[str, pd.Series],
    group_col: str,
    source_cols: list[str],
    err: float,
) -> None:
    loc_col = None
    for candidate in ["tbp_lv_location", "tbp_lv_location_simple", "anatom_site_general"]:
        if candidate in out.columns:
            loc_col = candidate
            break
    if loc_col is None:
        return

    patient_grouped = out.groupby(group_col, observed=False, sort=False)
    loc_grouped = out.groupby([group_col, loc_col], observed=False, sort=False)

    ud_loc_cols: list[str] = []
    for col in source_cols:
        loc_mean = loc_grouped[col].transform("mean")
        loc_std = loc_grouped[col].transform("std")

        ud_loc_col = f"ud_loc_{col}"
        generated_features[ud_loc_col] = (out[col] - loc_mean) / (loc_std + err)
        ud_loc_cols.append(ud_loc_col)

        generated_features[f"ud_percentile_{col}"] = patient_grouped[col].rank(pct=True)

    if not ud_loc_cols:
        return

    ud_abs = pd.DataFrame(
        {col: generated_features[col].abs() for col in ud_loc_cols},
        index=out.index,
    )
    generated_features["ud_count_location"] = (ud_abs > 2.0).sum(axis=1)
    generated_features["ud_max_severity_location"] = ud_abs.max(axis=1)
    generated_features["ud_consistency_location"] = ud_abs.std(axis=1)


def _add_engineered_columns(
    df: pd.DataFrame,
    group_col: str,
    categorical_cols: list[str],
    err: float,
    use_contextual_group_norm_features: bool,
    use_ugly_duckling_features: bool,
    use_anatomical_zscore_features: bool,
    use_area_aggregation_features: bool,
    use_patient_lof_features: bool,
    patient_lof_min_group_size: int,
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

    additional_features = {
        "asymmetry_ratio": out["tbp_lv_symm_2axis"]
        / (np.abs(out["tbp_lv_symm_2axis_angle"]) + err),
        "asymmetry_area_ratio": (out["tbp_lv_symm_2axis"] * out["tbp_lv_areaMM2"])
        / (out["tbp_lv_perimeterMM"] + err),
        "color_variation_intensity": out["tbp_lv_color_std_mean"]
        * out["tbp_lv_radial_color_std_max"],
        "color_contrast_ratio": out["lesion_color_difference"]
        / (out["tbp_lv_color_std_mean"] + err),
        "shape_irregularity": out["tbp_lv_perimeterMM"]
        / (2.0 * np.sqrt(np.pi * out["tbp_lv_areaMM2"] + err)),
        "border_density": out["tbp_lv_norm_border"] / (out["tbp_lv_areaMM2"] + err),
        "size_age_ratio": out["tbp_lv_areaMM2"] / (out["age_approx"] + err),
        "area_diameter_ratio": out["tbp_lv_areaMM2"]
        / (out["clin_size_long_diam_mm"] + err),
        "position_norm_3d": out["position_distance_3d"] / (out["tbp_lv_areaMM2"] + err),
        "position_angle_3d_xz": np.arctan2(out["tbp_lv_z"], out["tbp_lv_x"]),
        "position_angle_3d_yz": np.arctan2(out["tbp_lv_z"], out["tbp_lv_y"]),
        "lab_chroma": np.sqrt(np.square(out["tbp_lv_A"]) + np.square(out["tbp_lv_B"])),
        "lab_hue": np.arctan2(out["tbp_lv_B"], out["tbp_lv_A"]),
        "texture_contrast": out["tbp_lv_stdL"] / (out["tbp_lv_stdLExt"] + err),
        "texture_uniformity": 1.0 / (out["tbp_lv_color_std_mean"] + err),
        "color_difference_AB": np.abs(out["tbp_lv_deltaA"]) + np.abs(out["tbp_lv_deltaB"]),
        "color_difference_total": np.abs(out["tbp_lv_deltaA"])
        + np.abs(out["tbp_lv_deltaB"])
        + np.abs(out["tbp_lv_deltaL"]),
    }
    additional_features["color_change_ratio"] = additional_features[
        "color_difference_total"
    ] / (np.abs(out["tbp_lv_deltaLBnorm"]) + err)
    additional_features["lesion_complexity_index"] = (
        out["shape_complexity_index"] * out["color_variance_ratio"]
    )
    additional_features["age_adjusted_lesion_size"] = out["log_lesion_area"] / (
        out["age_approx"] + err
    )
    additional_features["color_uniformity_index"] = 1.0 / (out["color_uniformity"] + err)
    additional_features["border_sharpness_index"] = out["tbp_lv_norm_border"] / (
        out["tbp_lv_color_std_mean"] + err
    )
    additional_features["relative_depth_3d"] = np.abs(out["tbp_lv_z"]) / (
        out["position_distance_3d"] + err
    )
    additional_features["lesion_aspect_ratio"] = out["clin_size_long_diam_mm"] / (
        out["tbp_lv_minorAxisMM"] + err
    )
    additional_features["color_contrast_variation"] = (
        out["color_contrast_index"] * out["tbp_lv_color_std_mean"]
    )
    additional_features["symmetry_color_index"] = (
        out["tbp_lv_symm_2axis"] * out["tbp_lv_norm_color"]
    )
    additional_features["age_adjusted_complexity"] = additional_features[
        "lesion_complexity_index"
    ] / (out["age_approx"] + err)
    additional_features["color_gradient_intensity"] = additional_features[
        "color_difference_total"
    ] / (out["tbp_lv_perimeterMM"] + err)
    additional_features["lesion_density_index"] = out["lesion_color_difference"] / (
        out["tbp_lv_areaMM2"] + err
    )
    additional_features["color_diversity_index"] = (
        out["color_range"] * out["color_variance_ratio"]
    )
    additional_features["border_irregularity_size_ratio"] = out["border_complexity"] / (
        out["clin_size_long_diam_mm"] + err
    )
    additional_features["estimated_growth_rate"] = out["tbp_lv_areaMM2"] / (
        out["age_approx"] + err
    )
    additional_features["color_asymmetry_ratio"] = out["tbp_lv_radial_color_std_max"] / (
        out["tbp_lv_symm_2axis"] + err
    )
    additional_features["sphericity_3d"] = out["volume_approximation_3d"] / (
        np.power(out["tbp_lv_areaMM2"] + err, 1.5)
    )
    additional_features["relative_position_x"] = out["tbp_lv_x"] / (
        out["position_distance_3d"] + err
    )
    additional_features["relative_position_y"] = out["tbp_lv_y"] / (
        out["position_distance_3d"] + err
    )
    additional_features["malignancy_composite_index"] = (
        out["lesion_severity_index"]
        + additional_features["lesion_complexity_index"]
        + out["lesion_visibility_score"]
    ) / 3.0
    out = pd.concat(
        [out, pd.DataFrame(additional_features, index=out.index)],
        axis=1,
        copy=False,
    )

    if group_col in out.columns:
        out[group_col] = (
            out[group_col].astype("object").fillna("UNK").astype(str).astype("category")
        )

    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].astype("object").fillna("UNK").astype(str).astype("category")

    patient_grouped = out.groupby(group_col, observed=False, sort=False)
    patient_variation_features = {
        "relative_lesion_size": out["clin_size_long_diam_mm"]
        / (patient_grouped["clin_size_long_diam_mm"].transform("mean") + err),
        "lesion_complexity_index_patient_variation": out["lesion_complexity_index"]
        - patient_grouped["lesion_complexity_index"].transform("mean"),
        "color_diversity_index_patient_variation": out["color_diversity_index"]
        - patient_grouped["color_diversity_index"].transform("mean"),
        "border_irregularity_size_ratio_patient_variation": out[
            "border_irregularity_size_ratio"
        ]
        - patient_grouped["border_irregularity_size_ratio"].transform("mean"),
    }
    out = pd.concat(
        [out, pd.DataFrame(patient_variation_features, index=out.index)],
        axis=1,
        copy=False,
    )
    patient_grouped = out.groupby(group_col, observed=False, sort=False)

    generated_features: dict[str, pd.Series] = {}
    source_cols = [
        col for col in (BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS) if col in out.columns
    ]

    for col in source_cols:
        group_mean = patient_grouped[col].transform("mean")
        group_std = patient_grouped[col].transform("std")
        generated_features[f"{col}_patient_norm"] = (out[col] - group_mean) / (
            group_std + err
        )

    if use_contextual_group_norm_features:
        _add_contextual_group_norm_features(
            out=out,
            generated_features=generated_features,
            group_col=group_col,
            source_cols=source_cols,
            err=err,
        )

    if use_area_aggregation_features and "tbp_lv_areaMM2" in out.columns:
        generated_features["tbp_lv_areaMM2_patient"] = patient_grouped[
            "tbp_lv_areaMM2"
        ].transform("sum")
        if "anatom_site_general" in out.columns:
            generated_features["tbp_lv_areaMM2_bp"] = out.groupby(
                [group_col, "anatom_site_general"], observed=False, sort=False
            )["tbp_lv_areaMM2"].transform("sum")

    if use_anatomical_zscore_features and "anatom_site_general" in out.columns:
        site_grouped = out.groupby("anatom_site_general", observed=False, sort=False)
        for col in ["tbp_lv_areaMM2", "tbp_lv_norm_border", "tbp_lv_norm_color"]:
            if col not in out.columns:
                continue
            site_mean = site_grouped[col].transform("mean")
            site_std = site_grouped[col].transform("std")
            generated_features[f"{col}_anatomical_zscore"] = (out[col] - site_mean) / (
                site_std + err
            )

    if use_ugly_duckling_features:
        _add_ugly_duckling_features(
            out=out,
            generated_features=generated_features,
            group_col=group_col,
            source_cols=source_cols,
            err=err,
        )

    if use_patient_lof_features:
        lof_feature_cols = [
            col
            for col in [
                "tbp_lv_areaMM2",
                "tbp_lv_perimeterMM",
                "tbp_lv_norm_border",
                "tbp_lv_norm_color",
                "tbp_lv_H",
                "tbp_lv_deltaLBnorm",
                "lesion_color_difference",
                "lesion_complexity_index",
                "color_diversity_index",
                "position_distance_3d",
            ]
            if col in out.columns
        ]
        if len(lof_feature_cols) >= 3:
            generated_features["patient_lof_score"] = _compute_patient_lof_feature(
                df=out,
                group_col=group_col,
                feature_cols=lof_feature_cols,
                min_group_size=patient_lof_min_group_size,
            )

    for cat_col in categorical_cols:
        if cat_col not in out.columns:
            continue
        grouped = out.groupby([group_col, cat_col], observed=False, sort=False)
        for num_col in BASE_NUMERIC_COLS:
            if num_col not in out.columns:
                continue
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
    use_contextual_group_norm_features: bool = True,
    use_ugly_duckling_features: bool = True,
    use_anatomical_zscore_features: bool = True,
    use_area_aggregation_features: bool = True,
    use_patient_lof_features: bool = False,
    patient_lof_min_group_size: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_enhanced = _add_engineered_columns(
        df=train_df,
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
        use_contextual_group_norm_features=use_contextual_group_norm_features,
        use_ugly_duckling_features=use_ugly_duckling_features,
        use_anatomical_zscore_features=use_anatomical_zscore_features,
        use_area_aggregation_features=use_area_aggregation_features,
        use_patient_lof_features=use_patient_lof_features,
        patient_lof_min_group_size=patient_lof_min_group_size,
    )
    test_enhanced = _add_engineered_columns(
        df=test_df,
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
        use_contextual_group_norm_features=use_contextual_group_norm_features,
        use_ugly_duckling_features=use_ugly_duckling_features,
        use_anatomical_zscore_features=use_anatomical_zscore_features,
        use_area_aggregation_features=use_area_aggregation_features,
        use_patient_lof_features=use_patient_lof_features,
        patient_lof_min_group_size=patient_lof_min_group_size,
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


def _absolute_feature_target_corr(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
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
        col for col, keep in zip(numeric_frame.columns, keep_mask.tolist()) if bool(keep)
    ]


@dataclass
class FoldTabularPreprocessor:
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
        self.numeric_cols = list(dict.fromkeys(self.numeric_cols))
        self.active_numeric_cols = list(self.numeric_cols)
        self.numeric_medians: dict[str, float] = {}
        self.scaler = StandardScaler()
        self.encoder = _build_one_hot_encoder()
        self.quantile_transformer: QuantileTransformer | None = None
        self.is_fitted = False

    def _transform_base(self, df: pd.DataFrame) -> np.ndarray:
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
                if selected_numeric and len(selected_numeric) < len(self.active_numeric_cols):
                    self.active_numeric_cols = selected_numeric
                    numeric_frame = numeric_frame[self.active_numeric_cols]
                    print(
                        "Null-importance filter kept "
                        f"{len(self.active_numeric_cols)}/{len(self.numeric_cols)} "
                        "numeric features."
                    )

            self.numeric_medians = {
                col: float(numeric_frame[col].median()) for col in self.active_numeric_cols
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
        if not self.is_fitted:
            raise RuntimeError(
                "FoldTabularPreprocessor must be fitted before transform."
            )

        base_matrix = self._transform_base(df)
        if self.use_quantile_transform and self.quantile_transformer is not None:
            return self.quantile_transformer.transform(base_matrix).astype(np.float32)
        return base_matrix
