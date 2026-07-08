"""Feature engineering for medical imaging tabular data (skin lesion analysis)."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

try:
    from .tabular_prep_01 import _check_required_columns, _prepare_base_numeric
except ImportError:
    from tabular_prep_01 import _check_required_columns, _prepare_base_numeric

try:
    from .config import BASE_NUMERIC_COLS, ENGINEERED_NUMERIC_COLS
except ImportError:
    from config import BASE_NUMERIC_COLS, ENGINEERED_NUMERIC_COLS


def _add_ugly_duckling_features(
    out: pd.DataFrame,
    generated_features: dict[str, pd.Series],
    group_col: str,
    source_cols: list[str],
    err: float,
) -> None:
    """Add 'ugly duckling' features for anomaly detection within patient cohorts.

    The 'ugly duckling' concept: lesions that look unusual compared to a patient's
    other lesions may indicate malignancy. Creates z-scores by location and adds
    aggregate anomaly indicators.

    Args:
        out: DataFrame with all data.
        generated_features: Dictionary to accumulate new features.
        group_col: Grouping column (e.g., 'patient_id').
        source_cols: Base columns to compute anomaly scores.
        err: Epsilon for numerical stability.
    """
    loc_col = None
    for candidate in [
        "tbp_lv_location",
        "tbp_lv_location_simple",
        "anatom_site_general",
    ]:
        if candidate in out.columns:
            loc_col = candidate
            break
    if loc_col is None:
        return

    patient_grouped = out.groupby(group_col, observed=False, sort=False)
    loc_grouped = out.groupby([group_col, loc_col], observed=False, sort=False)

    ud_loc_cols: list[str] = []
    for col in source_cols:
        loc_sum = loc_grouped[col].transform("sum").astype(np.float32, copy=False)
        loc_count = loc_grouped[col].transform("count").astype(np.float32, copy=False)
        loc_std = loc_grouped[col].transform("std").astype(np.float32, copy=False)

        # 1. Leave-One-Out (LOO) Mean calculation
        loo_mean = np.where(
            loc_count > 1,
            (loc_sum - out[col]) / (loc_count - 1),
            out[col]
        )

        # 2 & 3. Calculate distance with fallback to 0.0 for 1-lesion patients
        ud_loc_col = f"ud_loc_{col}"
        generated_features[ud_loc_col] = pd.Series(
            np.where(
                loc_count > 1,
                (out[col] - loo_mean) / (loc_std + err),
                0.0  # Fallback: strictly 0 (no deviation)
            ),
            index=out.index
        ).astype(np.float32, copy=False)
        ud_loc_cols.append(ud_loc_col)

    if not ud_loc_cols:
        return

    # ud_abs intrinsically represents the standardized distance for each feature
    ud_abs = pd.DataFrame(
        {col: generated_features[col].abs() for col in ud_loc_cols},
        index=out.index,
    )
    generated_features["ud_count_location"] = (ud_abs > 2.0).sum(axis=1).astype(np.float32)
    generated_features["ud_max_severity_location"] = ud_abs.max(axis=1).astype(np.float32)
    generated_features["ud_consistency_location"] = ud_abs.std(axis=1).fillna(0.0).astype(np.float32)
    
    # Add Euclidean distance of standardized features (Composite Score)
    generated_features["ud_composite_distance"] = np.sqrt(
        (ud_abs ** 2).sum(axis=1)
    ).astype(np.float32)


def _add_contextual_group_norm_features(
    out: pd.DataFrame,
    generated_features: dict[str, pd.Series],
    group_col: str,
    source_cols: list[str],
    err: float,
) -> None:
    """Add features normalized by anatomical context (location/site).

    Creates z-score normalized features grouped by patient AND anatomical location,
    capturing how features deviate within anatomical regions.

    Args:
        out: DataFrame with all data.
        generated_features: Dictionary to accumulate new features.
        group_col: Grouping column (e.g., 'patient_id').
        source_cols: Base columns to normalize.
        err: Epsilon for numerical stability.
    """
    context_groups = [
        ("patient_site_general_norm", "anatom_site_general"),
    ]

    for suffix, context_col in context_groups:
        if context_col not in out.columns:
            continue

        # Pre-compute group statistics for ALL columns at once (cache-friendly)
        grouped = out.groupby([group_col, context_col], observed=False, sort=False)

        # Transform using cached statistics (vectorized)
        for col in source_cols:
            group_mean = grouped[col].transform("mean").astype(np.float32, copy=False)
            group_std = grouped[col].transform("std").astype(np.float32, copy=False)
            generated_features[f"{col}_{suffix}"] = (
                (out[col] - group_mean) / (group_std + err)
            ).astype(np.float32, copy=False)


def _add_patient_lof_features(
    out: pd.DataFrame,
    generated_features: dict[str, pd.Series],
    group_col: str,
    train_stats: dict | None = None,
    train_stats_out: dict | None = None,
) -> None:
    """Add Local Outlier Factor (LOF) score for each lesion per patient cohort.

    The score indicates how much a lesion's visual/shape properties deviate from
    the patient's typical lesions (acting as a multi-dimensional ugly duckling indicator).
    """
    lof_features = [
        "clin_size_long_diam_mm",
        "tbp_lv_areaMM2",
        "tbp_lv_color_std_mean",
        "tbp_lv_symm_2axis",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
    ]
    # Filter only columns that exist in the dataframe
    lof_features = [col for col in lof_features if col in out.columns]
    if not lof_features:
        generated_features["patient_lof_score"] = pd.Series(1.0, index=out.index)
        return

    # Fill NaNs/Infs for security
    data_subset = out[[group_col] + lof_features].copy()
    for col in lof_features:
        data_subset[col] = data_subset[col].replace([np.inf, -np.inf], np.nan)
        data_subset[col] = data_subset[col].fillna(data_subset[col].median())

    # Standardize features using train_stats to avoid leakage and distribution shifts
    if (
        train_stats is not None
        and "lof_mean" in train_stats
        and "lof_std" in train_stats
    ):
        mean = train_stats["lof_mean"]
        std = train_stats["lof_std"]
    else:
        mean = data_subset[lof_features].mean(numeric_only=True)
        std = (
            data_subset[lof_features]
            .std(numeric_only=True)
            .replace(0.0, 1.0)
            .fillna(1.0)
        )
        if train_stats_out is not None:
            train_stats_out["lof_mean"] = mean
            train_stats_out["lof_std"] = std

    data_subset[lof_features] = (data_subset[lof_features] - mean) / std

    scores = np.ones(len(out), dtype=np.float32)

    # Group by patient
    grouped = data_subset.groupby(group_col, observed=False, sort=False)
    for _, group in grouped:
        n_samples = len(group)
        if n_samples >= 3:
            n_neighbors = min(20, n_samples - 1)
            if n_neighbors >= 2:
                try:
                    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
                    lof.fit(group[lof_features])
                    scores[group.index.to_numpy()] = -lof.negative_outlier_factor_
                except Exception:
                    pass

    generated_features["patient_lof_score"] = pd.Series(scores, index=out.index)


def _add_engineered_columns(
    df: pd.DataFrame,
    group_col: str,
    categorical_cols: list[str],
    err: float,
    use_contextual_group_norm_features: bool,
    use_ugly_duckling_features: bool,
    use_anatomical_zscore_features: bool,
    use_area_aggregation_features: bool,
    train_stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Main orchestrator for all feature engineering transformations.

    Generates 80+ derived features from base medical measurements including:
    - Geometric ratios and indices (lesion shape, border properties)
    - Color features (LAB space metrics, contrast, uniformity)
    - 3D spatial features (position, volume approximation)
    - Interaction features (size × age, color × symmetry)
    - Patient/cohort normalized features
    - Contextual (anatomical) normalized features
    - Anomaly detection features (ugly duckling)
    - Patient-specific aggregate features

    Args:
        df: Input DataFrame with base numeric and categorical columns.
        group_col: Column name for patient/group identifier.
        categorical_cols: Categorical column names for encoding.
        err: Epsilon for numerical stability (default: 1e-5).
        use_contextual_group_norm_features: Enable location-based normalization.
        use_ugly_duckling_features: Enable anomaly detection features.
        use_anatomical_zscore_features: Enable anatomical z-score features.
        use_area_aggregation_features: Enable area aggregation by patient/anatomical site.
        train_stats: Precomputed training statistics (if None, computes them).

    Returns:
        Tuple of (DataFrame with original + all engineered features, train_stats dictionary).
    """
    train_stats_out = {}
    if train_stats is None:
        train_stats = {}

    out, base_medians = _prepare_base_numeric(
        df=df,
        err=err,
        medians=train_stats.get("base_medians"),
    )
    train_stats_out["base_medians"] = base_medians
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
    out["lesion_visibility_score"] = (
        out["tbp_lv_deltaLBnorm"] + out["tbp_lv_norm_color"]
    )
    out["symmetry_border_consistency"] = (
        out["tbp_lv_symm_2axis"] * out["tbp_lv_norm_border"]
    )

    out["color_consistency"] = out["tbp_lv_stdL"] / (out["tbp_lv_Lext"] + err)
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
    out["size_color_contrast_ratio"] = out["clin_size_long_diam_mm"] / (
        out["tbp_lv_deltaLBnorm"] + err
    )
    out["age_normalized_nevi_confidence"] = out["tbp_lv_nevi_confidence"] / (
        out["age_approx"] + err
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
        "lab_chroma": np.sqrt(np.square(out["tbp_lv_A"]) + np.square(out["tbp_lv_B"])),
        "lab_hue": np.arctan2(out["tbp_lv_B"], out["tbp_lv_A"]),
        "texture_contrast": out["tbp_lv_stdL"] / (out["tbp_lv_stdLExt"] + err),
        "texture_uniformity": 1.0 / (out["tbp_lv_color_std_mean"] + err),
        "color_difference_AB": np.abs(out["tbp_lv_deltaA"])
        + np.abs(out["tbp_lv_deltaB"]),
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
    additional_features["color_uniformity_index"] = 1.0 / (
        out["color_uniformity"] + err
    )
    additional_features["border_sharpness_index"] = out["tbp_lv_norm_border"] / (
        out["tbp_lv_color_std_mean"] + err
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
    additional_features["color_asymmetry_ratio"] = out[
        "tbp_lv_radial_color_std_max"
    ] / (out["tbp_lv_symm_2axis"] + err)
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
            out[col] = (
                out[col].astype("object").fillna("UNK").astype(str).astype("category")
            )

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
    # Cast all existing numeric columns to float32 early to save memory before heavy group normalization
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].astype(np.float32, copy=False)
    gc.collect()

    patient_grouped = out.groupby(group_col, observed=False, sort=False)

    generated_features: dict[str, pd.Series] = {}
    source_cols = [
        col
        for col in (BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS)
        if col in out.columns
    ]

    for col in source_cols:
        group_mean = (
            patient_grouped[col].transform("mean").astype(np.float32, copy=False)
        )
        group_std = patient_grouped[col].transform("std").astype(np.float32, copy=False)
        generated_features[f"{col}_patient_norm"] = (
            (out[col] - group_mean) / (group_std + err)
        ).astype(np.float32, copy=False)

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
        site_stats = train_stats.get("site_stats")
        train_site_stats = {}
        for col in ["tbp_lv_areaMM2", "tbp_lv_norm_border", "tbp_lv_norm_color"]:
            if col not in out.columns:
                continue

            if site_stats is None:
                stats_df = out.groupby(
                    "anatom_site_general", observed=False, sort=False
                )[col].agg(["mean", "std"])
                train_site_stats[col] = stats_df
            else:
                stats_df = site_stats[col]

            # Map stats
            site_mean = (
                out["anatom_site_general"].map(stats_df["mean"]).astype(np.float32)
            )
            site_std = (
                out["anatom_site_general"].map(stats_df["std"]).astype(np.float32)
            )

            # Fallbacks
            site_mean = site_mean.fillna(out[col].mean())
            site_std = site_std.fillna(0.0)

            generated_features[f"{col}_anatomical_zscore"] = (out[col] - site_mean) / (
                site_std + err
            )

        train_stats_out["site_stats"] = (
            train_site_stats if site_stats is None else site_stats
        )

    if use_ugly_duckling_features:
        _add_ugly_duckling_features(
            out=out,
            generated_features=generated_features,
            group_col=group_col,
            source_cols=source_cols,
            err=err,
        )
        _add_patient_lof_features(
            out=out,
            generated_features=generated_features,
            group_col=group_col,
            train_stats=train_stats,
            train_stats_out=train_stats_out,
        )

    for cat_col in categorical_cols:
        if cat_col not in out.columns:
            continue
        grouped = out.groupby([group_col, cat_col], observed=False, sort=False)
        for num_col in BASE_NUMERIC_COLS:
            if num_col not in out.columns:
                continue
            group_mean = grouped[num_col].transform("mean").astype(np.float32, copy=False)
            group_std = grouped[num_col].transform("std").astype(np.float32, copy=False)
            generated_features[f"{num_col}_{cat_col}"] = ((out[num_col] - group_mean) / (
                group_std + err
            )).astype(np.float32, copy=False)

    for col in categorical_cols:
        if col not in out.columns:
            continue
        generated_features[f"{col}_count"] = out.groupby(
            [group_col, col], observed=False, sort=False
        )[col].transform("count")

    generated_features["count_per_patient"] = patient_grouped[group_col].transform(
        "count"
    )
    generated_features["is_single_lesion_patient"] = (
        generated_features["count_per_patient"] == 1
    ).astype(np.float32)

    if generated_features:
        # Explicitly cast all generated features to float32 before concatenating
        for k in list(generated_features.keys()):
            if np.issubdtype(generated_features[k].dtype, np.number):
                generated_features[k] = generated_features[k].astype(
                    np.float32, copy=False
                )

        out = pd.concat(
            [out, pd.DataFrame(generated_features, index=out.index)],
            axis=1,
            copy=False,
        )
        del generated_features
        gc.collect()

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    engineered_medians = train_stats.get("engineered_medians")
    if engineered_medians is None:
        engineered_medians = out[numeric_cols].median(numeric_only=True)
        train_stats_out["engineered_medians"] = engineered_medians
    else:
        train_stats_out["engineered_medians"] = engineered_medians

    common_cols = numeric_cols.intersection(engineered_medians.index)
    out[common_cols] = out[common_cols].fillna(engineered_medians[common_cols])

    remaining_cols = numeric_cols.difference(engineered_medians.index)
    if len(remaining_cols) > 0:
        out[remaining_cols] = out[remaining_cols].fillna(out[remaining_cols].median())

    out[numeric_cols] = out[numeric_cols].astype(np.float32, copy=False)

    return out, train_stats_out


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply complete feature engineering pipeline to train and test sets.

    Wraps _add_engineered_columns for both datasets with consistent parameters,
    ensuring train and test receive identical feature transformations.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        group_col: Column name for grouping (e.g., 'patient_id').
        categorical_cols: List of categorical column names.
        err: Epsilon for numerical stability (default: 1e-5).
        use_contextual_group_norm_features: Enable location-based normalization (default: True).
        use_ugly_duckling_features: Enable anomaly detection (default: True).
        use_anatomical_zscore_features: Enable anatomical z-score (default: True).
        use_area_aggregation_features: Enable area aggregation (default: True).

    Returns:
        Tuple of (train_enhanced, test_enhanced) DataFrames with all features.
    """
    train_enhanced, train_stats = _add_engineered_columns(
        df=train_df,
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
        use_contextual_group_norm_features=use_contextual_group_norm_features,
        use_ugly_duckling_features=use_ugly_duckling_features,
        use_anatomical_zscore_features=use_anatomical_zscore_features,
        use_area_aggregation_features=use_area_aggregation_features,
        train_stats=None,
    )
    test_enhanced, _ = _add_engineered_columns(
        df=test_df,
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
        use_contextual_group_norm_features=use_contextual_group_norm_features,
        use_ugly_duckling_features=use_ugly_duckling_features,
        use_anatomical_zscore_features=use_anatomical_zscore_features,
        use_area_aggregation_features=use_area_aggregation_features,
        train_stats=train_stats,
    )
    return train_enhanced, test_enhanced


def add_feature_engineering_folds(
    fold_train_df: pd.DataFrame,
    fold_valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    categorical_cols: list[str],
    err: float = 1e-5,
    use_contextual_group_norm_features: bool = True,
    use_ugly_duckling_features: bool = True,
    use_anatomical_zscore_features: bool = True,
    use_area_aggregation_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leak-safe feature engineering for a single CV fold.

    Unlike ``add_feature_engineering`` (which fits engineering statistics on the
    whole training set), this fits **only on the fold-train split** and then applies
    those frozen statistics to the fold-valid and test splits via ``train_stats``.

    This matters because some engineered features are *global* statistics — anatomical
    z-scores (``groupby("anatom_site_general")``), median imputation, and the LOF
    standardization — which, if computed over train+valid together, leak validation
    information into the features seen at validation time. Patient-grouped features
    (``*_patient_norm``, ugly-duckling, area aggregation) are already leak-free as long
    as folds are patient-disjoint, so they are unaffected either way.

    Args:
        fold_train_df: Training split for this fold (statistics are fit on this only).
        fold_valid_df: Validation split for this fold (transform with fold-train stats).
        test_df: Test set (transform with fold-train stats).
        group_col: Column name for grouping (e.g., 'patient_id').
        categorical_cols: List of categorical column names.
        err: Epsilon for numerical stability (default: 1e-5).
        use_contextual_group_norm_features: Enable location-based normalization.
        use_ugly_duckling_features: Enable anomaly detection.
        use_anatomical_zscore_features: Enable anatomical z-score.
        use_area_aggregation_features: Enable area aggregation.

    Returns:
        Tuple of (train_enhanced, valid_enhanced, test_enhanced) DataFrames.
    """
    kwargs = dict(
        group_col=group_col,
        categorical_cols=categorical_cols,
        err=err,
        use_contextual_group_norm_features=use_contextual_group_norm_features,
        use_ugly_duckling_features=use_ugly_duckling_features,
        use_anatomical_zscore_features=use_anatomical_zscore_features,
        use_area_aggregation_features=use_area_aggregation_features,
    )
    # Fit on fold-train ONLY; capture the frozen statistics.
    train_enhanced, train_stats = _add_engineered_columns(
        df=fold_train_df, train_stats=None, **kwargs
    )
    # Transform valid and test with the fold-train statistics (no leakage).
    valid_enhanced, _ = _add_engineered_columns(
        df=fold_valid_df, train_stats=train_stats, **kwargs
    )
    test_enhanced, _ = _add_engineered_columns(
        df=test_df, train_stats=train_stats, **kwargs
    )
    return train_enhanced, valid_enhanced, test_enhanced
