from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

BASE_NUMERIC_COLS = [
    "age_approx",
    "clin_size_long_diam_mm",
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
]

ENGINEERED_NUMERIC_COLS = [
    "lesion_size_ratio",
    "lesion_shape_index",
    "hue_contrast",
    "luminance_contrast",
    "lesion_color_difference",
    "border_complexity",
    "color_uniformity",
    "position_distance_3d",
    "perimeter_to_area_ratio",
    # Removed: "area_to_perimeter_ratio" (inverse of perimeter_to_area_ratio)
    "lesion_visibility_score",
    "symmetry_border_consistency",
    # Removed: "consistency_symmetry_border" (duplicate of symmetry_border_consistency)
    "color_consistency",
    # Removed: "consistency_color" (duplicate of color_consistency)
    "size_age_interaction",
    "hue_color_std_interaction",
    "lesion_severity_index",
    "shape_complexity_index",
    "color_contrast_index",
    "log_lesion_area",
    "normalized_lesion_size",
    "mean_hue_difference",
    "std_dev_contrast",
    "color_shape_composite_index",
    "lesion_orientation_3d",
    # Removed: "overall_color_difference" (redundant with color_difference_total)
    "symmetry_perimeter_interaction",
    "comprehensive_lesion_index",
    "color_variance_ratio",
    "border_color_interaction",
    # Removed: "border_color_interaction_2" (redundant logic with border_color_interaction)
    "size_color_contrast_ratio",
    "age_normalized_nevi_confidence",
    # Removed: "age_normalized_nevi_confidence_2" (invalid combination of size + age)
    "color_asymmetry_index",
    "volume_approximation_3d",
    "color_range",
    "shape_color_consistency",
    "border_length_ratio",
    "age_size_symmetry_index",
    # Removed: "index_age_size_symmetry" (duplicate 3-way interaction with age_size_symmetry_index)
    "asymmetry_ratio",
    "asymmetry_area_ratio",
    "color_variation_intensity",
    "color_contrast_ratio",
    "shape_irregularity",
    "border_density",
    "size_age_ratio",
    "area_diameter_ratio",
    "position_norm_3d",
    # Removed: "position_angle_3d_xz" (redundant 3D angle)
    # Removed: "position_angle_3d_yz" (redundant 3D angle)
    "lab_chroma",
    "lab_hue",
    "texture_contrast",
    "texture_uniformity",
    "color_difference_AB",
    "color_difference_total",
    "color_change_ratio",
    "lesion_complexity_index",
    "age_adjusted_lesion_size",
    "color_uniformity_index",
    "border_sharpness_index",
    # Removed: "relative_depth_3d" (unnecessary 3D depth)
    "lesion_aspect_ratio",
    "color_contrast_variation",
    "symmetry_color_index",
    "age_adjusted_complexity",
    "relative_lesion_size",
    "color_gradient_intensity",
    "lesion_density_index",
    "color_diversity_index",
    "border_irregularity_size_ratio",
    "estimated_growth_rate",
    "color_asymmetry_ratio",
    # Removed: "sphericity_3d" (lesion is 2D on skin, not truly 3D)
    "relative_position_x",
    "relative_position_y",
    "malignancy_composite_index",
    "lesion_complexity_index_patient_variation",
    "color_diversity_index_patient_variation",
    "border_irregularity_size_ratio_patient_variation",
    "tbp_lv_areaMM2_anatomical_zscore",
    "tbp_lv_norm_border_anatomical_zscore",
    "tbp_lv_norm_color_anatomical_zscore",
    "tbp_lv_areaMM2_patient",
    "tbp_lv_areaMM2_bp",
    "patient_lof_score",
]

DEFAULT_CATEGORICAL_COLS = [
    "sex",
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
]

CONTEXT_NORM_SUFFIXES = [
    "patient_norm",  # Z-score normalized by patient
    "patient_site_general_norm",  # Z-score normalized by anatomical site
]

UGLY_DUCKLING_SOURCE_COLS = list(BASE_NUMERIC_COLS) + list(ENGINEERED_NUMERIC_COLS)
UGLY_DUCKLING_SUMMARY_COLS = [
    "ud_count_location",
    "ud_max_severity_location",
    "ud_consistency_location",
]


def _deduplicate_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


CONTEXT_NUMERIC_COLS = [
    f"{col}_{suffix}"
    for suffix in CONTEXT_NORM_SUFFIXES
    for col in (BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS)
]

UGLY_DUCKLING_NUMERIC_COLS = [f"ud_loc_{col}" for col in UGLY_DUCKLING_SOURCE_COLS] + [
    f"ud_percentile_{col}" for col in UGLY_DUCKLING_SOURCE_COLS
]

CATEGORICAL_COUNT_COLS = [f"{col}_count" for col in DEFAULT_CATEGORICAL_COLS]

NUM_CAT_INTERACTION_COLS = [
    f"{num_col}_{cat_col}"
    for num_col, cat_col in itertools.product(
        BASE_NUMERIC_COLS, DEFAULT_CATEGORICAL_COLS
    )
]

DEFAULT_NUMERIC_COLS = _deduplicate_keep_order(
    list(BASE_NUMERIC_COLS)
    + list(ENGINEERED_NUMERIC_COLS)
    + CONTEXT_NUMERIC_COLS
    + UGLY_DUCKLING_NUMERIC_COLS
    + UGLY_DUCKLING_SUMMARY_COLS
    + ["count_per_patient"]
    + CATEGORICAL_COUNT_COLS
    + NUM_CAT_INTERACTION_COLS
)


def build_ablation_columns(
    use_engineered_abcd: bool = True,
    use_patient_norm: bool = True,
    use_ugly_duckling_adv: bool = True,
    use_categorical_context: bool = True,
) -> tuple[list[str], list[str]]:
    """Canonical numeric/categorical candidate lists for the ablation notebooks.

    This is the SINGLE SOURCE OF TRUTH for the column set used by
    ``only_mode/tabular_only_standard.ipynb`` and it MUST be reused verbatim by any
    downstream stage (e.g. Stage-3 fusion) that loads a TabM checkpoint trained here.

    Why order matters: TabM's Piecewise-Linear Embeddings bind weights to numeric
    columns *by position*. A checkpoint trained on this exact ordered list will load
    silently-wrong (weights mapped to the wrong column) — not just crash on a shape
    mismatch — if the stage that consumes it rebuilds the columns in a different order.
    So the ordering below intentionally mirrors the notebook's inline construction
    exactly; do not "fix" it to match DEFAULT_NUMERIC_COLS.

    Returns (numeric_candidates, categorical_candidates) as a SUPERSET; the runner's
    ``select_available_columns`` still drops any candidate absent after FE.
    """
    src = list(BASE_NUMERIC_COLS) + list(ENGINEERED_NUMERIC_COLS)
    anatomical_zscore = [c for c in ENGINEERED_NUMERIC_COLS if c.endswith("_anatomical_zscore")]
    patient_variation = [c for c in ENGINEERED_NUMERIC_COLS if c.endswith("_patient_variation")]
    area_agg = [c for c in ["tbp_lv_areaMM2_patient", "tbp_lv_areaMM2_bp"] if c in ENGINEERED_NUMERIC_COLS]
    lof = [c for c in ["patient_lof_score"] if c in ENGINEERED_NUMERIC_COLS]
    non_abcd = set(anatomical_zscore) | set(patient_variation) | set(area_agg) | set(lof)

    num = list(BASE_NUMERIC_COLS)
    if use_engineered_abcd:
        num += [c for c in ENGINEERED_NUMERIC_COLS if c not in non_abcd]
    if use_patient_norm:
        num += [f"{c}_patient_norm" for c in src] + patient_variation
    if use_ugly_duckling_adv:
        num += (
            list(UGLY_DUCKLING_NUMERIC_COLS)
            + list(UGLY_DUCKLING_SUMMARY_COLS)
            + lof
            + anatomical_zscore
            + area_agg
        )
    if use_categorical_context:
        num += (
            list(CATEGORICAL_COUNT_COLS)
            + ["count_per_patient", "is_single_lesion_patient"]
            + [f"{c}_patient_site_general_norm" for c in src]
            + list(NUM_CAT_INTERACTION_COLS)
        )
        cat = list(DEFAULT_CATEGORICAL_COLS)
    else:
        cat = []
    return _deduplicate_keep_order(num), list(cat)


@dataclass(frozen=True)
class MidFusionLiteConfig:
    seed: int = 42
    preset_name: str = "quick"

    id_col: str = "isic_id"
    target_col: str = "target"
    group_col: str = "patient_id"

    n_splits: int = 3
    fold_indices: Optional[list[int]] = None
    fold_col: Optional[str] = "fold"

    image_size: int = 160
    train_batch_size: int = 32
    valid_batch_size: int = 64
    epochs: int = 6
    patience: int = 2
    num_workers: int = 0

    lr: float = 2.1e-4
    weight_decay: float = 1.32e-4

    negative_ratio: int = 6

    backbone_name: str = "efficientnet_b0"
    use_feature_engineering: bool = True
    use_contextual_group_norm_features: bool = True
    use_ugly_duckling_features: bool = True
    use_anatomical_zscore_features: bool = True
    use_area_aggregation_features: bool = True
    use_shades_of_gray: bool = True
    use_tabm_tabular: bool = True
    use_null_importance_filter: bool = False
    null_importance_repeats: int = 8
    null_importance_quantile: float = 0.95
    null_importance_min_kept: int = 64
    use_quantile_transform: bool = True
    quantile_noise_std: float = 1e-5
    tab_hidden_dim: int = 128
    tabm_arch_type: str = "tabm"
    tabm_k: int = 32
    tabm_n_blocks: int = 2
    tabm_d_block: int = 256
    fusion_hidden_dim: int = 256
    fusion_mode: str = "gmu"
    cross_attention_heads: int = 4
    cross_attention_dropout: float = 0.1
    tab_dropout: float = 0.3
    fusion_dropout: float = 0.2
    modality_dropout_prob: float = 0.15
    aux_loss_image_weight: float = 0.5
    aux_loss_tab_weight: float = 0.2
    modality_mode: str = "both"  # "both" | "image_only" | "tabular_only"
    train_image_backbone: bool = False
    download_pretrained_backbone: bool = False

    # Advanced Training Hyperparameters
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    gradient_clipping_norm: Optional[float] = 2.0
    label_smoothing: float = 0.01
    pos_weight: Optional[float] = None
    n_bins: int = 48
    d_embedding: int = 32

    use_amp: bool = True
    max_train_steps_per_epoch: Optional[int] = None
    max_valid_steps: Optional[int] = None
    show_predict_progress: bool = True
    predict_log_interval: int = 20

    save_checkpoints: bool = True
    output_dir: str = "./mid_fusion_lite_outputs"
    backbone_weights_path: Optional[str] = None

    train_metadata_path: Optional[str] = None
    test_metadata_path: Optional[str] = None
    sample_submission_path: Optional[str] = None
    train_hdf5_path: Optional[str] = None
    test_hdf5_path: Optional[str] = None

    use_staged_pretraining: bool = False
    # True  → 3-stage pipeline: pretrain tabular-only → pretrain image-only
    #          → freeze encoders & train fusion head → joint fine-tune all
    # False → end-to-end: initialise full fusion model and train all branches jointly from epoch 1
    stage1_epochs: int = 5
    stage2_epochs: int = 5
    stage2_train_backbone: bool = True  # Whether to unfreeze backbone in Stage 2
    stage3_freeze_epochs: int = 3
    stage3_unfreeze_epochs: int = 5
    # Learning rates for Stage 3A (frozen encoders, train fusion head only)
    # and Stage 3B (unfreeze encoders, joint fine-tune).
    # When None, stage3a_lr defaults to config.lr and stage3b_lr defaults to 0.1 * config.lr.
    stage3a_lr: Optional[float] = None
    stage3b_lr: Optional[float] = None
    # In Stage 3B the image backbone is fine-tuned at stage3b_lr, while the newer
    # tabular encoder + fusion head (which only trained briefly in Stage 3A) get a
    # higher lr = stage3b_lr * stage3b_head_lr_mult so they keep learning instead of
    # stalling at the backbone's tiny lr. Set to 1.0 to disable differential lr.
    stage3b_head_lr_mult: float = 5.0

    stage_to_run: str = "all"  # "all", "stage1", "stage2", "stage3"
    tabular_full_data: bool = False

    pretrained_tabular_dir: Optional[str] = None
    pretrained_image_dir: Optional[str] = None

    numeric_cols: list[str] = field(default_factory=lambda: list(DEFAULT_NUMERIC_COLS))
    categorical_cols: list[str] = field(
        default_factory=lambda: list(DEFAULT_CATEGORICAL_COLS)
    )

    def with_updates(self, **kwargs) -> "MidFusionLiteConfig":
        return replace(self, **kwargs)


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_data_paths(config: MidFusionLiteConfig) -> MidFusionLiteConfig:
    if (
        config.train_metadata_path
        and config.test_metadata_path
        and config.sample_submission_path
        and config.train_hdf5_path
        and config.test_hdf5_path
    ):
        return config

    train_metadata = _first_existing(
        [
            Path("/kaggle/input/isic-2024-challenge/train-metadata.csv"),
            Path("data/train-metadata.csv"),
            Path("../data/train-metadata.csv"),
        ]
    )
    test_metadata = _first_existing(
        [
            Path("/kaggle/input/isic-2024-challenge/test-metadata.csv"),
            Path("data/test-metadata.csv"),
            Path("../data/test-metadata.csv"),
            Path("tests/test-metadata.csv"),
            Path("../tests/test-metadata.csv"),
        ]
    )
    sample_submission = _first_existing(
        [
            Path("/kaggle/input/isic-2024-challenge/sample_submission.csv"),
            Path("data/sample_submission.csv"),
            Path("../data/sample_submission.csv"),
            Path("tests/sample_submission.csv"),
            Path("../tests/sample_submission.csv"),
        ]
    )
    train_hdf5 = _first_existing(
        [
            Path("/kaggle/input/isic-2024-challenge/train-image.hdf5"),
            Path("data/train-image.hdf5"),
            Path("../data/train-image.hdf5"),
        ]
    )
    test_hdf5 = _first_existing(
        [
            Path("/kaggle/input/isic-2024-challenge/test-image.hdf5"),
            Path("data/test-image.hdf5"),
            Path("../data/test-image.hdf5"),
            Path("tests/test-image.hdf5"),
            Path("../tests/test-image.hdf5"),
        ]
    )

    missing = []
    if train_metadata is None:
        missing.append("train-metadata.csv")
    if test_metadata is None:
        missing.append("test-metadata.csv")
    if sample_submission is None:
        missing.append("sample_submission.csv")
    if train_hdf5 is None:
        missing.append("train-image.hdf5")
    if test_hdf5 is None:
        missing.append("test-image.hdf5")

    if missing:
        raise FileNotFoundError(
            "Could not resolve required input files: " + ", ".join(missing)
        )

    return config.with_updates(
        train_metadata_path=str(train_metadata),
        test_metadata_path=str(test_metadata),
        sample_submission_path=str(sample_submission),
        train_hdf5_path=str(train_hdf5),
        test_hdf5_path=str(test_hdf5),
    )


def get_preset_config(name: str = "quick") -> MidFusionLiteConfig:
    normalized = name.lower().strip()

    if normalized == "smoke":
        return MidFusionLiteConfig(
            preset_name="smoke",
            n_splits=2,
            fold_indices=[0],
            image_size=64,
            train_batch_size=8,
            valid_batch_size=16,
            epochs=2,
            warmup_epochs=0,
            stage2_epochs=1,
            stage3_freeze_epochs=1,
            stage3_unfreeze_epochs=1,
            output_dir="./mid_fusion_lite_outputs/smoke",
        )

    if normalized == "kaggle_top1_fusion":
        num, cat = build_ablation_columns(True, True, True, True)
        return MidFusionLiteConfig(
            preset_name="kaggle_top1_fusion",
            numeric_cols=num,
            categorical_cols=cat,
            n_splits=5,
            fold_indices=None,
            image_size=224,
            train_batch_size=48,
            valid_batch_size=64,
            epochs=12,
            warmup_epochs=2,
            patience=10,
            lr=2.1e-4,
            weight_decay=1.32e-4,
            negative_ratio=5,
            backbone_name="efficientnet_b0",
            tab_hidden_dim=256,
            fusion_hidden_dim=256,
            fusion_mode="cross_attention",
            tabm_k=32,
            tabm_n_blocks=2,
            tabm_d_block=256,
            n_bins=48,
            d_embedding=32,
            tab_dropout=0.30,
            fusion_dropout=0.25,
            train_image_backbone=True,
            download_pretrained_backbone=False,
            max_train_steps_per_epoch=None,
            max_valid_steps=None,
            use_staged_pretraining=True,
            stage1_epochs=5,
            stage2_epochs=5,
            stage2_train_backbone=False,
            stage3_freeze_epochs=3,
            stage3_unfreeze_epochs=5,
            output_dir="./mid_fusion_lite_outputs/kaggle_top1_fusion",
        )

    if normalized == "full_lite":
        return MidFusionLiteConfig(
            preset_name="full_lite",
            n_splits=5,
            fold_indices=None,
            image_size=192,
            train_batch_size=48,
            valid_batch_size=64,
            epochs=10,
            warmup_epochs=2,
            patience=10,
            lr=2.1e-4,
            weight_decay=1.32e-4,
            negative_ratio=5,
            backbone_name="efficientnet_b0",
            tab_hidden_dim=128,
            fusion_hidden_dim=320,
            tabm_k=32,
            tabm_n_blocks=2,
            tabm_d_block=256,
            n_bins=48,
            d_embedding=32,
            tab_dropout=0.30,
            fusion_dropout=0.25,
            train_image_backbone=True,
            download_pretrained_backbone=False,
            max_train_steps_per_epoch=None,
            max_valid_steps=None,
            use_staged_pretraining=True,
            stage1_epochs=5,
            stage2_epochs=5,
            stage2_train_backbone=False,  # Freeze backbone in Stage 2 (pretrained weights only)
            stage3_freeze_epochs=3,
            stage3_unfreeze_epochs=5,
            output_dir="./mid_fusion_lite_outputs/full_lite",
        )

    if normalized == "image_only_train":
        return MidFusionLiteConfig(
            preset_name="image_only_train",
            n_splits=5,
            image_size=224,
            backbone_name="efficientnet_b0",
            modality_mode="image_only",
            output_dir="./mid_fusion_lite_outputs/image_only_train",
        )

    if normalized != "quick":
        raise ValueError("Preset must be one of: smoke, quick, full_lite, image_only_train")

    return MidFusionLiteConfig(
        preset_name="quick",
        n_splits=3,
        fold_indices=None,
        image_size=160,
        train_batch_size=32,
        valid_batch_size=64,
        epochs=5,
        warmup_epochs=1,
        patience=2,
        lr=3e-4,  # Reduced from 5e-4: match full_lite for consistency
        weight_decay=1e-4,
        negative_ratio=6,
        stage1_epochs=3,
        stage2_epochs=3,
        stage3_freeze_epochs=2,
        stage3_unfreeze_epochs=3,
        backbone_name="efficientnet_b0",
        tab_hidden_dim=128,
        fusion_hidden_dim=192,
        tab_dropout=0.1,
        fusion_dropout=0.2,
        train_image_backbone=False,
        download_pretrained_backbone=False,
        max_train_steps_per_epoch=180,
        max_valid_steps=80,
        output_dir="./mid_fusion_lite_outputs/quick",
    )
