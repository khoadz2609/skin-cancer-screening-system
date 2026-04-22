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
    "area_to_perimeter_ratio",
    "lesion_visibility_score",
    "symmetry_border_consistency",
    "consistency_symmetry_border",
    "color_consistency",
    "consistency_color",
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
    "overall_color_difference",
    "symmetry_perimeter_interaction",
    "comprehensive_lesion_index",
    "color_variance_ratio",
    "border_color_interaction",
    "border_color_interaction_2",
    "size_color_contrast_ratio",
    "age_normalized_nevi_confidence",
    "age_normalized_nevi_confidence_2",
    "color_asymmetry_index",
    "volume_approximation_3d",
    "color_range",
    "shape_color_consistency",
    "border_length_ratio",
    "age_size_symmetry_index",
    "index_age_size_symmetry",
]

DEFAULT_CATEGORICAL_COLS = [
    "sex",
    "anatom_site_general",
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
    "attribution",
]

DEFAULT_NUMERIC_COLS = (
    list(BASE_NUMERIC_COLS)
    + list(ENGINEERED_NUMERIC_COLS)
    + [f"{col}_patient_norm" for col in (BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS)]
    + ["count_per_patient"]
    + [f"{col}_count" for col in DEFAULT_CATEGORICAL_COLS]
    + [
        f"{num_col}_{cat_col}"
        for num_col, cat_col in itertools.product(
            BASE_NUMERIC_COLS, DEFAULT_CATEGORICAL_COLS
        )
    ]
)


@dataclass(frozen=True)
class MidFusionLiteConfig:
    seed: int = 42
    preset_name: str = "quick"

    id_col: str = "isic_id"
    target_col: str = "target"
    group_col: str = "patient_id"

    n_splits: int = 3
    fold_indices: Optional[list[int]] = None

    image_size: int = 160
    train_batch_size: int = 32
    valid_batch_size: int = 64
    epochs: int = 6
    patience: int = 2
    num_workers: int = 2

    lr: float = 3e-4
    weight_decay: float = 1e-4

    negative_ratio: int = 6

    backbone_name: str = "efficientnet_b0"
    use_feature_engineering: bool = True
    use_tabm_tabular: bool = True
    use_quantile_transform: bool = True
    quantile_noise_std: float = 1e-5
    tab_hidden_dim: int = 128
    tabm_arch_type: str = "tabm"
    tabm_k: int = 16
    tabm_n_blocks: int = 3
    tabm_d_block: int = 256
    fusion_hidden_dim: int = 256
    tab_dropout: float = 0.1
    fusion_dropout: float = 0.2
    train_image_backbone: bool = False
    use_pretrained_backbone: bool = False

    use_amp: bool = True
    use_data_parallel: bool = True
    data_parallel_device_ids: Optional[list[int]] = None
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
            Path("tests/test-metadata.csv"),
            Path("../tests/test-metadata.csv"),
        ]
    )
    sample_submission = _first_existing(
        [
            Path("/kaggle/input/isic-2024-challenge/sample_submission.csv"),
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
    normalized = name.strip().lower()

    if normalized == "smoke":
        return MidFusionLiteConfig(
            preset_name="smoke",
            n_splits=2,
            fold_indices=[0],
            image_size=128,
            train_batch_size=32,
            valid_batch_size=64,
            epochs=2,
            patience=1,
            lr=8e-4,
            weight_decay=1e-4,
            negative_ratio=3,
            backbone_name="efficientnet_b0",
            tab_hidden_dim=96,
            fusion_hidden_dim=128,
            train_image_backbone=False,
            use_pretrained_backbone=False,
            max_train_steps_per_epoch=80,
            max_valid_steps=40,
            output_dir="./mid_fusion_lite_outputs/smoke",
        )

    if normalized == "full_lite":
        return MidFusionLiteConfig(
            preset_name="full_lite",
            n_splits=5,
            fold_indices=None,
            image_size=192,
            train_batch_size=32,
            valid_batch_size=64,
            epochs=10,
            patience=3,
            lr=3e-4,
            weight_decay=2e-4,
            negative_ratio=10,
            backbone_name="efficientnet_b0",
            tab_hidden_dim=192,
            fusion_hidden_dim=320,
            tab_dropout=0.15,
            fusion_dropout=0.25,
            train_image_backbone=True,
            use_pretrained_backbone=False,
            max_train_steps_per_epoch=None,
            max_valid_steps=None,
            output_dir="./mid_fusion_lite_outputs/full_lite",
        )

    if normalized != "quick":
        raise ValueError("Preset must be one of: smoke, quick, full_lite")

    return MidFusionLiteConfig(
        preset_name="quick",
        n_splits=3,
        fold_indices=None,
        image_size=160,
        train_batch_size=32,
        valid_batch_size=64,
        epochs=5,
        patience=2,
        lr=5e-4,
        weight_decay=1e-4,
        negative_ratio=6,
        backbone_name="efficientnet_b0",
        tab_hidden_dim=128,
        fusion_hidden_dim=192,
        tab_dropout=0.1,
        fusion_dropout=0.2,
        train_image_backbone=False,
        use_pretrained_backbone=False,
        max_train_steps_per_epoch=180,
        max_valid_steps=80,
        output_dir="./mid_fusion_lite_outputs/quick",
    )
