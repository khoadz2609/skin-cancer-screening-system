from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

try:
    from .config import (
        BASE_NUMERIC_COLS,
        MidFusionLiteConfig,
        get_preset_config,
        resolve_data_paths,
    )
    from .data import (
        HDF5ImageTabularDataset,
        get_image_transforms,
        make_balanced_sampler,
    )
    from .tabular_engineering_02 import add_feature_engineering
    from .model import MidFusionLiteModel
    from .tabular_preprocessor_04 import FoldTabularPreprocessor, select_available_columns
    from .train import fit_one_fold, partial_auc_score, predict_loader
except ImportError:
    from config import (
        BASE_NUMERIC_COLS,
        MidFusionLiteConfig,
        get_preset_config,
        resolve_data_paths,
    )
    from data import (
        HDF5ImageTabularDataset,
        get_image_transforms,
        make_balanced_sampler,
    )
    from tabular_engineering_02 import add_feature_engineering
    from model import MidFusionLiteModel
    from tabular_preprocessor_04 import FoldTabularPreprocessor, select_available_columns
    from train import fit_one_fold, partial_auc_score, predict_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Some Kaggle images can expose CUDA while shipping a torch build
    # that does not include kernels for the allocated GPU architecture.
    try:
        x = torch.randn(1, 3, 16, 16, device="cuda")
        w = torch.randn(4, 3, 3, 3, device="cuda")
        _ = F.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except Exception as exc:
        print("CUDA is available but not runtime-compatible with this torch build.")
        print(f"Falling back to CPU. Details: {exc}")
        return torch.device("cpu")


def _initialize_lazy_parameters(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    n_tab_features: int,
) -> None:
    # Initialize modules that still contain UninitializedParameter
    # (for example nn.LazyLinear).
    has_uninitialized = any(
        isinstance(param, torch.nn.parameter.UninitializedParameter)
        for param in model.parameters()
    )
    if not has_uninitialized:
        return

    batch_size = 2
    was_training = model.training
    model.eval()
    # inference_mode marks created tensors as inference tensors, which can
    # break autograd later when these parameters are used for training.
    with torch.no_grad():
        dummy_images = torch.zeros(
            batch_size,
            3,
            int(image_size),
            int(image_size),
            device=device,
        )
        dummy_tabular = torch.zeros(
            batch_size,
            int(n_tab_features),
            device=device,
        )
        _ = model(dummy_images, dummy_tabular)

    if was_training:
        model.train()


def _validate_columns(
    config: MidFusionLiteConfig, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    required_train_cols = {config.id_col, config.target_col, config.group_col}
    missing_train_cols = [
        col for col in required_train_cols if col not in train_df.columns
    ]
    if missing_train_cols:
        raise KeyError(
            "Missing required columns in train metadata: "
            + ", ".join(missing_train_cols)
        )

    if config.id_col not in test_df.columns:
        raise KeyError(f"Missing {config.id_col} in test metadata")


def _ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _read_csv_with_fallback(path: str, requested_cols: set[str]) -> pd.DataFrame:
    header_cols = list(pd.read_csv(path, nrows=0).columns)
    selected_cols = [col for col in header_cols if col in requested_cols]

    read_kwargs: dict[str, Any] = {"low_memory": True}
    if selected_cols:
        read_kwargs["usecols"] = selected_cols

    try:
        return pd.read_csv(path, **read_kwargs)
    except pd.errors.ParserError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(
            f"Parser out-of-memory while reading {path}. "
            "Retrying with python engine."
        )
        retry_kwargs = dict(read_kwargs)
        retry_kwargs["engine"] = "python"
        return pd.read_csv(path, **retry_kwargs)


def _filter_supported_config_updates(
    config: MidFusionLiteConfig,
    updates: dict[str, Any],
) -> dict[str, Any]:
    if not updates:
        return updates

    supported_keys = {field.name for field in dataclass_fields(type(config))}
    filtered = {k: v for k, v in updates.items() if k in supported_keys}
    dropped = sorted(set(updates) - set(filtered))
    if dropped:
        print("Ignoring unsupported config overrides: " + ", ".join(dropped))
    return filtered


def _apply_patient_post_processing(
    preds: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Normalize predictions patient-wise (using mean-subtraction).

    This centers each patient's predictions around 0, putting outliers
    above 0 and typical lesions below 0, aligning with the ugly duckling sign.
    """
    df = pd.DataFrame({"pred": preds, "group": groups})
    group_mean = df.groupby("group", observed=False)["pred"].transform("mean")
    processed = df["pred"] - group_mean
    return processed.to_numpy()


def run_mid_fusion_cv(config: MidFusionLiteConfig) -> dict[str, Any]:
    config = resolve_data_paths(config)

    if config.backbone_weights_path and not Path(config.backbone_weights_path).exists():
        print(
            "Backbone weights path does not exist on this machine. "
            "Ignoring backbone_weights_path and using default backbone init."
        )
        config = config.with_updates(backbone_weights_path=None)

    set_seed(config.seed)
    device = _get_device()

    output_dir = _ensure_output_dir(config.output_dir)

    print(f"Running preset: {config.preset_name}")
    print(f"Device: {device.type.upper()}")
    print(f"Output directory: {output_dir.resolve()}")

    requested_numeric_cols = set(config.numeric_cols)
    if bool(config.use_feature_engineering):
        requested_numeric_cols.update(BASE_NUMERIC_COLS)

    requested_train_cols = {
        config.id_col,
        config.target_col,
        config.group_col,
        "tbp_lv_nevi_confidence", # Ensure this is read for hard negative mining
        *config.categorical_cols,
        *requested_numeric_cols,
    }
    requested_test_cols = {
        config.id_col,
        *config.categorical_cols,
        *requested_numeric_cols,
    }
    if bool(config.use_feature_engineering):
        requested_test_cols.add(config.group_col)

    train_df = _read_csv_with_fallback(
        path=config.train_metadata_path,
        requested_cols=requested_train_cols,
    )
    test_df = _read_csv_with_fallback(
        path=config.test_metadata_path,
        requested_cols=requested_test_cols,
    )
    sample_submission = pd.read_csv(config.sample_submission_path)

    _validate_columns(config, train_df, test_df)

    if bool(config.use_feature_engineering):
        try:
            train_df, test_df = add_feature_engineering(
                train_df=train_df,
                test_df=test_df,
                group_col=config.group_col,
                categorical_cols=config.categorical_cols,
                use_ugly_duckling_features=bool(config.use_ugly_duckling_features),
            )
        except MemoryError:
            print(
                "MemoryError during feature engineering. "
                "Retrying without heavy engineered features."
            )
            gc.collect()
    else:
        print("Skipping heavy feature engineering (use_feature_engineering=False).")

    numeric_cols, categorical_cols = select_available_columns(
        train_df=train_df,
        test_df=test_df,
        numeric_candidates=config.numeric_cols,
        categorical_candidates=config.categorical_cols,
    )

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    y = train_df[config.target_col].astype(np.int64).to_numpy()
    groups = train_df[config.group_col].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    folds = list(splitter.split(train_df, y, groups))

    if config.fold_indices is None:
        selected_folds = list(range(config.n_splits))
    else:
        selected_folds = list(config.fold_indices)

    selected_folds_set = set(selected_folds)
    print(f"Selected folds: {sorted(selected_folds_set)}")

    transforms = get_image_transforms(config.image_size)

    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    oof_mask = np.zeros(len(train_df), dtype=bool)
    test_pred_sum = np.zeros(len(test_df), dtype=np.float32)

    fold_metrics: list[dict[str, Any]] = []
    n_test_ensembles = 0

    for fold_index, (train_idx, valid_idx) in enumerate(folds):
        if fold_index not in selected_folds_set:
            continue

        print("=" * 80)
        print(f"Fold {fold_index + 1}/{config.n_splits}")

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

        preprocessor = FoldTabularPreprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            use_quantile_transform=(
                bool(config.use_quantile_transform) and bool(config.use_tabm_tabular)
            ),
            quantile_noise_std=float(config.quantile_noise_std),
            quantile_random_state=config.seed + fold_index + 1,
            use_null_importance_filter=bool(config.use_null_importance_filter),
            null_importance_repeats=int(config.null_importance_repeats),
            null_importance_quantile=float(config.null_importance_quantile),
            null_importance_min_kept=int(config.null_importance_min_kept),
            target_col=config.target_col,
        )
        preprocessor.fit(fold_train_df)

        x_train = preprocessor.transform(fold_train_df)
        x_valid = preprocessor.transform(fold_valid_df)
        x_test = preprocessor.transform(test_df)

        y_train = fold_train_df[config.target_col].astype(np.float32).to_numpy()
        y_valid = fold_valid_df[config.target_col].astype(np.float32).to_numpy()

        train_dataset = HDF5ImageTabularDataset(
            ids=fold_train_df[config.id_col].astype(str).to_numpy(),
            tabular=x_train,
            hdf5_path=config.train_hdf5_path,
            targets=y_train,
            transforms=transforms["train"],
            use_shades_of_gray=bool(config.use_shades_of_gray),
        )
        valid_dataset = HDF5ImageTabularDataset(
            ids=fold_valid_df[config.id_col].astype(str).to_numpy(),
            tabular=x_valid,
            hdf5_path=config.train_hdf5_path,
            targets=y_valid,
            transforms=transforms["valid"],
            use_shades_of_gray=bool(config.use_shades_of_gray),
        )
        test_dataset = HDF5ImageTabularDataset(
            ids=test_df[config.id_col].astype(str).to_numpy(),
            tabular=x_test,
            hdf5_path=config.test_hdf5_path,
            targets=None,
            transforms=transforms["valid"],
            use_shades_of_gray=bool(config.use_shades_of_gray),
        )

        if "tbp_lv_nevi_confidence" in fold_train_df.columns:
            # Nevus confidence: 0 (least likely nevus) to 100 (most likely nevus).
            # Hard negatives are those with low nevus confidence (meaning they look like melanoma).
            nevi_conf = fold_train_df["tbp_lv_nevi_confidence"].fillna(50.0).to_numpy()
            # Add a base weight of 0.1 so easy negatives still have a chance of being sampled
            hard_negative_weights = 0.1 + 0.9 * (((100.0 - nevi_conf) / 100.0) ** 2.0)
        else:
            hard_negative_weights = None

        sampler = make_balanced_sampler(
            targets=y_train,
            negative_ratio=config.negative_ratio,
            seed=config.seed + fold_index,
            hard_negative_weights=hard_negative_weights,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.valid_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        model = MidFusionLiteModel(
            backbone_name=config.backbone_name,
            n_tab_features=x_train.shape[1],
            use_tabm_tabular=config.use_tabm_tabular,
            tab_hidden_dim=config.tab_hidden_dim,
            tabm_arch_type=config.tabm_arch_type,
            tabm_k=config.tabm_k,
            tabm_n_blocks=config.tabm_n_blocks,
            tabm_d_block=config.tabm_d_block,
            fusion_hidden_dim=config.fusion_hidden_dim,
            fusion_mode=config.fusion_mode,
            cross_attention_heads=config.cross_attention_heads,
            cross_attention_dropout=config.cross_attention_dropout,
            tab_dropout=config.tab_dropout,
            fusion_dropout=config.fusion_dropout,
            train_image_backbone=config.train_image_backbone,
            use_pretrained_backbone=config.use_pretrained_backbone,
            backbone_weights_path=config.backbone_weights_path,
            modality_dropout_prob=config.modality_dropout_prob,
            modality_mode=config.modality_mode,
        )
        model = model.to(device)
        _initialize_lazy_parameters(
            model=model,
            device=device,
            image_size=config.image_size,
            n_tab_features=x_train.shape[1],
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
            eta_min=config.lr * 0.1,
        )

        n_pos = max(float(y_train.sum()), 1.0)
        n_neg = max(float(len(y_train) - y_train.sum()), 1.0)
        # Restore double weighting: this is crucial for the partial AUC metric
        pos_weight = np.sqrt(n_neg / n_pos)

        best_state, history, best_valid_score = fit_one_fold(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=config.epochs,
            patience=config.patience,
            pos_weight=pos_weight,
            use_amp=(config.use_amp and device.type == "cuda"),
            max_train_steps_per_epoch=config.max_train_steps_per_epoch,
            max_valid_steps=config.max_valid_steps,
            aux_loss_image_weight=config.aux_loss_image_weight,
            aux_loss_tab_weight=config.aux_loss_tab_weight,
        )

        model.load_state_dict(best_state)
        predict_log_interval = max(1, int(config.predict_log_interval))
        if bool(config.show_predict_progress):
            print(
                f"Running fold {fold_index + 1} inference "
                f"(log every {predict_log_interval} steps)..."
            )

        valid_pred = predict_loader(
            model=model,
            loader=valid_loader,
            device=device,
            use_amp=(config.use_amp and device.type == "cuda"),
            progress_name=f"fold{fold_index + 1}-valid",
            log_interval=(
                predict_log_interval if bool(config.show_predict_progress) else 0
            ),
        )
        test_fold_pred = predict_loader(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=(config.use_amp and device.type == "cuda"),
            progress_name=f"fold{fold_index + 1}-test",
            log_interval=(
                predict_log_interval if bool(config.show_predict_progress) else 0
            ),
        )

        fold_pauc = partial_auc_score(y_valid, valid_pred)

        oof_pred[valid_idx] = valid_pred
        oof_mask[valid_idx] = True
        test_pred_sum += test_fold_pred
        n_test_ensembles += 1

        fold_info = {
            "fold": fold_index,
            "best_valid_pauc": float(best_valid_score),
            "fold_valid_pauc": float(fold_pauc),
            "n_train": int(len(fold_train_df)),
            "n_valid": int(len(fold_valid_df)),
            "epochs_ran": int(len(history)),
        }
        fold_metrics.append(fold_info)
        print("Fold summary:", fold_info)

        if config.save_checkpoints:
            checkpoint_path = output_dir / f"mid_fusion_lite_fold_{fold_index}.pt"
            torch.save(best_state, checkpoint_path)

        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if n_test_ensembles == 0:
        raise RuntimeError("No folds were selected for training.")

    test_pred = test_pred_sum / float(n_test_ensembles)

    # Apply Patient-wise Post-processing if configured
    if bool(config.use_patient_post_processing):
        print("Applying Patient-wise Post-processing on predictions...")
        raw_cv_pauc = partial_auc_score(y[oof_mask], oof_pred[oof_mask]) if np.any(oof_mask) else float("nan")
        
        # Post-process OOF predictions
        oof_pred_processed = _apply_patient_post_processing(oof_pred, train_df[config.group_col].to_numpy())
        
        # Calculate CV pAUC with post-processed predictions
        cv_pauc = partial_auc_score(y[oof_mask], oof_pred_processed[oof_mask]) if np.any(oof_mask) else float("nan")
        print(f"-> Raw OOF pAUC: {raw_cv_pauc:.6f} | Post-processed OOF pAUC: {cv_pauc:.6f}")
        
        # Post-process test predictions
        test_pred = _apply_patient_post_processing(test_pred, test_df[config.group_col].to_numpy())
    else:
        if not np.any(oof_mask):
            cv_pauc = float("nan")
        else:
            cv_pauc = partial_auc_score(y[oof_mask], oof_pred[oof_mask])

    oof_df = train_df[[config.id_col, config.target_col]].copy()
    oof_df["oof_pred"] = oof_pred
    if bool(config.use_patient_post_processing):
        oof_df["oof_pred_processed"] = oof_pred_processed
    oof_df["used_in_cv"] = oof_mask.astype(int)

    submission_df = sample_submission.copy()
    submission_df["target"] = test_pred

    oof_path = output_dir / "oof_predictions.csv"
    submission_path = output_dir / "submission_mid_fusion_lite.csv"
    metrics_path = output_dir / "metrics.json"

    oof_df.to_csv(oof_path, index=False)
    submission_df.to_csv(submission_path, index=False)

    metrics_payload = {
        "preset": config.preset_name,
        "cv_pauc": float(cv_pauc),
        "n_test_ensembles": int(n_test_ensembles),
        "fold_metrics": fold_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"CV pAUC: {cv_pauc:.6f}")
    print(f"Saved OOF: {oof_path}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved metrics: {metrics_path}")

    return {
        "config": config,
        "cv_pauc": float(cv_pauc),
        "fold_metrics": fold_metrics,
        "oof_df": oof_df,
        "submission_df": submission_df,
        "oof_path": str(oof_path),
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
    }


def run_with_preset(preset_name: str = "quick", **override_kwargs) -> dict[str, Any]:
    config = get_preset_config(preset_name)
    if override_kwargs:
        filtered_updates = _filter_supported_config_updates(config, override_kwargs)
        if filtered_updates:
            config = config.with_updates(**filtered_updates)
    return run_mid_fusion_cv(config)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Mid Fusion Lite CV pipeline")
    parser.add_argument(
        "--preset", default="quick", choices=["smoke", "quick", "full_lite"]
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--backbone-weights-path", default=None)
    parser.add_argument(
        "--folds", default=None, help="Comma-separated fold ids, e.g. 0,1"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)

    args = parser.parse_args()

    config = get_preset_config(args.preset)

    updates = {}
    if args.output_dir is not None:
        updates["output_dir"] = args.output_dir
    if args.backbone_weights_path is not None:
        updates["backbone_weights_path"] = args.backbone_weights_path
    if args.folds is not None:
        updates["fold_indices"] = [
            int(x.strip()) for x in args.folds.split(",") if x.strip()
        ]
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.image_size is not None:
        updates["image_size"] = args.image_size

    if updates:
        filtered_updates = _filter_supported_config_updates(config, updates)
        if filtered_updates:
            config = config.with_updates(**filtered_updates)

    run_mid_fusion_cv(config)


if __name__ == "__main__":
    main()
