from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

try:
    from .config import MidFusionLiteConfig, get_preset_config, resolve_data_paths
    from .data import (
        HDF5ImageTabularDataset,
        get_image_transforms,
        make_balanced_sampler,
    )
    from .model import MidFusionLiteModel
    from .tabular import FoldTabularPreprocessor, select_available_columns
    from .train import fit_one_fold, partial_auc_score, predict_loader
except ImportError:
    from config import MidFusionLiteConfig, get_preset_config, resolve_data_paths
    from model import MidFusionLiteModel
    from tabular import FoldTabularPreprocessor, select_available_columns
    from train import fit_one_fold, partial_auc_score, predict_loader

    from data import (
        HDF5ImageTabularDataset,
        get_image_transforms,
        make_balanced_sampler,
    )


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


def run_mid_fusion_cv(config: MidFusionLiteConfig) -> dict[str, Any]:
    config = resolve_data_paths(config)
    set_seed(config.seed)
    device = _get_device()

    output_dir = _ensure_output_dir(config.output_dir)

    print(f"Running preset: {config.preset_name}")
    print(f"Device: {device.type.upper()}")
    print(f"Output directory: {output_dir.resolve()}")

    train_df = pd.read_csv(config.train_metadata_path, low_memory=False)
    test_df = pd.read_csv(config.test_metadata_path, low_memory=False)
    sample_submission = pd.read_csv(config.sample_submission_path)

    _validate_columns(config, train_df, test_df)

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
        )
        valid_dataset = HDF5ImageTabularDataset(
            ids=fold_valid_df[config.id_col].astype(str).to_numpy(),
            tabular=x_valid,
            hdf5_path=config.train_hdf5_path,
            targets=y_valid,
            transforms=transforms["valid"],
        )
        test_dataset = HDF5ImageTabularDataset(
            ids=test_df[config.id_col].astype(str).to_numpy(),
            tabular=x_test,
            hdf5_path=config.test_hdf5_path,
            targets=None,
            transforms=transforms["valid"],
        )

        sampler = make_balanced_sampler(
            targets=y_train,
            negative_ratio=config.negative_ratio,
            seed=config.seed + fold_index,
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
            tab_hidden_dim=config.tab_hidden_dim,
            fusion_hidden_dim=config.fusion_hidden_dim,
            tab_dropout=config.tab_dropout,
            fusion_dropout=config.fusion_dropout,
            train_image_backbone=config.train_image_backbone,
        ).to(device)

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
        pos_weight = n_neg / n_pos

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
        )

        model.load_state_dict(best_state)

        valid_pred = predict_loader(
            model=model,
            loader=valid_loader,
            device=device,
            use_amp=(config.use_amp and device.type == "cuda"),
        )
        test_fold_pred = predict_loader(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=(config.use_amp and device.type == "cuda"),
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

    if not np.any(oof_mask):
        cv_pauc = float("nan")
    else:
        cv_pauc = partial_auc_score(y[oof_mask], oof_pred[oof_mask])

    oof_df = train_df[[config.id_col, config.target_col]].copy()
    oof_df["oof_pred"] = oof_pred
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
        config = config.with_updates(**override_kwargs)
    return run_mid_fusion_cv(config)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Mid Fusion Lite CV pipeline")
    parser.add_argument(
        "--preset", default="quick", choices=["smoke", "quick", "full_lite"]
    )
    parser.add_argument("--output-dir", default=None)
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
    if args.folds is not None:
        updates["fold_indices"] = [
            int(x.strip()) for x in args.folds.split(",") if x.strip()
        ]
    if args.epochs is not None:
        updates["epochs"] = args.epochs
    if args.image_size is not None:
        updates["image_size"] = args.image_size

    if updates:
        config = config.with_updates(**updates)

    run_mid_fusion_cv(config)


if __name__ == "__main__":
    main()
