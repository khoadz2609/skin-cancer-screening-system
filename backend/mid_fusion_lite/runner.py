from __future__ import annotations

import gc
import json
import os
import random
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Optional, Callable

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
        modality_mode_for_run_mode,
        resolve_data_paths,
        resolve_run_mode,
    )
    from .data import (
        HDF5ImageTabularDataset,
        get_image_transforms,
        make_balanced_sampler,
    )
    from .tabular_engineering_02 import add_feature_engineering_folds
    from .model import MidFusionLiteModel
    from .tabular_preprocessor_04 import FoldTabularPreprocessor, select_available_columns
    from .train import fit_one_fold, partial_auc_score, predict_loader
except ImportError:
    from config import (
        BASE_NUMERIC_COLS,
        MidFusionLiteConfig,
        get_preset_config,
        modality_mode_for_run_mode,
        resolve_data_paths,
        resolve_run_mode,
    )
    from data import (
        HDF5ImageTabularDataset,
        get_image_transforms,
        make_balanced_sampler,
    )
    from tabular_engineering_02 import add_feature_engineering_folds
    from model import MidFusionLiteModel
    from tabular_preprocessor_04 import FoldTabularPreprocessor, select_available_columns
    from train import fit_one_fold, partial_auc_score, predict_loader


def resolve_weights_path(weights_path: Optional[str], fold_index: int) -> Optional[str]:
    if not weights_path:
        return None
    path = Path(weights_path)
    if not path.is_dir():
        raise ValueError(f"weights_path must be a directory, got: {weights_path}")

    pt_files = list(path.glob("*.pt*"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in directory: {weights_path}")
        
    # If there's exactly one file, just return it
    if len(pt_files) == 1:
        return str(pt_files[0])
        
    # Determine if files are 0-indexed or 1-indexed based on the minimum number
    file_nums = {}
    for f in pt_files:
        nums = re.findall(r'\d+', f.name)
        if nums:
            file_nums[int(nums[-1])] = f
            
    if not file_nums:
        raise FileNotFoundError(f"Could not parse fold numbers from .pt files in: {weights_path}")
        
    min_idx = min(file_nums.keys())
    target_idx = fold_index + 1 if min_idx == 1 else fold_index
    
    if target_idx in file_nums:
        return str(file_nums[target_idx])
        
    raise FileNotFoundError(f"Could not find a weight file for fold {fold_index} (expected index {target_idx}) in: {weights_path}")


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


def _pretrained_checkpoint_names(branch: str, fold_index: int) -> list[str]:
    """Return supported checkpoint names in preferred order for one fold."""
    if branch == "tabular":
        return [
            f"mid_fusion_lite_fold_{fold_index}.pt",
            f"tabm_fold_{fold_index}.pt",
            f"tabm_fold_{fold_index + 1}.pt",
            f"tabular_fold_{fold_index}.pt",
        ]
    if branch == "image":
        return [
            f"mid_fusion_lite_fold_{fold_index}.pt",
            f"image_only_fold_{fold_index}.pt",
            f"image_fold_{fold_index}.pt",
            f"image_fold_{fold_index + 1}.pt",
        ]
    raise ValueError(f"Unsupported checkpoint branch: {branch}")


def _resolve_pretrained_checkpoint(
    source: str,
    fold_index: int,
    branch: str,
) -> Optional[Path]:
    """Resolve a checkpoint template, file, or directory (including subfolders)."""
    if "{fold}" in source or "{fold_1}" in source:
        candidate = Path(source.format(fold=fold_index, fold_1=fold_index + 1))
        return candidate if candidate.is_file() else None

    source_path = Path(source)
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        return None

    names = _pretrained_checkpoint_names(branch, fold_index)
    for name in names:
        direct_candidate = source_path / name
        if direct_candidate.is_file():
            return direct_candidate

    # Kaggle datasets often wrap uploaded files in one or more extra folders.
    for name in names:
        matches = sorted(path for path in source_path.rglob(name) if path.is_file())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            formatted = "\n  - ".join(str(path) for path in matches)
            raise RuntimeError(
                f"Multiple {branch} checkpoints match fold {fold_index} under "
                f"{source_path}:\n  - {formatted}\n"
                "Use a path template with {fold} or {fold_1} to disambiguate."
            )
    return None


def _missing_stage3_checkpoint_error(
    source: str,
    fold_index: int,
    branch: str,
) -> FileNotFoundError:
    expected = ", ".join(_pretrained_checkpoint_names(branch, fold_index))
    config_key = (
        "pretrained_tabular_dir" if branch == "tabular" else "pretrained_image_dir"
    )
    return FileNotFoundError(
        f"Stage 3 requires a pre-trained {branch} checkpoint for fold "
        f"{fold_index}, but none was found under: {source}\n"
        f"Expected one of: {expected}\n"
        f"Set {config_key} to the correct directory, checkpoint file, or a "
        "template containing {fold}/{fold_1}. Stage 3 will not train the "
        "missing unimodal branch from scratch."
    )


def _model_kwargs_for_mode(
    config: MidFusionLiteConfig,
    modality_mode: str,
    n_tab_features: int,
    bins: Any,
    init_bias: Optional[float],
    cat_cardinalities: Optional[list[int]],
    fold_index: int,
) -> dict[str, Any]:
    """Build only the constructor arguments used by the requested mode."""
    mode = str(modality_mode).strip().lower()
    kwargs: dict[str, Any] = {
        "modality_mode": mode,
        "init_bias": init_bias,
    }

    if mode in {"both", "image_only"}:
        kwargs.update(
            backbone_name=config.backbone_name,
            fusion_hidden_dim=config.fusion_hidden_dim,
            fusion_mode=config.fusion_mode,
            fusion_dropout=config.fusion_dropout,
            train_image_backbone=config.train_image_backbone,
            download_pretrained_backbone=config.download_pretrained_backbone,
            backbone_weights_path=resolve_weights_path(
                config.backbone_weights_path, fold_index
            ),
        )

    if mode in {"both", "tabular_only"}:
        kwargs.update(
            n_tab_features=n_tab_features,
            use_tabm_tabular=config.use_tabm_tabular,
            tab_dropout=config.tab_dropout,
            bins=bins,
            d_embedding=config.d_embedding,
            cat_cardinalities=cat_cardinalities,
        )
        if config.use_tabm_tabular:
            kwargs.update(
                tabm_arch_type=config.tabm_arch_type,
                tabm_k=config.tabm_k,
                tabm_n_blocks=config.tabm_n_blocks,
                tabm_d_block=config.tabm_d_block,
            )
        else:
            kwargs["tab_hidden_dim"] = config.tab_hidden_dim

    if mode == "both":
        kwargs["modality_dropout_prob"] = config.modality_dropout_prob
        if str(config.fusion_mode).strip().lower() == "cross_attention":
            kwargs.update(
                cross_attention_heads=config.cross_attention_heads,
                cross_attention_dropout=config.cross_attention_dropout,
            )

    return kwargs





def _run_stage1_tabular(
    fold_index: int,
    config: MidFusionLiteConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    x_train_shape: int,
    bins: Any,
    cat_cardinalities: Optional[list[int]] = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float]:
    best_tab_state = None
    history = []
    best_tab_score = 0.0
    if config.pretrained_tabular_dir is not None:
        tab_source = str(config.pretrained_tabular_dir)
        tab_path = _resolve_pretrained_checkpoint(
            tab_source, fold_index=fold_index, branch="tabular"
        )
        if tab_path is not None:
            print(f"\n[Stage 1] Loading pre-trained tabular weights from checkpoint: {tab_path}")
            best_tab_state = torch.load(tab_path, map_location="cpu", weights_only=True)
        elif resolve_run_mode(config) == "fusion":
            raise _missing_stage3_checkpoint_error(
                tab_source, fold_index=fold_index, branch="tabular"
            )
        else:
            print(
                "\n[Stage 1] WARNING: Pretrained tabular checkpoint not found "
                f"under {tab_source}. Training from scratch..."
            )
    elif resolve_run_mode(config) == "fusion":
        raise _missing_stage3_checkpoint_error(
            "<not configured>", fold_index=fold_index, branch="tabular"
        )

    if best_tab_state is None:
        print(f"\n[Stage 1] Pretraining Tabular encoder (epochs={config.stage1_epochs})...")
        init_bias = -float(np.log(max(1.0, float(config.negative_ratio))))
        tabular_model_kwargs: dict[str, Any] = {
            "n_tab_features": x_train_shape,
            "use_tabm_tabular": config.use_tabm_tabular,
            "tab_dropout": config.tab_dropout,
            "modality_mode": "tabular_only",
            "bins": bins,
            "d_embedding": config.d_embedding,
            "cat_cardinalities": cat_cardinalities,
            "init_bias": init_bias,
        }
        if config.use_tabm_tabular:
            tabular_model_kwargs.update(
                tabm_arch_type=config.tabm_arch_type,
                tabm_k=config.tabm_k,
                tabm_n_blocks=config.tabm_n_blocks,
                tabm_d_block=config.tabm_d_block,
            )
        else:
            tabular_model_kwargs["tab_hidden_dim"] = config.tab_hidden_dim

        model_tab = MidFusionLiteModel(**tabular_model_kwargs).to(device)

        _initialize_lazy_parameters(model_tab, device, config.image_size, x_train_shape)

        train_ds_tab = HDF5ImageTabularDataset(
            ids=train_loader.dataset.ids,
            tabular=train_loader.dataset.tabular,
            hdf5_path=train_loader.dataset.hdf5_path,
            targets=train_loader.dataset.targets,
            transforms=None,
            use_shades_of_gray=False,
            modality_mode="tabular_only",
            image_size=config.image_size,
        )
        valid_ds_tab = HDF5ImageTabularDataset(
            ids=valid_loader.dataset.ids,
            tabular=valid_loader.dataset.tabular,
            hdf5_path=valid_loader.dataset.hdf5_path,
            targets=valid_loader.dataset.targets,
            transforms=None,
            use_shades_of_gray=False,
            modality_mode="tabular_only",
            image_size=config.image_size,
        )
        train_loader_tab = DataLoader(
            train_ds_tab,
            batch_size=train_loader.batch_size,
            sampler=train_loader.sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last,
        )
        valid_loader_tab = DataLoader(
            valid_ds_tab,
            batch_size=valid_loader.batch_size,
            sampler=None,
            num_workers=valid_loader.num_workers,
            pin_memory=valid_loader.pin_memory,
            drop_last=False,
        )

        opt_tab = torch.optim.AdamW(model_tab.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        warmup_epochs = config.warmup_epochs
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                opt_tab, start_factor=0.1, total_iters=warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_tab, T_max=max(1, config.stage1_epochs - warmup_epochs), eta_min=config.min_lr
            )
            scheduler_tab = torch.optim.lr_scheduler.SequentialLR(
                opt_tab, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler_tab = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_tab, T_max=max(1, config.stage1_epochs), eta_min=config.min_lr
            )

        best_tab_state, history, best_tab_score = fit_one_fold(
            model=model_tab,
            train_loader=train_loader_tab,
            valid_loader=valid_loader_tab,
            optimizer=opt_tab,
            scheduler=scheduler_tab,
            device=device,
            epochs=config.stage1_epochs,
            patience=config.patience,
            pos_weight=pos_weight,
            use_amp=(config.use_amp and device.type == "cuda"),
            max_train_steps_per_epoch=config.max_train_steps_per_epoch,
            max_valid_steps=config.max_valid_steps,
            aux_loss_image_weight=0.0,
            aux_loss_tab_weight=1.0,
            gradient_clipping_norm=config.gradient_clipping_norm,
            label_smoothing=config.label_smoothing,
        )
        del model_tab, opt_tab
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Kaggle input datasets are read-only. Newly trained checkpoints always
        # belong in output_dir, never in the pre-trained source directory.
        if best_tab_state is not None and config.save_checkpoints:
            save_dir = Path(config.output_dir) / "stage1_tabular"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"mid_fusion_lite_fold_{fold_index}.pt"
            torch.save(best_tab_state, save_path)
            print(f"  [Stage 1] Checkpoint saved → {save_path}")

    return best_tab_state, history, best_tab_score


def _run_stage2_image(
    fold_index: int,
    config: MidFusionLiteConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    x_train_shape: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], float]:
    best_img_state = None
    history: list[dict[str, Any]] = []
    best_img_score = 0.0
    if config.pretrained_image_dir is not None:
        image_source = str(config.pretrained_image_dir)
        img_path = _resolve_pretrained_checkpoint(
            image_source, fold_index=fold_index, branch="image"
        )
        if img_path is not None:
            print(f"\n[Stage 2] Loading pre-trained image weights from checkpoint: {img_path}")
            best_img_state = torch.load(img_path, map_location="cpu", weights_only=True)
        elif resolve_run_mode(config) == "fusion":
            raise _missing_stage3_checkpoint_error(
                image_source, fold_index=fold_index, branch="image"
            )
        else:
            print(
                "\n[Stage 2] WARNING: Pretrained image checkpoint not found "
                f"under {image_source}. Training from scratch..."
            )
    elif resolve_run_mode(config) == "fusion":
        raise _missing_stage3_checkpoint_error(
            "<not configured>", fold_index=fold_index, branch="image"
        )

    if best_img_state is None:
        print(f"\n[Stage 2] Pretraining Image encoder (epochs={config.stage2_epochs})...")
        init_bias = -float(np.log(max(1.0, float(config.negative_ratio))))
        model_img = MidFusionLiteModel(
            backbone_name=config.backbone_name,
            fusion_hidden_dim=config.fusion_hidden_dim,
            fusion_mode=config.fusion_mode,
            fusion_dropout=config.fusion_dropout,
            train_image_backbone=config.stage2_train_backbone,
            download_pretrained_backbone=config.download_pretrained_backbone,
            backbone_weights_path=resolve_weights_path(config.backbone_weights_path, fold_index),
            modality_mode="image_only",
            init_bias=init_bias,
        ).to(device)

        _initialize_lazy_parameters(model_img, device, config.image_size, x_train_shape)

        opt_img = torch.optim.AdamW(model_img.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        warmup_epochs = max(1, config.warmup_epochs) if config.warmup_epochs > 0 else 0
        if warmup_epochs > 0 and config.stage2_epochs > warmup_epochs:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                opt_img, start_factor=0.1, total_iters=warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_img, T_max=config.stage2_epochs - warmup_epochs, eta_min=config.min_lr
            )
            scheduler_img = torch.optim.lr_scheduler.SequentialLR(
                opt_img, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
            )
        else:
            scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_img, T_max=max(1, config.stage2_epochs), eta_min=config.min_lr
            )

        best_img_state, history, best_img_score = fit_one_fold(
            model=model_img,
            train_loader=train_loader,
            valid_loader=valid_loader,
            optimizer=opt_img,
            scheduler=scheduler_img,
            device=device,
            epochs=config.stage2_epochs,
            patience=config.patience,
            pos_weight=pos_weight,
            use_amp=(config.use_amp and device.type == "cuda"),
            max_train_steps_per_epoch=config.max_train_steps_per_epoch,
            max_valid_steps=config.max_valid_steps,
            aux_loss_image_weight=1.0,
            aux_loss_tab_weight=0.0,
            gradient_clipping_norm=config.gradient_clipping_norm,
            label_smoothing=config.label_smoothing,
        )
        del model_img, opt_img
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Save Stage 2 checkpoint so it can be reloaded without retraining.
        # This is the cleanest image-only checkpoint — before Stage 3 fusion distorts image_head.
        # Uses pretrained_image_dir if set, otherwise output_dir/stage2_image/.
        if best_img_state is not None and config.save_checkpoints:
            # Never write into a pre-trained source such as /kaggle/input.
            save_dir = Path(config.output_dir) / "stage2_image"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"mid_fusion_lite_fold_{fold_index}.pt"
            torch.save(best_img_state, save_path)
            print(f"  [Stage 2] Checkpoint saved → {save_path}")

    if best_img_state is None:
        raise RuntimeError("Stage 2 did not produce or load an image checkpoint")
    return best_img_state, history, float(best_img_score)


def _run_stage3_fusion(
    fold_index: int,
    config: MidFusionLiteConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    x_train_shape: int,
    bins: Any,
    best_tab_state: Optional[dict[str, torch.Tensor]],
    best_img_state: Optional[dict[str, torch.Tensor]],
    cat_cardinalities: Optional[list[int]] = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float]:
    print(f"\n[Stage 3] Fusion & Fine-Tuning...")
    init_bias = -float(np.log(max(1.0, float(config.negative_ratio))))
    model = MidFusionLiteModel(
        **_model_kwargs_for_mode(
            config=config,
            modality_mode="both",
            n_tab_features=x_train_shape,
            bins=bins,
            init_bias=init_bias,
            cat_cardinalities=cat_cardinalities,
            fold_index=fold_index,
        )
    ).to(device)

    _initialize_lazy_parameters(model, device, config.image_size, x_train_shape)

    model.load_unimodal_weights(
        tabular_state=best_tab_state,
        image_state=best_img_state,
    )

    print(f"\n  [Stage 3A] Freeze encoders, train fusion head (epochs={config.stage3_freeze_epochs})...")
    for p in model.tabular_encoder.parameters():
        p.requires_grad = False
    for p in model.image_encoder.parameters():
        p.requires_grad = False

    _stage3a_lr = config.stage3a_lr if config.stage3a_lr is not None else config.lr
    fusion_params = [p for p in model.parameters() if p.requires_grad]
    opt_fusion = torch.optim.AdamW(fusion_params, lr=_stage3a_lr, weight_decay=config.weight_decay)
    print(f"  Stage 3A lr={_stage3a_lr:.2e}")

    # Warmup for Stage 3A: cross_attention/fusion head is randomly initialized →
    # constant high LR causes unstable valid_loss oscillation in early epochs.
    n_freeze = max(1, int(config.stage3_freeze_epochs))
    warmup_3a = 1 if n_freeze > 1 else 0
    if warmup_3a > 0:
        _w3a = torch.optim.lr_scheduler.LinearLR(
            opt_fusion, start_factor=0.1, total_iters=warmup_3a
        )
        _c3a = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_fusion, T_max=max(1, n_freeze - warmup_3a), eta_min=config.min_lr
        )
        scheduler_3a = torch.optim.lr_scheduler.SequentialLR(
            opt_fusion, schedulers=[_w3a, _c3a], milestones=[warmup_3a]
        )
    else:
        scheduler_3a = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_fusion, T_max=n_freeze, eta_min=config.min_lr
        )

    best_fusion_state, history_a, best_score_a = fit_one_fold(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=opt_fusion,
        scheduler=scheduler_3a,
        device=device,
        epochs=config.stage3_freeze_epochs,
        patience=config.patience,
        pos_weight=pos_weight,
        use_amp=(config.use_amp and device.type == "cuda"),
        max_train_steps_per_epoch=config.max_train_steps_per_epoch,
        max_valid_steps=config.max_valid_steps,
        # Fix: encoders are frozen in Stage 3A → auxiliary losses cannot propagate
        # gradient to the backbone, so setting them to 0 avoids wasted computation.
        aux_loss_image_weight=0.0,
        aux_loss_tab_weight=0.0,
        gradient_clipping_norm=config.gradient_clipping_norm,
        label_smoothing=config.label_smoothing,
    )

    model.load_state_dict(best_fusion_state)

    print(f"\n  [Stage 3B] Unfreeze encoders, joint fine-tuning (epochs={config.stage3_unfreeze_epochs})...")
    for p in model.tabular_encoder.parameters():
        p.requires_grad = True
    if config.train_image_backbone:
        for p in model.image_encoder.parameters():
            p.requires_grad = True

    _stage3b_lr = config.stage3b_lr if config.stage3b_lr is not None else 0.1 * config.lr
    # Differential lr: the image backbone fine-tunes gently at _stage3b_lr, while the
    # tabular encoder + fusion head (only briefly trained in Stage 3A) get a higher lr so
    # they keep adapting instead of freezing at the backbone's tiny lr. Split params by
    # identity to avoid double-counting shared tensors.
    _head_lr_mult = float(getattr(config, "stage3b_head_lr_mult", 5.0))
    backbone_param_ids = (
        {id(p) for p in model.image_encoder.parameters()}
        if model.image_encoder is not None
        else set()
    )
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) in backbone_param_ids
    ]
    head_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in backbone_param_ids
    ]
    _head_lr = _stage3b_lr * _head_lr_mult
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": _stage3b_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": _head_lr})
    opt_finetune = torch.optim.AdamW(param_groups, lr=_stage3b_lr, weight_decay=config.weight_decay)
    print(f"  Stage 3B lr: backbone={_stage3b_lr:.2e} | head/tabular={_head_lr:.2e}")

    # Fix: add 1-epoch LinearLR warmup before cosine decay.
    # Unfreezing encoders mid-training causes sudden large gradient spikes;
    # warmup lets the optimizer settle before the full lr is applied.
    n_unfreeze = max(1, int(config.stage3_unfreeze_epochs))
    warmup_unfreeze = 1 if n_unfreeze > 1 else 0
    if warmup_unfreeze > 0:
        _warmup = torch.optim.lr_scheduler.LinearLR(
            opt_finetune, start_factor=0.1, total_iters=warmup_unfreeze
        )
        _cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_finetune, T_max=max(1, n_unfreeze - warmup_unfreeze), eta_min=config.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt_finetune, schedulers=[_warmup, _cosine], milestones=[warmup_unfreeze]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_finetune, T_max=n_unfreeze, eta_min=config.min_lr
        )

    best_final_state, history_b, best_score_b = fit_one_fold(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=opt_finetune,
        scheduler=scheduler,
        device=device,
        epochs=config.stage3_unfreeze_epochs,
        patience=config.patience,
        pos_weight=pos_weight,
        use_amp=(config.use_amp and device.type == "cuda"),
        max_train_steps_per_epoch=config.max_train_steps_per_epoch,
        max_valid_steps=config.max_valid_steps,
        aux_loss_image_weight=config.aux_loss_image_weight,
        aux_loss_tab_weight=config.aux_loss_tab_weight,
        gradient_clipping_norm=config.gradient_clipping_norm,
        label_smoothing=config.label_smoothing,
    )

    combined_history = history_a + history_b
    # Fix: return the checkpoint that actually achieved the best score.
    # If Stage 3A outperformed Stage 3B, best_final_state (3B) would be returned
    # with a misleadingly high score from 3A — now we return the correct pair.
    if best_score_b >= best_score_a:
        return best_final_state, combined_history, float(best_score_b)
    else:
        print(
            f"  Stage 3A pAUC ({best_score_a:.5f}) > Stage 3B ({best_score_b:.5f})."
            " Returning Stage 3A best checkpoint."
        )
        return best_fusion_state, combined_history, float(best_score_a)


def _run_staged_pretraining_for_fold(
    fold_index: int,
    config: MidFusionLiteConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    x_train_shape: int,
    bins: Any,
    cat_cardinalities: Optional[list[int]] = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float]:
    run_mode = resolve_run_mode(config)
    training_title = (
        "STAGE 3 FUSION FROM PRETRAINED BRANCHES"
        if run_mode == "fusion"
        else "3-STAGE PROGRESSIVE TRAINING"
    )
    print(f"\n" + "=" * 80)
    print(f"  >>> RUNNING {training_title} FOR FOLD {fold_index + 1} <<<")
    print("=" * 80)

    best_tab_state, _, _ = _run_stage1_tabular(
        fold_index, config, train_loader, valid_loader, device, pos_weight, x_train_shape, bins,
        cat_cardinalities=cat_cardinalities,
    )

    best_img_state, _, _ = _run_stage2_image(
        fold_index, config, train_loader, valid_loader, device, pos_weight, x_train_shape
    )

    return _run_stage3_fusion(
        fold_index, config, train_loader, valid_loader, device, pos_weight, x_train_shape, bins, best_tab_state, best_img_state,
        cat_cardinalities=cat_cardinalities,
    )


def run_mid_fusion_cv(config: MidFusionLiteConfig, epoch_callback: Optional[Callable[[int, float], None]] = None) -> dict[str, Any]:
    config = resolve_data_paths(config)

    run_mode = resolve_run_mode(config)
    # Preserve epoch/backbone semantics for legacy non-staged unimodal configs.
    if config.run_mode is None and not bool(config.use_staged_pretraining):
        if run_mode == "tabular_only":
            config = config.with_updates(stage1_epochs=config.epochs)
        elif run_mode == "image_only":
            config = config.with_updates(
                stage2_epochs=config.epochs,
                stage2_train_backbone=config.train_image_backbone,
            )
    stage_modality_mode = modality_mode_for_run_mode(run_mode)
    uses_stage_runner = run_mode != "end_to_end"

    if config.backbone_weights_path:
        if "{fold}" not in config.backbone_weights_path and not Path(config.backbone_weights_path).exists():
            print(
                "Backbone weights path does not exist on this machine. "
                "Ignoring backbone_weights_path and using default backbone init."
            )
            config = config.with_updates(backbone_weights_path=None)

    set_seed(config.seed)
    device = _get_device()

    output_dir = _ensure_output_dir(config.output_dir)

    print(f"Running preset: {config.preset_name}")
    print(f"Run mode: {run_mode}")
    print(f"Device: {device.type.upper()}")
    print(f"Output directory: {output_dir.resolve()}")

    uses_tabular_data = stage_modality_mode != "image_only"
    requested_numeric_cols = set(config.numeric_cols) if uses_tabular_data else set()
    if uses_tabular_data and bool(config.use_feature_engineering):
        requested_numeric_cols.update(BASE_NUMERIC_COLS)

    requested_train_cols = {
        config.id_col,
        config.target_col,
        config.group_col,
        "tbp_lv_nevi_confidence", # Ensure this is read for hard negative mining
        *(config.categorical_cols if uses_tabular_data else []),
        *requested_numeric_cols,
    }
    requested_test_cols = {
        config.id_col,
        config.group_col,
        *(config.categorical_cols if uses_tabular_data else []),
        *requested_numeric_cols,
    }

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

    # NOTE: Feature engineering is intentionally NOT applied here on the whole train_df.
    # Some engineered features are global statistics (anatomical z-scores, median
    # imputation, LOF standardization) that would leak validation info if fit over
    # train+valid together. Instead, add_feature_engineering_folds is called INSIDE the
    # fold loop below, fitting stats on fold-train only. Column selection likewise moves
    # into the loop (the engineered column set is deterministic across folds).
    if uses_tabular_data and not bool(config.use_feature_engineering):
        print("Skipping heavy feature engineering (use_feature_engineering=False).")
    elif not uses_tabular_data:
        print("Skipping tabular feature engineering for image-only Stage 2.")

    y = train_df[config.target_col].astype(np.int64).to_numpy()
    groups = train_df[config.group_col].to_numpy()

    if config.fold_col is not None and config.fold_col in train_df.columns:
        print(f"Found fold column '{config.fold_col}' in train metadata. Using pre-defined fold splits.")
        # Identify unique folds in sorted order (excluding NaNs and negative indices)
        unique_folds = sorted([f for f in train_df[config.fold_col].unique() if pd.notna(f) and f >= 0])
        folds = []
        for f in unique_folds:
            valid_idx = np.where(train_df[config.fold_col] == f)[0]
            train_idx = np.where((train_df[config.fold_col] != f) & (train_df[config.fold_col] >= 0))[0]
            folds.append((train_idx, valid_idx))
        if len(folds) != config.n_splits:
            print(f"[Warning] Number of unique folds in '{config.fold_col}' ({len(folds)}) does not match config.n_splits ({config.n_splits}). Using folds from column.")
    else:
        print(f"Fold column '{config.fold_col}' not found. Generating fold splits dynamically using StratifiedGroupKFold.")
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

        # Leak-safe feature engineering: fit engineering statistics on THIS fold's train
        # split only, then transform fold-valid and test with those frozen stats. test_df
        # is left untouched (the helper copies internally) so each fold re-derives its own
        # fold_test_df — required because the fold-train statistics differ per fold.
        fold_test_df = test_df
        if uses_tabular_data and bool(config.use_feature_engineering):
            try:
                fold_train_df, fold_valid_df, fold_test_df = add_feature_engineering_folds(
                    fold_train_df=fold_train_df,
                    fold_valid_df=fold_valid_df,
                    test_df=test_df,
                    group_col=config.group_col,
                    categorical_cols=config.categorical_cols,
                    use_contextual_group_norm_features=bool(config.use_contextual_group_norm_features),
                    use_ugly_duckling_features=bool(config.use_ugly_duckling_features),
                    use_anatomical_zscore_features=bool(config.use_anatomical_zscore_features),
                    use_area_aggregation_features=bool(config.use_area_aggregation_features),
                )
            except MemoryError:
                print(
                    "MemoryError during feature engineering. "
                    "Falling back to raw features for this fold."
                )
                gc.collect()
                fold_test_df = test_df

        y_train = fold_train_df[config.target_col].astype(np.float32).to_numpy()
        y_valid = fold_valid_df[config.target_col].astype(np.float32).to_numpy()

        # We pass the full dataset to the preprocessor and dataloader.
        # Undersampling is dynamically handled by make_balanced_sampler per epoch.
        fold_train_df_sampled = fold_train_df
        if uses_tabular_data:
            # Column selection is deterministic across folds (engineered names are fixed),
            # but must run on engineered frames to avoid leakage.
            numeric_cols, categorical_cols = select_available_columns(
                train_df=fold_train_df,
                test_df=fold_test_df,
                numeric_candidates=config.numeric_cols,
                categorical_candidates=config.categorical_cols,
            )
            print(
                f"Numeric features: {len(numeric_cols)} | "
                f"Categorical features: {len(categorical_cols)}"
            )
            preprocessor = FoldTabularPreprocessor(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                use_quantile_transform=(
                    bool(config.use_quantile_transform)
                    and bool(config.use_tabm_tabular)
                ),
                quantile_noise_std=float(config.quantile_noise_std),
                quantile_random_state=config.seed + fold_index + 1,
                use_null_importance_filter=bool(config.use_null_importance_filter),
                null_importance_repeats=int(config.null_importance_repeats),
                null_importance_quantile=float(config.null_importance_quantile),
                null_importance_min_kept=int(config.null_importance_min_kept),
                target_col=config.target_col,
                categorical_as_codes=bool(config.use_tabm_tabular),
            )
            preprocessor.fit(fold_train_df_sampled)
            x_train_sampled = preprocessor.transform(fold_train_df_sampled)
            x_valid = preprocessor.transform(fold_valid_df)
            x_test = preprocessor.transform(fold_test_df)
        else:
            # The dataset/model API expects a tabular tensor, but Stage 2 never reads it.
            preprocessor = None
            x_train_sampled = np.zeros((len(fold_train_df_sampled), 1), dtype=np.float32)
            x_valid = np.zeros((len(fold_valid_df), 1), dtype=np.float32)
            x_test = np.zeros((len(fold_test_df), 1), dtype=np.float32)

        y_train_sampled = fold_train_df_sampled[config.target_col].astype(np.float32).to_numpy()
        ids_train_sampled = fold_train_df_sampled[config.id_col].astype(str).to_numpy()

        train_dataset = HDF5ImageTabularDataset(
            ids=ids_train_sampled,
            tabular=x_train_sampled,
            hdf5_path=config.train_hdf5_path,
            targets=y_train_sampled,
            transforms=transforms["train"],
            use_shades_of_gray=bool(config.use_shades_of_gray),
            modality_mode=stage_modality_mode,
            image_size=config.image_size,
        )
        valid_dataset = HDF5ImageTabularDataset(
            ids=fold_valid_df[config.id_col].astype(str).to_numpy(),
            tabular=x_valid,
            hdf5_path=config.train_hdf5_path,
            targets=y_valid,
            transforms=transforms["valid"],
            use_shades_of_gray=bool(config.use_shades_of_gray),
            modality_mode=stage_modality_mode,
            image_size=config.image_size,
        )
        test_dataset = HDF5ImageTabularDataset(
            ids=fold_test_df[config.id_col].astype(str).to_numpy(),
            tabular=x_test,
            hdf5_path=config.test_hdf5_path,
            targets=None,
            transforms=transforms["valid"],
            use_shades_of_gray=bool(config.use_shades_of_gray),
            modality_mode=stage_modality_mode,
            image_size=config.image_size,
        )

        if getattr(config, "tabular_full_data", False) and run_mode == "tabular_only":
            # For tree models or pure tabular pretraining on full data, no undersampling
            train_sampler = None
            train_shuffle = True
        elif config.negative_ratio > 0:
            # Dynamic undersampling per epoch
            train_sampler = make_balanced_sampler(
                targets=y_train_sampled,
                negative_ratio=config.negative_ratio,
                seed=config.seed + fold_index
            )
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size,
            sampler=train_sampler,
            shuffle=train_shuffle,
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

        # Categorical features are fed to TabM as integer codes (one column each, appended
        # after the numeric block) via cat_cardinalities; PLE bins cover numeric features only.
        cat_cardinalities = (
            list(preprocessor.cat_cardinalities)
            if preprocessor is not None and config.use_tabm_tabular
            else None
        )
        n_numeric = int(preprocessor.n_numeric_out) if preprocessor is not None else 0

        bins = None
        if uses_tabular_data and config.use_tabm_tabular and n_numeric > 0:
            import rtdl_num_embeddings
            x_train_tensor = torch.tensor(x_train_sampled[:, :n_numeric], dtype=torch.float32)
            raw_bins = rtdl_num_embeddings.compute_bins(
                x_train_tensor,
                n_bins=config.n_bins
            )
            # Widen the extreme bin edges to prevent clipping test set outliers
            bins = []
            for b in raw_bins:
                b_wide = b.clone()
                if len(b_wide) > 1:
                    span = b_wide[-1] - b_wide[0]
                    margin = max(1e-3, float(0.1 * span))
                    b_wide[0] -= margin
                    b_wide[-1] += margin
                bins.append(b_wide)

        if config.pos_weight is not None:
            pos_weight_full    = float(config.pos_weight)
            pos_weight_sampled = float(config.pos_weight)
        else:
            n_pos = max(float(y_train.sum()), 1.0)
            n_neg = max(float(len(y_train) - y_train.sum()), 1.0)
            # pos_weight_full: used when NO sampler is active (tabular full-data Stage 1).
            # Reflects the true class imbalance in the data.
            pos_weight_full = np.sqrt(n_neg / n_pos)
            # pos_weight_sampled: used when undersampling is active (negative_ratio batches).
            # The manual undersampling already corrects for imbalance, so pos_weight should only reflect
            # the residual ratio seen in each batch, not the raw data ratio.
            pos_weight_sampled = np.sqrt(max(float(config.negative_ratio), 1.0))

        # Each stage is normally launched as its OWN run with its own config.negative_ratio
        # (e.g. Stage 1 at 1:100, Stage 2 at 1:1, Stage 3 at 1:2), so the log should only
        # describe the stage that actually trains in THIS run — not lump Stage 2/3 together.
        neg_ratio = int(max(float(config.negative_ratio), 1.0))
        run_label = {
            "tabular_only": "Tabular only",
            "image_only": "Image only",
            "fusion": "Fusion",
            "progressive": "Progressive tabular -> image -> fusion",
            "end_to_end": "End-to-end fusion",
        }[run_mode]

        # No sampler only in the tabular full-data pretraining special case; otherwise the
        # balanced sampler (negative_ratio:1) is active and pos_weight_sampled applies.
        stage1_no_sampler = (
            bool(getattr(config, "tabular_full_data", False))
            and run_mode == "tabular_only"
        )
        if stage1_no_sampler:
            print(f"  [{run_label}] no sampler (tabular full-data) -> pos_weight_full={pos_weight_full:.3f}")
        else:
            print(f"  [{run_label}] sampler 1:{neg_ratio} -> pos_weight_sampled={pos_weight_sampled:.3f}")

        if uses_stage_runner:
            if run_mode == "tabular_only":
                print(f"\n  >>> RUNNING STAGE 1 (TABULAR ONLY) FOR FOLD {fold_index + 1} <<<")
                # Stage 1 with tabular_full_data uses no sampler → use full-data pos_weight
                stage1_pos_weight = pos_weight_full if getattr(config, "tabular_full_data", False) else pos_weight_sampled
                best_state, history, best_valid_score = _run_stage1_tabular(
                    fold_index, config, train_loader, valid_loader, device, stage1_pos_weight, x_train_sampled.shape[1], bins,
                    cat_cardinalities=cat_cardinalities,
                )
            elif run_mode == "image_only":
                print(
                    f"\n  >>> RUNNING STAGE 2 (IMAGE ONLY) FOR FOLD "
                    f"{fold_index + 1} <<<"
                )
                best_state, history, best_valid_score = _run_stage2_image(
                    fold_index=fold_index,
                    config=config,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    device=device,
                    pos_weight=pos_weight_sampled,
                    x_train_shape=x_train_sampled.shape[1],
                )
            else:
                best_state, history, best_valid_score = _run_staged_pretraining_for_fold(
                    fold_index=fold_index,
                    config=config,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    device=device,
                    # Stage 2 & 3 run with the balanced undersampler active (negative_ratio),
                    # which already corrects most of the class imbalance per batch. Stacking
                    # pos_weight_full (≈sqrt of the raw ratio) on top double-counts the
                    # correction: positives dominate the gradient, train_loss explodes, and
                    # valid_loss drifts upward while pAUC plateaus. Use pos_weight_sampled so
                    # the loss only reflects the residual imbalance the sampler leaves behind.
                    pos_weight=pos_weight_sampled,
                    x_train_shape=x_train_sampled.shape[1],
                    bins=bins,
                    cat_cardinalities=cat_cardinalities,
                )
            
            # Recreate model so we can load the best stage weights and run inference
            init_bias = -float(np.log(max(1.0, float(config.negative_ratio))))
            model = MidFusionLiteModel(
                **_model_kwargs_for_mode(
                    config=config,
                    modality_mode=stage_modality_mode,
                    n_tab_features=x_train_sampled.shape[1],
                    bins=bins,
                    init_bias=init_bias,
                    cat_cardinalities=cat_cardinalities,
                    fold_index=fold_index,
                )
            ).to(device)
            _initialize_lazy_parameters(model, device, config.image_size, x_train_sampled.shape[1])
            model.load_state_dict(best_state)
        else:
            init_bias = -float(np.log(max(1.0, float(config.negative_ratio))))
            model = MidFusionLiteModel(
                **_model_kwargs_for_mode(
                    config=config,
                    modality_mode=stage_modality_mode,
                    n_tab_features=x_train_sampled.shape[1],
                    bins=bins,
                    init_bias=init_bias,
                    cat_cardinalities=cat_cardinalities,
                    fold_index=fold_index,
                )
            )
            model = model.to(device)
            _initialize_lazy_parameters(
                model=model,
                device=device,
                image_size=config.image_size,
                n_tab_features=x_train_sampled.shape[1],
            )

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )

            warmup_epochs = int(config.warmup_epochs)
            total_epochs = int(config.epochs)
            if warmup_epochs > 0 and total_epochs > warmup_epochs:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs - warmup_epochs,
                    eta_min=config.min_lr,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_epochs),
                    eta_min=config.min_lr,
                )

            best_state, history, best_valid_score = fit_one_fold(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=config.epochs,
                patience=config.patience,
                pos_weight=pos_weight_sampled,  # Sampler active → use sampled pos_weight
                use_amp=(config.use_amp and device.type == "cuda"),
                max_train_steps_per_epoch=config.max_train_steps_per_epoch,
                max_valid_steps=config.max_valid_steps,
                aux_loss_image_weight=config.aux_loss_image_weight,
                aux_loss_tab_weight=config.aux_loss_tab_weight,
                gradient_clipping_norm=config.gradient_clipping_norm,
                label_smoothing=config.label_smoothing,
                epoch_callback=epoch_callback,
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

        del model
        if "optimizer" in locals():
            del optimizer
        if "scheduler" in locals():
            del scheduler
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
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    if fold_metrics:
        fold_paucs = [f["best_valid_pauc"] for f in fold_metrics]
        for f in fold_metrics:
            print(f"Fold {f['fold']}: pAUC = {f['best_valid_pauc']:.6f} (Epochs ran: {f['epochs_ran']})")
        avg_pauc = sum(fold_paucs) / len(fold_paucs)
        print("-" * 80)
        print(f"Average Fold pAUC : {avg_pauc:.6f}")
        
    print(f"Overall OOF pAUC  : {cv_pauc:.6f}")
    print("=" * 80)
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
    epoch_callback = override_kwargs.pop("epoch_callback", None)
    config = get_preset_config(preset_name)
    if override_kwargs:
        filtered_updates = _filter_supported_config_updates(config, override_kwargs)
        if filtered_updates:
            config = config.with_updates(**filtered_updates)
    return run_mid_fusion_cv(config, epoch_callback=epoch_callback)


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
    parser.add_argument(
        "--run-mode",
        choices=["tabular_only", "image_only", "fusion", "progressive", "end_to_end"],
        default=None,
    )
    # Deprecated compatibility flag; equivalent to --run-mode progressive.
    parser.add_argument("--staged-training", action="store_true", default=False)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage3-freeze-epochs", type=int, default=None)
    parser.add_argument("--stage3-unfreeze-epochs", type=int, default=None)
    parser.add_argument("--pretrained-tabular-dir", default=None)
    parser.add_argument("--pretrained-image-dir", default=None)

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
    if args.run_mode is not None:
        updates["run_mode"] = args.run_mode
    elif args.staged_training:
        updates["run_mode"] = "progressive"
    if args.stage1_epochs is not None:
        updates["stage1_epochs"] = args.stage1_epochs
    if args.stage2_epochs is not None:
        updates["stage2_epochs"] = args.stage2_epochs
    if args.stage3_freeze_epochs is not None:
        updates["stage3_freeze_epochs"] = args.stage3_freeze_epochs
    if args.stage3_unfreeze_epochs is not None:
        updates["stage3_unfreeze_epochs"] = args.stage3_unfreeze_epochs
    if args.pretrained_tabular_dir is not None:
        updates["pretrained_tabular_dir"] = args.pretrained_tabular_dir
    if args.pretrained_image_dir is not None:
        updates["pretrained_image_dir"] = args.pretrained_image_dir

    if updates:
        filtered_updates = _filter_supported_config_updates(config, updates)
        if filtered_updates:
            config = config.with_updates(**filtered_updates)

    run_mid_fusion_cv(config)


if __name__ == "__main__":
    main()
