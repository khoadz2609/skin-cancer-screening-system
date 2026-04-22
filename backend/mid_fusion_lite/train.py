from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _build_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler(device="cuda", enabled=use_amp)
    except Exception:
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=use_amp)


def partial_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    min_tpr = 0.80
    max_fpr = abs(1.0 - min_tpr)

    v_gt = np.abs(y_true - 1.0)
    v_pred = 1.0 - y_pred

    try:
        partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    except ValueError:
        return float("nan")

    partial_auc = 0.5 * max_fpr**2 + (
        (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    )
    return float(partial_auc)


def _build_criterion(pos_weight: float, device: torch.device) -> nn.Module:
    weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: Any,
    max_steps: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device=device, use_amp=use_amp):
            logits = model(images, tabular)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        seen += batch_size

        if max_steps is not None and (step + 1) >= max_steps:
            break

    if seen == 0:
        return float("nan")
    return total_loss / seen


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_steps: Optional[int] = None,
) -> dict[str, float | np.ndarray]:
    model.eval()

    total_loss = 0.0
    seen = 0
    preds = []
    targets = []

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with _autocast_context(device=device, use_amp=use_amp):
            logits = model(images, tabular)
            loss = criterion(logits, y)

        prob = torch.sigmoid(logits)

        preds.append(prob.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

        batch_size = y.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        seen += batch_size

        if max_steps is not None and (step + 1) >= max_steps:
            break

    if seen == 0:
        return {
            "loss": float("nan"),
            "auroc": float("nan"),
            "pauc": float("nan"),
            "preds": np.array([], dtype=np.float32),
            "targets": np.array([], dtype=np.float32),
        }

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    try:
        auroc = float(roc_auc_score(y_true, y_pred))
    except ValueError:
        auroc = float("nan")

    pauc = partial_auc_score(y_true, y_pred)

    return {
        "loss": total_loss / seen,
        "auroc": auroc,
        "pauc": pauc,
        "preds": y_pred,
        "targets": y_true,
    }


@torch.inference_mode()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    progress_name: Optional[str] = None,
    log_interval: int = 0,
) -> np.ndarray:
    model.eval()
    predictions = []
    total_steps = len(loader)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)

        with _autocast_context(device=device, use_amp=use_amp):
            logits = model(images, tabular)

        prob = torch.sigmoid(logits)
        predictions.append(prob.detach().cpu().numpy())

        if progress_name and log_interval > 0:
            if step == 1 or step % log_interval == 0 or step == total_steps:
                print(f"[{progress_name}] predict step {step}/{total_steps}")

    if not predictions:
        return np.array([], dtype=np.float32)
    return np.concatenate(predictions)


def fit_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int,
    patience: int,
    pos_weight: float,
    use_amp: bool,
    max_train_steps_per_epoch: Optional[int] = None,
    max_valid_steps: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float]:
    use_amp = bool(use_amp and device.type == "cuda")
    criterion = _build_criterion(pos_weight=pos_weight, device=device)
    scaler = _build_grad_scaler(use_amp=use_amp)

    best_score = -np.inf
    best_state = _state_dict_to_cpu(model)
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            max_steps=max_train_steps_per_epoch,
        )

        valid_stats = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_steps=max_valid_steps,
        )

        if scheduler is not None:
            scheduler.step()

        valid_loss = float(valid_stats["loss"])
        valid_auroc = float(valid_stats["auroc"])
        valid_pauc = float(valid_stats["pauc"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "valid_loss": valid_loss,
                "valid_auroc": valid_auroc,
                "valid_pauc": valid_pauc,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.5f} | "
            f"valid_loss={valid_loss:.5f} | "
            f"valid_auroc={valid_auroc:.5f} | "
            f"valid_pauc={valid_pauc:.5f}"
        )

        if valid_pauc > best_score:
            best_score = valid_pauc
            best_state = _state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > patience:
                print("Early stopping triggered.")
                break

    return best_state, history, float(best_score)
