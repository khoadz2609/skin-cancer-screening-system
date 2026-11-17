from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def _state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _preferred_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(device: torch.device, use_amp: bool):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=_preferred_amp_dtype(device))
    return nullcontext()


def _build_grad_scaler(use_amp: bool, device: Optional[torch.device] = None):
    amp_dtype = _preferred_amp_dtype(device) if device is not None else torch.float16
    need_scaler = bool(use_amp and amp_dtype == torch.float16)
    try:
        return torch.amp.GradScaler(device="cuda", enabled=need_scaler)
    except Exception:
        try:
            return torch.amp.GradScaler("cuda", enabled=need_scaler)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=need_scaler)


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


def _build_criterion(pos_weight: Optional[float], device: torch.device) -> nn.Module:
    if pos_weight is not None and pos_weight > 0:
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    return nn.BCEWithLogitsLoss()


class SmoothPAUCLoss(nn.Module):
    """Pairwise margin surrogate focusing on high-scoring hard negatives (low-FPR regime)."""

    def __init__(self, margin: float = 0.2, top_neg_fraction: float = 0.25) -> None:
        super().__init__()
        self.margin = float(margin)
        self.top_neg_fraction = float(np.clip(top_neg_fraction, 0.01, 1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        if probs.ndim == 2:
            probs = probs.mean(dim=1)
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        if not torch.any(pos_mask) or not torch.any(neg_mask):
            return logits.sum() * 0.0

        pos_probs = probs[pos_mask]
        neg_probs = probs[neg_mask]
        k = max(1, int(math.ceil(neg_probs.numel() * self.top_neg_fraction)))
        hard_neg_probs, _ = torch.topk(neg_probs, k=min(k, neg_probs.numel()))

        diff = pos_probs.unsqueeze(1) - hard_neg_probs.unsqueeze(0)
        return torch.relu(self.margin - diff).pow(2).mean()


def _extract_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_f = logits.float()
    if isinstance(logits_f, torch.Tensor) and logits_f.ndim == 2:
        return torch.sigmoid(logits_f).mean(dim=1)
    return torch.sigmoid(logits_f)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: Any,
    max_steps: Optional[int] = None,
    aux_loss_image_weight: float = 0.5,
    aux_loss_tab_weight: float = 0.2,
    gradient_clipping_norm: Optional[float] = None,
    label_smoothing: float = 0.0,
    pauc_loss_weight: float = 0.0,
    pauc_criterion: Optional[nn.Module] = None,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0

    def _compute_loss_with_ensemble(logits, targets_loss):
        """Handle both regular [B] and ensemble [B, K] logits."""
        if logits.ndim == 2:  # [B, K] ensemble
            B, K = logits.shape
            logits_flat = logits.reshape(-1)  # [B*K]
            targets_flat = targets_loss.repeat_interleave(K)  # [B*K]
            return criterion(logits_flat, targets_flat)
        else:  # [B] regular
            return criterion(logits, targets_loss)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if label_smoothing > 0.0:
            targets_loss = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets_loss = targets

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device=device, use_amp=use_amp):
            logits = model(images, tabular)

            if isinstance(logits, tuple) and len(logits) == 3:
                fusion_logits, image_logits, tab_logits = logits

                # Compute losses with ensemble support
                loss_fusion = _compute_loss_with_ensemble(fusion_logits, targets_loss)
                loss_image = _compute_loss_with_ensemble(image_logits, targets_loss)
                loss_tab = _compute_loss_with_ensemble(tab_logits, targets_loss)

                loss = (
                    loss_fusion
                    + aux_loss_image_weight * loss_image
                    + aux_loss_tab_weight * loss_tab
                )
            else:
                fusion_logits = logits
                loss = _compute_loss_with_ensemble(logits, targets_loss)

            if pauc_criterion is not None and pauc_loss_weight > 0.0:
                loss = loss + float(pauc_loss_weight) * pauc_criterion(fusion_logits, targets)

        scaler.scale(loss).backward()

        if gradient_clipping_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

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

            # Handle ensemble logits [B, K] vs regular [B]
            if isinstance(logits, torch.Tensor) and logits.ndim == 2:
                # Loss: mean over the k independent predictions (matches training,
                # see _compute_loss_with_ensemble) — NOT loss of the averaged logit.
                K = logits.shape[1]
                loss = criterion(logits.reshape(-1), y.repeat_interleave(K))
                # Metric: average the k probabilities (TabM inference rule).
                prob = torch.sigmoid(logits).mean(dim=1)
            else:
                loss = criterion(logits, y)
                prob = torch.sigmoid(logits)

        preds.append(prob.float().detach().cpu().numpy())
        targets.append(y.float().detach().cpu().numpy())

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
    tta_steps: int = 1,
) -> np.ndarray:
    model.eval()
    predictions = []
    total_steps = len(loader)
    n_tta = max(1, int(tta_steps))

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)

        views = [images]
        if n_tta >= 2 and images.ndim == 4:
            views.append(torch.flip(images, dims=[3]))
        if n_tta >= 3 and images.ndim == 4:
            views.append(torch.flip(images, dims=[2]))
        if n_tta >= 4 and images.ndim == 4:
            views.append(images.transpose(2, 3).contiguous())

        prob_sum = None
        for view_img in views:
            with _autocast_context(device=device, use_amp=use_amp):
                logits = model(view_img, tabular)
            view_prob = _extract_probs_from_logits(logits)
            prob_sum = view_prob if prob_sum is None else (prob_sum + view_prob)

        prob = prob_sum / float(len(views))
        predictions.append(prob.float().detach().cpu().numpy())

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
    aux_loss_image_weight: float = 0.5,
    aux_loss_tab_weight: float = 0.2,
    gradient_clipping_norm: Optional[float] = None,
    label_smoothing: float = 0.0,
    epoch_callback: Optional[Callable[[int, float], None]] = None,
    pauc_loss_weight: float = 0.0,
    pauc_loss_margin: float = 0.2,
    pauc_loss_top_neg_fraction: float = 0.25,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], float]:
    use_amp = bool(use_amp and device.type == "cuda")
    criterion = _build_criterion(pos_weight=pos_weight, device=device)
    scaler = _build_grad_scaler(use_amp=use_amp, device=device)
    pauc_criterion = (
        SmoothPAUCLoss(
            margin=pauc_loss_margin,
            top_neg_fraction=pauc_loss_top_neg_fraction,
        ).to(device)
        if float(pauc_loss_weight) > 0.0
        else None
    )

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
            aux_loss_image_weight=aux_loss_image_weight,
            aux_loss_tab_weight=aux_loss_tab_weight,
            gradient_clipping_norm=gradient_clipping_norm,
            label_smoothing=label_smoothing,
            pauc_loss_weight=float(pauc_loss_weight),
            pauc_criterion=pauc_criterion,
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

        if epoch_callback is not None:
            epoch_callback(epoch, valid_pauc)

        if valid_pauc > best_score:
            best_score = valid_pauc
            best_state = _state_dict_to_cpu(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    return best_state, history, float(best_score)
