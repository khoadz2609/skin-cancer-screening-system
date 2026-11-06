"""Smoke test: Grad-CAM overlay generation on a randomly initialised fusion model."""
from __future__ import annotations

import torch
from PIL import Image

from backend.gradcam_utils import generate_gradcam_overlay
from backend.mid_fusion_lite.model import MidFusionLiteModel

MODEL_KWARGS = {
    "modality_mode": "both",
    "backbone_name": "efficientnet_b0",
    "train_image_backbone": False,
    "use_tabm_tabular": False,
    "n_tab_features": 610,
    "tab_hidden_dim": 128,
    "tabm_arch_type": "model1",
    "tabm_k": 32,
    "tabm_n_blocks": 2,
    "tabm_d_block": 32,
    "fusion_mode": "cross_attention",
    "fusion_hidden_dim": 256,
    "cross_attention_heads": 8,
    "cross_attention_dropout": 0.1,
    "tab_dropout": 0.1,
    "fusion_dropout": 0.1,
    "modality_dropout_prob": 0.1,
    "keep_tabm_ensemble": True,
    "download_pretrained_backbone": False,
}


def test_gradcam_overlay_returns_image() -> None:
    torch.manual_seed(0)
    model = MidFusionLiteModel(**MODEL_KWARGS)
    model.eval()

    original = Image.new("RGB", (224, 224))
    overlay = generate_gradcam_overlay(
        model,
        torch.randn(1, 3, 224, 224),
        torch.randn(1, 610),
        original,
    )

    assert isinstance(overlay, Image.Image)
    assert overlay.size == original.size
