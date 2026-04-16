from __future__ import annotations

import timm
import torch
from torch import nn


class TabularEncoder(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MidFusionLiteModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        n_tab_features: int,
        tab_hidden_dim: int,
        fusion_hidden_dim: int,
        tab_dropout: float,
        fusion_dropout: float,
        train_image_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        if not train_image_backbone:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

        self.image_head = nn.Sequential(
            # LazyLinear infers the true backbone output width at first forward pass.
            # This avoids shape mismatches across timm backbones (e.g. MobileNetV3).
            nn.LazyLinear(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        self.tabular_encoder = TabularEncoder(
            in_features=n_tab_features,
            hidden_dim=tab_hidden_dim,
            dropout=tab_dropout,
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim + tab_hidden_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images)
        if image_features.ndim > 2:
            image_features = torch.flatten(image_features, 1)
        image_features = self.image_head(image_features)

        tabular_features = self.tabular_encoder(tabular)
        fused = torch.cat([image_features, tabular_features], dim=1)

        logits = self.fusion_head(fused).squeeze(1)
        return logits
