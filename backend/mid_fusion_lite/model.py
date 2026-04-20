from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

try:
    from tabm import TabM
except ImportError:
    TabM = None


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


class TabMTabularEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        dropout: float,
        arch_type: str,
        k: int,
        n_blocks: int,
        d_block: int,
    ) -> None:
        super().__init__()

        if TabM is None:
            raise ImportError(
                "TABM encoder requires package 'tabm'. "
                "Install dependencies with: pip install tabm rtdl_num_embeddings"
            )

        self.encoder = TabM.make(
            n_num_features=in_features,
            d_out=hidden_dim,
            arch_type=arch_type,
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        if isinstance(features, tuple):
            features = features[0]

        # TabM can return (B, k, d_out); use mean over ensemble axis for fusion.
        if features.ndim == 3:
            return features.mean(dim=1)

        if features.ndim == 2:
            return features

        raise RuntimeError(f"Unexpected TABM output shape: {tuple(features.shape)}")


class MidFusionLiteModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        n_tab_features: int,
        use_tabm_tabular: bool,
        tab_hidden_dim: int,
        tabm_arch_type: str,
        tabm_k: int,
        tabm_n_blocks: int,
        tabm_d_block: int,
        fusion_hidden_dim: int,
        tab_dropout: float,
        fusion_dropout: float,
        train_image_backbone: bool = False,
        use_pretrained_backbone: bool = False,
        backbone_weights_path: str | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = self._build_image_encoder(
            backbone_name=backbone_name,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone_weights_path=backbone_weights_path,
        )

        if not train_image_backbone:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False

        self.image_head = nn.Sequential(
            # LazyLinear infers the true backbone output width at first forward pass.
            # This avoids shape mismatches across timm backbones.
            nn.LazyLinear(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )

        if use_tabm_tabular:
            self.tabular_encoder = TabMTabularEncoder(
                in_features=n_tab_features,
                hidden_dim=tab_hidden_dim,
                dropout=tab_dropout,
                arch_type=tabm_arch_type,
                k=tabm_k,
                n_blocks=tabm_n_blocks,
                d_block=tabm_d_block,
            )
        else:
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

    @staticmethod
    def _build_image_encoder(
        backbone_name: str,
        use_pretrained_backbone: bool,
        backbone_weights_path: str | None,
    ) -> nn.Module:
        create_kwargs = {
            "model_name": backbone_name,
            "num_classes": 0,
            "global_pool": "avg",
        }

        encoder = timm.create_model(pretrained=False, **create_kwargs)

        if backbone_weights_path:
            MidFusionLiteModel._load_backbone_weights(
                encoder=encoder,
                weights_path=backbone_weights_path,
            )
            return encoder

        if not use_pretrained_backbone:
            return encoder

        try:
            return timm.create_model(pretrained=True, **create_kwargs)
        except Exception as exc:
            print(
                "Warning: failed to download pretrained weights. "
                "Falling back to pretrained=False."
            )
            print(f"Details: {exc}")
            return encoder

    @staticmethod
    def _load_backbone_weights(encoder: nn.Module, weights_path: str) -> None:
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Backbone weights file not found: {weights_path}")

        raw_obj = torch.load(path, map_location="cpu")
        if isinstance(raw_obj, dict):
            if "state_dict" in raw_obj and isinstance(raw_obj["state_dict"], dict):
                state_dict = raw_obj["state_dict"]
            elif "model" in raw_obj and isinstance(raw_obj["model"], dict):
                state_dict = raw_obj["model"]
            else:
                state_dict = raw_obj
        else:
            raise ValueError(
                "Unsupported checkpoint format. Expected a dict-like state_dict."
            )

        normalized: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("image_encoder."):
                key = key[len("image_encoder.") :]
            if key.startswith("model."):
                key = key[len("model.") :]
            normalized[key] = value

        missing, unexpected = encoder.load_state_dict(normalized, strict=False)
        print(
            "Loaded local backbone weights:",
            path,
            f"| missing={len(missing)} | unexpected={len(unexpected)}",
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
