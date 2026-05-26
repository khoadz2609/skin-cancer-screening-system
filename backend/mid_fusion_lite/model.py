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


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.tab_norm = nn.LayerNorm(hidden_dim)
        self.image_to_tab = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.tab_to_image = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, image_features: torch.Tensor, tab_features: torch.Tensor
    ) -> torch.Tensor:
        # image_features: (B, L, hidden_dim) where L is number of spatial patches
        # tab_features: (B, hidden_dim)
        tab_token = tab_features.unsqueeze(1)  # (B, 1, hidden_dim)

        norm_image = self.image_norm(image_features)
        norm_tab = self.tab_norm(tab_token)

        # Image patches attend to Tabular information
        image_ctx, _ = self.image_to_tab(
            query=norm_image,
            key=norm_tab,
            value=norm_tab,
            need_weights=False,
        )

        # Tabular information attends to Image patches
        tab_ctx, _ = self.tab_to_image(
            query=norm_tab,
            key=norm_image,
            value=norm_image,
            need_weights=False,
        )

        image_out = image_features + self.dropout(image_ctx)  # (B, L, D)
        tab_out = tab_token + self.dropout(tab_ctx)  # (B, 1, D)

        # Average pooling over spatial dimensions
        image_out_pooled = image_out.mean(dim=1)  # (B, D)
        tab_out_squeezed = tab_out.squeeze(1)  # (B, D)

        return torch.cat([image_out_pooled, tab_out_squeezed], dim=1)


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
        fusion_mode: str,
        cross_attention_heads: int,
        cross_attention_dropout: float,
        tab_dropout: float,
        fusion_dropout: float,
        train_image_backbone: bool = False,
        use_pretrained_backbone: bool = False,
        backbone_weights_path: str | None = None,
    ) -> None:
        super().__init__()

        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"concat", "cross_attention"}:
            raise ValueError("fusion_mode must be one of: 'concat', 'cross_attention'")

        self.image_encoder = self._build_image_encoder(
            backbone_name=backbone_name,
            use_pretrained_backbone=use_pretrained_backbone,
            backbone_weights_path=backbone_weights_path,
            fusion_mode=self.fusion_mode,
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

        if self.fusion_mode == "cross_attention":
            if fusion_hidden_dim % int(cross_attention_heads) != 0:
                raise ValueError(
                    "fusion_hidden_dim must be divisible by cross_attention_heads "
                    f"(got {fusion_hidden_dim} and {cross_attention_heads})."
                )
            self.tab_to_fusion = (
                nn.Identity()
                if tab_hidden_dim == fusion_hidden_dim
                else nn.Linear(tab_hidden_dim, fusion_hidden_dim)
            )
            self.cross_modal_fusion = CrossModalAttentionFusion(
                hidden_dim=fusion_hidden_dim,
                num_heads=int(cross_attention_heads),
                dropout=float(cross_attention_dropout),
            )
            fusion_in_dim = fusion_hidden_dim * 2
        else:
            self.tab_to_fusion = None
            self.cross_modal_fusion = None
            fusion_in_dim = fusion_hidden_dim + tab_hidden_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
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
        fusion_mode: str,
    ) -> nn.Module:
        create_kwargs = {
            "model_name": backbone_name,
            "num_classes": 0,
            "global_pool": "" if fusion_mode == "cross_attention" else "avg",
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

        if self.fusion_mode == "cross_attention":
            if image_features.ndim == 4:
                # image_features is (B, C, H, W)
                B, C, H, W = image_features.shape
                # Reshape to (B, H*W, C) for MultiheadAttention
                image_features = image_features.view(B, C, -1).transpose(1, 2)
            elif image_features.ndim == 3:
                # Already (B, L, C) from sequence-based backbones
                pass
            else:
                raise ValueError(
                    f"Unexpected image_features shape for cross_attention: {image_features.shape}"
                )

            image_features = self.image_head(
                image_features
            )  # (B, H*W or L, fusion_hidden_dim)
        else:
            if image_features.ndim > 2:
                image_features = torch.flatten(image_features, 1)
            image_features = self.image_head(image_features)

        tabular_features = self.tabular_encoder(tabular)
        if self.fusion_mode == "cross_attention":
            projected_tab = self.tab_to_fusion(tabular_features)
            fused = self.cross_modal_fusion(image_features, projected_tab)
        else:
            fused = torch.cat([image_features, tabular_features], dim=1)

        logits = self.fusion_head(fused).squeeze(1)
        return logits
