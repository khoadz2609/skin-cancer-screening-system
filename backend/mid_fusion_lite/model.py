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
        bins: Optional[list[list[float]]] = None,
        d_embedding: int = 24,
    ) -> None:
        super().__init__()

        if TabM is None:
            raise ImportError(
                "TABM encoder requires package 'tabm'. "
                "Install dependencies with: pip install tabm rtdl_num_embeddings"
            )

        num_embeddings = None
        if bins is not None:
            import rtdl_num_embeddings
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins,
                d_embedding=d_embedding,
                activation=False,
                version='B',
            )

        self.encoder = TabM.make(
            n_num_features=in_features,
            d_out=hidden_dim,
            arch_type=arch_type,
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            num_embeddings=num_embeddings,
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

        # Bottleneck: compress image patches for efficient cross-modal fusion
        self.image_bottleneck = nn.Linear(hidden_dim, hidden_dim)

        # Image patches attend to tabular bottleneck (weak direction)
        self.image_to_tab = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Tabular attends to all image patches (strong direction)
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

        # Bottleneck: create compressed representation of tabular info
        tab_bottleneck = self.image_bottleneck(norm_tab)  # (B, 1, hidden_dim)

        # Image patches attend to compressed tabular bottleneck (weak direction)
        image_ctx, _ = self.image_to_tab(
            query=norm_image,
            key=tab_bottleneck,
            value=tab_bottleneck,
            need_weights=False,
        )

        # Tabular attends to all image patches (strong direction)
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


class GatedMultimodalUnit(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.image_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.image_transform = nn.Linear(hidden_dim, hidden_dim)
        self.tab_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, image_features: torch.Tensor, tab_features: torch.Tensor) -> torch.Tensor:
        concat_features = torch.cat([image_features, tab_features], dim=1)
        z = self.image_gate(concat_features) # (B, 1)
        
        h_image = self.image_transform(image_features)
        h_tab = self.tab_transform(tab_features)
        
        # GMU equation
        fused = z * h_image + (1.0 - z) * h_tab
        return fused


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
        modality_dropout_prob: float = 0.15,
        modality_mode: str = "both",
        bins: Optional[list[list[float]]] = None,
        d_embedding: int = 24,
    ) -> None:
        super().__init__()

        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"concat", "cross_attention", "gmu"}:
            raise ValueError("fusion_mode must be one of: 'concat', 'cross_attention', 'gmu'")

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
            nn.Linear(self.image_encoder.num_features, fusion_hidden_dim),
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
                bins=bins,
                d_embedding=d_embedding,
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
        elif self.fusion_mode == "gmu":
            self.tab_to_fusion = (
                nn.Identity()
                if tab_hidden_dim == fusion_hidden_dim
                else nn.Linear(tab_hidden_dim, fusion_hidden_dim)
            )
            self.cross_modal_fusion = GatedMultimodalUnit(
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout,
            )
            fusion_in_dim = fusion_hidden_dim
        else:
            self.tab_to_fusion = None
            self.cross_modal_fusion = None
            fusion_in_dim = fusion_hidden_dim + tab_hidden_dim

        self.image_only_head = nn.Linear(fusion_hidden_dim, 1)
        self.tabular_only_head = nn.Linear(tab_hidden_dim, 1)
        self.modality_dropout_prob = modality_dropout_prob
        self.modality_mode = str(modality_mode).strip().lower()
        if self.modality_mode not in {"both", "image_only", "tabular_only"}:
            raise ValueError("modality_mode must be: 'both', 'image_only', 'tabular_only'")


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

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        # --- IMAGE BRANCH ---
        # Skip image encoder if tabular_only to save massive computation time, 
        # but only AFTER LazyLinear has been initialized (weight is no longer UninitializedParameter)
        is_lazy_init = not isinstance(self.image_head[0].weight, torch.nn.parameter.UninitializedParameter)
        
        if self.modality_mode == "tabular_only" and is_lazy_init:
            B = images.size(0)
            fusion_dim = self.image_head[0].out_features
            
            if self.fusion_mode == "cross_attention":
                image_features = torch.zeros(B, 1, fusion_dim, device=images.device)
            else:
                image_features = torch.zeros(B, fusion_dim, device=images.device)
            image_logits = torch.zeros(B, device=images.device)
        else:
            image_features = self.image_encoder(images)

            if self.fusion_mode == "cross_attention":
                if image_features.ndim == 4:
                    B, C, H, W = image_features.shape
                    image_features = image_features.view(B, C, -1).transpose(1, 2)
                image_features = self.image_head(image_features)
                
                image_pooled = image_features.mean(dim=1) if image_features.ndim == 3 else image_features
                image_logits = self.image_only_head(image_pooled).squeeze(1)
            else:
                if image_features.ndim > 2:
                    image_features = torch.flatten(image_features, 1)
                image_features = self.image_head(image_features)
                image_logits = self.image_only_head(image_features).squeeze(1)

        # --- TABULAR BRANCH ---
        tabular_features = self.tabular_encoder(tabular)
        
        # Modality dropout for tabular data during training
        if self.training and torch.rand(1).item() < self.modality_dropout_prob:
            tabular_features = torch.zeros_like(tabular_features)
            
        tabular_logits = self.tabular_only_head(tabular_features).squeeze(1)

        # Modality isolation: zero out the inactive branch
        if self.modality_mode == "tabular_only":
            image_features = torch.zeros_like(image_features) if image_features.ndim == 2 else torch.zeros_like(image_features)
        elif self.modality_mode == "image_only":
            tabular_features = torch.zeros_like(tabular_features)

        if self.fusion_mode == "cross_attention":
            projected_tab = self.tab_to_fusion(tabular_features)
            fused = self.cross_modal_fusion(image_features, projected_tab)
        elif self.fusion_mode == "gmu":
            projected_tab = self.tab_to_fusion(tabular_features)
            fused = self.cross_modal_fusion(image_features, projected_tab)
        else:
            fused = torch.cat([image_features, tabular_features], dim=1)

        logits = self.fusion_head(fused).squeeze(1)
        
        if self.training:
            return logits, image_logits, tabular_logits
        return logits
