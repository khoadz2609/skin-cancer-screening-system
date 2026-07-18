from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
from torch import nn

try:
    from tabm import TabM, LinearEnsemble
except ImportError:
    TabM = None
    LinearEnsemble = None


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
        keep_ensemble: bool = True,
        cat_cardinalities: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if TabM is None:
            raise ImportError(
                "TABM encoder requires package 'tabm'. "
                "Install dependencies with: pip install tabm rtdl_num_embeddings"
            )

        # Categorical features (if any) are passed to TabM as integer codes via
        # cat_cardinalities (one-hot internally); only the leading numeric columns get
        # piecewise-linear embeddings. `bins` must therefore describe numeric features only.
        self.n_cat = len(cat_cardinalities) if cat_cardinalities else 0
        self.n_num = in_features - self.n_cat

        num_embeddings = None
        if bins is not None:
            import rtdl_num_embeddings

            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                bins,
                d_embedding=d_embedding,
                activation=False,
                version="B",
            )

        self.encoder = TabM.make(
            n_num_features=self.n_num,
            cat_cardinalities=cat_cardinalities if self.n_cat > 0 else None,
            d_out=None,
            arch_type=arch_type,
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            num_embeddings=num_embeddings,
        )
        self.output_dim = d_block
        self.keep_ensemble = keep_ensemble
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_cat > 0:
            x_num = x[:, : self.n_num]
            x_cat = x[:, self.n_num :].round().long()
            features = self.encoder(x_num, x_cat)
        else:
            features = self.encoder(x)

        if isinstance(features, tuple):
            features = features[0]

        # TabM returns (B, K, d_block) when K > 1 (d_out=None → backbone output)
        if features.ndim == 3:
            if self.keep_ensemble:
                return features  # Keep ensemble: [B, K, d_block]
            else:
                return features.mean(dim=1)  # Collapse: [B, d_block]

        if features.ndim == 2:
            return features

        raise RuntimeError(f"Unexpected TABM output shape: {tuple(features.shape)}")


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.tab_norm = nn.LayerNorm(hidden_dim)

        # Bottleneck: compress tabular features for efficient cross-modal fusion
        self.tab_bottleneck = nn.Linear(hidden_dim, hidden_dim)

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
        self,
        image_features: torch.Tensor,
        tab_features: torch.Tensor,
        image_keep_mask: Optional[torch.Tensor] = None,
        tab_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # image_features: (B, L, hidden_dim) where L is number of spatial patches
        # tab_features: (B, hidden_dim)
        batch_size = image_features.size(0)
        if image_keep_mask is None:
            image_keep_mask = image_features.new_ones((batch_size, 1))
        if tab_keep_mask is None:
            tab_keep_mask = tab_features.new_ones((batch_size, 1))

        image_keep_mask = image_keep_mask.to(
            device=image_features.device, dtype=image_features.dtype
        ).reshape(batch_size, 1, 1)
        tab_keep_mask = tab_keep_mask.to(
            device=tab_features.device, dtype=tab_features.dtype
        ).reshape(batch_size, 1, 1)
        interaction_mask = image_keep_mask * tab_keep_mask

        # Mask both the residual stream and the cross-modal context. Masking only
        # before the projection is insufficient because Linear/MHA biases can
        # otherwise recreate a non-zero representation for a missing modality.
        image_features = image_features * image_keep_mask
        tab_token = tab_features.unsqueeze(1) * tab_keep_mask  # (B, 1, hidden_dim)

        norm_image = self.image_norm(image_features)
        norm_tab = self.tab_norm(tab_token)

        # Bottleneck: create compressed representation of tabular info
        tab_bottleneck = self.tab_bottleneck(norm_tab)  # (B, 1, hidden_dim)

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

        image_out = image_features + interaction_mask * self.dropout(image_ctx)
        tab_out = tab_token + interaction_mask * self.dropout(tab_ctx)

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
            nn.Sigmoid(),
        )
        self.image_transform = nn.Linear(hidden_dim, hidden_dim)
        self.tab_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        image_features: torch.Tensor,
        tab_features: torch.Tensor,
        image_keep_mask: Optional[torch.Tensor] = None,
        tab_keep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = image_features.size(0)
        if image_keep_mask is None:
            image_keep_mask = image_features.new_ones((batch_size, 1))
        if tab_keep_mask is None:
            tab_keep_mask = tab_features.new_ones((batch_size, 1))

        image_keep_mask = image_keep_mask.to(
            device=image_features.device, dtype=image_features.dtype
        ).reshape(batch_size, 1)
        tab_keep_mask = tab_keep_mask.to(
            device=tab_features.device, dtype=tab_features.dtype
        ).reshape(batch_size, 1)

        masked_image = image_features * image_keep_mask
        masked_tab = tab_features * tab_keep_mask
        concat_features = torch.cat([masked_image, masked_tab], dim=1)
        z = self.image_gate(concat_features)  # (B, 1)

        # Apply the mask again after Linear so a learned bias cannot recreate a
        # representation for the missing branch.
        h_image = self.image_transform(masked_image) * image_keep_mask
        h_tab = self.tab_transform(masked_tab) * tab_keep_mask

        both_present = image_keep_mask * tab_keep_mask
        image_only = image_keep_mask * (1.0 - tab_keep_mask)
        tab_only = (1.0 - image_keep_mask) * tab_keep_mask

        # Use the learned GMU gate when both modalities exist. If one modality is
        # missing, fall back directly to the available transformed feature.
        fused = (
            both_present * (z * h_image + (1.0 - z) * h_tab)
            + image_only * h_image
            + tab_only * h_tab
        )
        return fused


class MidFusionLiteModel(nn.Module):
    def __init__(
        self,
        backbone_name: Optional[str] = None,
        n_tab_features: Optional[int] = None,
        use_tabm_tabular: Optional[bool] = None,
        tab_hidden_dim: Optional[int] = None,
        tabm_arch_type: Optional[str] = None,
        tabm_k: Optional[int] = None,
        tabm_n_blocks: Optional[int] = None,
        tabm_d_block: Optional[int] = None,
        fusion_hidden_dim: Optional[int] = None,
        fusion_mode: Optional[str] = None,
        cross_attention_heads: Optional[int] = None,
        cross_attention_dropout: Optional[float] = None,
        tab_dropout: Optional[float] = None,
        fusion_dropout: Optional[float] = None,
        train_image_backbone: bool = False,
        download_pretrained_backbone: bool = False,
        backbone_weights_path: str | None = None,
        modality_dropout_prob: Optional[float] = None,
        modality_mode: str = "both",
        bins: Optional[list[list[float]]] = None,
        d_embedding: int = 24,
        init_bias: Optional[float] = None,
        keep_tabm_ensemble: bool = True,
        cat_cardinalities: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        self.modality_mode = str(modality_mode).strip().lower()
        if self.modality_mode not in {"both", "image_only", "tabular_only"}:
            raise ValueError(
                "modality_mode must be: 'both', 'image_only', 'tabular_only'"
            )

        uses_image = self.modality_mode in {"both", "image_only"}
        uses_tabular = self.modality_mode in {"both", "tabular_only"}

        def require(name: str, value):
            if value is None:
                raise ValueError(
                    f"{name} is required when modality_mode='{self.modality_mode}'"
                )
            return value

        # fusion_mode controls spatial vs pooled image features, so it remains
        # relevant for image_only checkpoints. It is unused for tabular_only.
        resolved_fusion_mode = (
            require("fusion_mode", fusion_mode) if uses_image else (fusion_mode or "concat")
        )
        self.fusion_mode = str(resolved_fusion_mode).strip().lower()
        if self.fusion_mode not in {"concat", "cross_attention", "gmu"}:
            raise ValueError(
                "fusion_mode must be one of: 'concat', 'cross_attention', 'gmu'"
            )

        if self.modality_mode == "both":
            modality_dropout_prob = require(
                "modality_dropout_prob", modality_dropout_prob
            )
        self.modality_dropout_prob = float(modality_dropout_prob or 0.0)
        if not 0.0 <= self.modality_dropout_prob <= 1.0:
            raise ValueError("modality_dropout_prob must be between 0 and 1")
        self.train_image_backbone = train_image_backbone
        self.keep_tabm_ensemble = keep_tabm_ensemble
        self.tabm_k = 1

        # --- IMAGE BRANCH ---
        if uses_image:
            backbone_name = str(require("backbone_name", backbone_name))
            fusion_hidden_dim = int(require("fusion_hidden_dim", fusion_hidden_dim))
            fusion_dropout = float(require("fusion_dropout", fusion_dropout))
            self.image_encoder = self._build_image_encoder(
                backbone_name=backbone_name,
                download_pretrained_backbone=download_pretrained_backbone,
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
            self.image_only_head = nn.Linear(fusion_hidden_dim, 1)
        else:
            self.image_encoder = None
            self.image_head = None
            self.image_only_head = None

        # --- TABULAR BRANCH ---
        if uses_tabular:
            n_tab_features = int(require("n_tab_features", n_tab_features))
            use_tabm_tabular = bool(
                require("use_tabm_tabular", use_tabm_tabular)
            )
            tab_dropout = float(require("tab_dropout", tab_dropout))
            if use_tabm_tabular:
                tabm_arch_type = str(require("tabm_arch_type", tabm_arch_type))
                tabm_k = int(require("tabm_k", tabm_k))
                tabm_n_blocks = int(require("tabm_n_blocks", tabm_n_blocks))
                tabm_d_block = int(require("tabm_d_block", tabm_d_block))
                self.tabm_k = tabm_k
                self.tabular_encoder = TabMTabularEncoder(
                    in_features=n_tab_features,
                    # Kept for TabMTabularEncoder API compatibility; TabM uses
                    # tabm_d_block rather than the MLP-only hidden dimension.
                    hidden_dim=int(tab_hidden_dim or 0),
                    dropout=tab_dropout,
                    arch_type=tabm_arch_type,
                    k=tabm_k,
                    n_blocks=tabm_n_blocks,
                    d_block=tabm_d_block,
                    bins=bins,
                    d_embedding=d_embedding,
                    keep_ensemble=keep_tabm_ensemble,
                    cat_cardinalities=cat_cardinalities,
                )
                # With d_out=None inside TabM, the backbone outputs raw
                # (B, K, d_block) features. The prediction head below is
                # identical to what TabM uses internally when d_out=1.
                effective_tab_dim = self.tabular_encoder.output_dim
                if LinearEnsemble is not None and keep_tabm_ensemble and tabm_k > 1:
                    self.tabular_only_head = LinearEnsemble(
                        in_features=effective_tab_dim,
                        out_features=1,
                        bias=True,
                        k=tabm_k,
                    )
                else:
                    self.tabular_only_head = nn.Linear(effective_tab_dim, 1)
            else:
                tab_hidden_dim = int(require("tab_hidden_dim", tab_hidden_dim))
                self.tabular_encoder = TabularEncoder(
                    in_features=n_tab_features,
                    hidden_dim=tab_hidden_dim,
                    dropout=tab_dropout,
                )
                effective_tab_dim = tab_hidden_dim
                self.tabular_only_head = nn.Linear(effective_tab_dim, 1)
        else:
            self.tabular_encoder = None
            self.tabular_only_head = None
            effective_tab_dim = 0  # fallback, unused outside two-modality fusion

        # --- FUSION BRANCH ---
        if self.modality_mode == "both":
            # uses_image already resolved these values above.
            assert fusion_hidden_dim is not None
            assert fusion_dropout is not None
            if self.fusion_mode == "cross_attention":
                cross_attention_heads = int(
                    require("cross_attention_heads", cross_attention_heads)
                )
                cross_attention_dropout = float(
                    require("cross_attention_dropout", cross_attention_dropout)
                )
                if fusion_hidden_dim % int(cross_attention_heads) != 0:
                    raise ValueError(
                        "fusion_hidden_dim must be divisible by cross_attention_heads "
                        f"(got {fusion_hidden_dim} and {cross_attention_heads})."
                    )
                self.tab_to_fusion = (
                    nn.Identity()
                    if effective_tab_dim == fusion_hidden_dim
                    else nn.Linear(effective_tab_dim, fusion_hidden_dim)
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
                    if effective_tab_dim == fusion_hidden_dim
                    else nn.Linear(effective_tab_dim, fusion_hidden_dim)
                )
                self.cross_modal_fusion = GatedMultimodalUnit(
                    hidden_dim=fusion_hidden_dim,
                    dropout=fusion_dropout,
                )
                fusion_in_dim = fusion_hidden_dim
            else:
                self.tab_to_fusion = None
                self.cross_modal_fusion = None
                fusion_in_dim = fusion_hidden_dim + effective_tab_dim

            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_in_dim, fusion_hidden_dim),
                nn.BatchNorm1d(fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden_dim, 1),
            )
        else:
            self.tab_to_fusion = None
            self.cross_modal_fusion = None
            self.fusion_head = None

        # --- INIT WEIGHTS ---
        for m in [
            self.image_head,
            self.tabular_encoder,
            self.fusion_head,
            self.image_only_head,
            self.tabular_only_head,
        ]:
            if m is not None:
                self._init_custom_weights(m)
        if self.cross_modal_fusion is not None:
            self._init_custom_weights(self.cross_modal_fusion)
        if getattr(self, 'tab_to_fusion', None) is not None and isinstance(self.tab_to_fusion, nn.Linear):
            self._init_custom_weights(self.tab_to_fusion)

        if init_bias is not None:
            if self.image_only_head is not None:
                nn.init.constant_(self.image_only_head.bias, init_bias)
            if self.tabular_only_head is not None:
                if hasattr(self.tabular_only_head, "bias") and self.tabular_only_head.bias is not None:
                    nn.init.constant_(self.tabular_only_head.bias, init_bias)
            if self.fusion_head is not None:
                nn.init.constant_(self.fusion_head[-1].bias, init_bias)

    def _init_custom_weights(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _sample_modality_keep_masks(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-sample image/tabular keep masks with shape ``(B, 1)``.

        ``modality_dropout_prob`` is the total probability of dropping exactly
        one modality. Half of dropped samples lose image and half lose tabular;
        both modalities are never dropped together. Dropout is restricted to
        two-modality training, so evaluation and unimodal modes return all ones.
        """
        keep = torch.ones((batch_size, 1), device=device, dtype=dtype)
        if (
            not self.training
            or self.modality_mode != "both"
            or self.modality_dropout_prob <= 0.0
        ):
            return keep, keep.clone()

        outcomes = torch.rand((batch_size, 1), device=device)
        half_probability = self.modality_dropout_prob / 2.0
        drop_image = outcomes < half_probability
        drop_tabular = (outcomes >= half_probability) & (
            outcomes < self.modality_dropout_prob
        )
        image_keep = (~drop_image).to(dtype=dtype)
        tabular_keep = (~drop_tabular).to(dtype=dtype)
        return image_keep, tabular_keep

    def train(self, mode: bool = True) -> "MidFusionLiteModel":
        """Set training mode, but keep a frozen image backbone in eval mode.

        When the backbone is frozen (all of its parameters have
        ``requires_grad=False``) its BatchNorm layers must NOT keep updating their
        running statistics on the resampled / heavily-augmented training stream —
        ``torch.no_grad()`` only disables autograd, not BN stat tracking. Letting
        them drift makes the validation features (computed with the drifted stats)
        inconsistent with the pretrained backbone. This guard covers both
        construction-time freezing (``train_image_backbone=False``) and runtime
        freezing (e.g. Stage 3A, where the encoder is frozen to train the fusion
        head). Once the backbone is unfrozen (Stage 3B), it trains normally again.
        """
        super().train(mode)
        if self.image_encoder is not None:
            backbone_frozen = not any(
                p.requires_grad for p in self.image_encoder.parameters()
            )
            if backbone_frozen:
                self.image_encoder.eval()
        return self

    def load_unimodal_weights(
        self,
        tabular_state: dict[str, torch.Tensor] | None = None,
        image_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Load pretrained tabular and/or image branch weights from unimodal checkpoints."""
        if tabular_state is not None:
            filtered_tab_state = {}
            for k, v in tabular_state.items():
                if k.startswith("tabular_encoder."):
                    filtered_tab_state[k] = v
            missing, unexpected = self.load_state_dict(filtered_tab_state, strict=False)
            print(
                f"Loaded pretrained tabular weights | missing={len(missing)} | unexpected={len(unexpected)}"
            )

        if image_state is not None:
            filtered_img_state = {}
            # Check if this is a MidFusion state dict (contains 'image_encoder.') or standalone
            has_midfusion_prefix = any(k.startswith("image_encoder.") for k in image_state.keys())
            
            if has_midfusion_prefix:
                for k, v in image_state.items():
                    if (
                        k.startswith("image_encoder.")
                        or k.startswith("image_head.")
                        or k.startswith("image_only_head.")
                    ):
                        filtered_img_state[k] = v
            else:
                # Standalone timm model weights (e.g., from top solutions)
                print("Detected standalone vision checkpoint. Mapping keys to 'image_encoder.*'...")
                for k, v in image_state.items():
                    key = k
                    if key.startswith("module."):
                        key = key[len("module.") :]
                    if key.startswith("model."):
                        key = key[len("model.") :]
                    # Map to the image_encoder attribute of MidFusionLiteModel
                    filtered_img_state[f"image_encoder.{key}"] = v

            missing, unexpected = self.load_state_dict(filtered_img_state, strict=False)
            print(
                f"Loaded pretrained image weights | missing={len(missing)} | unexpected={len(unexpected)}"
            )

    @staticmethod
    def _build_image_encoder(
        backbone_name: str,
        download_pretrained_backbone: bool,
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

        if not download_pretrained_backbone:
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

        raw_obj = torch.load(path, map_location="cpu", weights_only=True)
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

    def forward(
        self, images: torch.Tensor, tabular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        
        # --- FAST PATHS FOR UNIMODAL ---
        if self.modality_mode == "tabular_only":
            tabular_features = self.tabular_encoder(tabular)
            tabular_ensemble_mode = tabular_features.ndim == 3 and self.keep_tabm_ensemble

            if tabular_ensemble_mode:
                if hasattr(self.tabular_only_head, "k"):
                    # LinearEnsemble output is (B, K, 1) -> squeeze to (B, K)
                    return self.tabular_only_head(tabular_features).squeeze(-1)
                else:
                    B, K, D = tabular_features.shape
                    tab_features_flat = tabular_features.reshape(-1, D)
                    tabular_logits_ensemble = self.tabular_only_head(tab_features_flat)
                    return tabular_logits_ensemble.reshape(B, K)
            else:
                return self.tabular_only_head(tabular_features).squeeze(1)

        elif self.modality_mode == "image_only":
            if not self.train_image_backbone:
                with torch.no_grad():
                    image_features = self.image_encoder(images)
            else:
                image_features = self.image_encoder(images)
            image_features = image_features.float()

            if self.fusion_mode == "cross_attention":
                if image_features.ndim == 4:
                    B, C, H, W = image_features.shape
                    image_features = image_features.view(B, C, -1).transpose(1, 2)
                image_features = self.image_head(image_features)
                image_pooled = image_features.mean(dim=1) if image_features.ndim == 3 else image_features
                return self.image_only_head(image_pooled).squeeze(1)
            else:
                if image_features.ndim > 2:
                    image_features = torch.flatten(image_features, 1)
                image_features = self.image_head(image_features)
                return self.image_only_head(image_features).squeeze(1)

        # --- BOTH MODALITIES (FUSION) ---
        # Image branch
        if not self.train_image_backbone:
            with torch.no_grad():
                image_features = self.image_encoder(images)
        else:
            image_features = self.image_encoder(images)
        image_features = image_features.float()

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

        # Tabular branch
        tabular_features = self.tabular_encoder(tabular)
        tabular_ensemble_mode = tabular_features.ndim == 3 and self.keep_tabm_ensemble

        # Auxiliary heads always receive clean unimodal features. Modality dropout
        # regularizes the fusion path only; otherwise a dropped branch would train
        # its auxiliary classifier mostly through its bias.
        if tabular_ensemble_mode:
            if hasattr(self.tabular_only_head, "k"):
                tabular_logits = self.tabular_only_head(tabular_features).squeeze(-1)
            else:
                B, K, D = tabular_features.shape
                tab_features_flat = tabular_features.reshape(-1, D)
                tabular_logits_ensemble = self.tabular_only_head(tab_features_flat)
                tabular_logits = tabular_logits_ensemble.reshape(B, K)
        else:
            tabular_logits = self.tabular_only_head(tabular_features).squeeze(1)

        image_keep_mask, tab_keep_mask = self._sample_modality_keep_masks(
            batch_size=image_features.size(0),
            device=image_features.device,
            dtype=image_features.dtype,
        )

        # Fusion
        if tabular_ensemble_mode:
            B, K, D_tab = tabular_features.shape
            D_img = image_features.shape[-1]
            image_mask_flat = image_keep_mask.repeat_interleave(K, dim=0)
            tab_mask_flat = tab_keep_mask.repeat_interleave(K, dim=0)

            if self.fusion_mode == "cross_attention":
                # image_features is 3D [B, seq_len, D_img] for cross_attention
                # Expand across K ensemble members -> [B, K, seq_len, D_img]
                S = image_features.shape[1]  # spatial / sequence length
                image_features_expanded = image_features.unsqueeze(1).expand(-1, K, -1, -1)
                img_for_fusion = image_features_expanded.reshape(B * K, S, D_img)

                tab_flat = tabular_features.reshape(B * K, D_tab)
                projected_tab = self.tab_to_fusion(tab_flat)

                fused = self.cross_modal_fusion(
                    img_for_fusion,
                    projected_tab,
                    image_keep_mask=image_mask_flat,
                    tab_keep_mask=tab_mask_flat,
                )
                fused = fused.reshape(B, K, -1)
            elif self.fusion_mode == "gmu":
                # image_features is 2D [B, D_img] for gmu
                image_features_expanded = image_features.unsqueeze(1).expand(-1, K, -1)
                tab_flat = tabular_features.reshape(B * K, D_tab)
                img_flat = image_features_expanded.reshape(B * K, D_img)
                projected_tab = self.tab_to_fusion(tab_flat)
                fused_flat = self.cross_modal_fusion(
                    img_flat,
                    projected_tab,
                    image_keep_mask=image_mask_flat,
                    tab_keep_mask=tab_mask_flat,
                )
                fused = fused_flat.reshape(B, K, -1)
            else:
                # image_features is 2D [B, D_img] for concat
                image_features_expanded = image_features.unsqueeze(1).expand(-1, K, -1)
                fused = torch.cat(
                    [
                        image_features_expanded * image_keep_mask.unsqueeze(1),
                        tabular_features * tab_keep_mask.unsqueeze(1),
                    ],
                    dim=-1,
                )

            B, K, D_fused = fused.shape
            fused_flat = fused.reshape(-1, D_fused)
            logits_ensemble = self.fusion_head(fused_flat)
            logits = logits_ensemble.reshape(B, K)
        else:
            if self.fusion_mode == "cross_attention":
                projected_tab = self.tab_to_fusion(tabular_features)
                fused = self.cross_modal_fusion(
                    image_features,
                    projected_tab,
                    image_keep_mask=image_keep_mask,
                    tab_keep_mask=tab_keep_mask,
                )
            elif self.fusion_mode == "gmu":
                projected_tab = self.tab_to_fusion(tabular_features)
                fused = self.cross_modal_fusion(
                    image_features,
                    projected_tab,
                    image_keep_mask=image_keep_mask,
                    tab_keep_mask=tab_keep_mask,
                )
            else:
                fused = torch.cat(
                    [
                        image_features * image_keep_mask,
                        tabular_features * tab_keep_mask,
                    ],
                    dim=1,
                )

            logits = self.fusion_head(fused).squeeze(1)

        if self.training:
            return logits, image_logits, tabular_logits
        return logits
