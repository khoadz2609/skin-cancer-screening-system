from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import timm
import torch
from torch import nn

try:
    from .config import CLINICAL_TOKEN_GROUP_NAMES, build_clinical_feature_groups
except ImportError:
    from config import CLINICAL_TOKEN_GROUP_NAMES, build_clinical_feature_groups

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

    def forward_with_embeddings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run TabM and return (backbone_features, numeric_embeddings, cat_embeddings).

        Exposes TabM's per-feature numeric embeddings without altering module state_dict keys,
        so pretrained Stage-1 TabM checkpoints load with zero missing/unexpected keys.
        """
        if self.n_cat > 0:
            x_num = x[:, : self.n_num]
            x_cat = x[:, self.n_num :].round().long()
        else:
            x_num = x
            x_cat = None

        num_module = getattr(self.encoder, "num_module", None)
        cat_module = getattr(self.encoder, "cat_module", None)

        if (
            num_module is not None
            or cat_module is not None
        ) and hasattr(self.encoder, "ensemble_view") and hasattr(self.encoder, "backbone"):
            emb_num = num_module(x_num) if num_module is not None else x_num.unsqueeze(-1)
            if emb_num.ndim == 2:
                emb_num = emb_num.unsqueeze(-1)
            chunks = [emb_num.flatten(1, -1)]
            emb_cat = None
            if x_cat is not None and cat_module is not None:
                emb_cat = cat_module(x_cat)
                chunks.append(emb_cat.flatten(1, -1))
            stacked = torch.column_stack(chunks)
            features = self.encoder.backbone(self.encoder.ensemble_view(stacked))
        else:
            emb_num = x_num.unsqueeze(-1)
            emb_cat = None
            if self.n_cat > 0:
                features = self.encoder(x_num, x_cat)
            else:
                features = self.encoder(x)

        if isinstance(features, tuple):
            features = features[0]

        if features.ndim == 3:
            if self.keep_ensemble:
                return features, emb_num, emb_cat
            return features.mean(dim=1), emb_num, emb_cat

        if features.ndim == 2:
            return features, emb_num, emb_cat

        raise RuntimeError(f"Unexpected TABM output shape: {tuple(features.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _, _ = self.forward_with_embeddings(x)
        return features


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


class SemanticGroupTokenizer(nn.Module):
    """Tokenize tabular inputs & TabM global context into M=7 clinical semantic tokens.

    Operates on per-feature embeddings (e.g. TabM PiecewiseLinearEmbeddings output of
    shape [B, N_num, D_emb]) plus TabM's global backbone summary without reordering
    columns or altering TabM's checkpoint keys.
    """

    def __init__(
        self,
        n_num: int,
        d_num_embedding: int,
        hidden_dim: int,
        global_tab_dim: int,
        dropout: float,
        numeric_cols: Optional[list[str]] = None,
        categorical_cols: Optional[list[str]] = None,
        cat_emb_dim: int = 0,
        feature_token_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_num = int(n_num)
        self.d_num_embedding = int(d_num_embedding)
        self.hidden_dim = int(hidden_dim)
        self.feature_token_dim = int(feature_token_dim)
        self.group_names = list(CLINICAL_TOKEN_GROUP_NAMES[:-1])
        self.token_names = list(CLINICAL_TOKEN_GROUP_NAMES)

        groups = build_clinical_feature_groups(
            numeric_cols=numeric_cols,
            n_numeric=self.n_num,
            categorical_cols=categorical_cols,
        )
        for idx, name in enumerate(self.group_names):
            col_indices = torch.tensor(groups[name], dtype=torch.long)
            self.register_buffer(f"_group_idx_{idx}", col_indices, persistent=False)

        self.num_feat_proj = nn.Linear(self.d_num_embedding, self.feature_token_dim)
        self.feat_id_emb = nn.Parameter(
            torch.zeros(1, max(1, self.n_num), self.feature_token_dim)
        )
        nn.init.normal_(self.feat_id_emb, std=0.02)
        self.feat_act = nn.Sequential(
            nn.LayerNorm(self.feature_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        scorer_hidden = max(4, self.feature_token_dim // 2)
        self.group_attn_scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_token_dim, scorer_hidden),
                    nn.GELU(),
                    nn.Linear(scorer_hidden, 1),
                )
                for _ in self.group_names
            ]
        )
        self.group_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_token_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in self.group_names
            ]
        )
        self.empty_group_tokens = nn.Parameter(
            torch.zeros(len(self.group_names), hidden_dim)
        )
        nn.init.normal_(self.empty_group_tokens, std=0.02)

        self.cat_emb_proj = (
            nn.Sequential(
                nn.Linear(cat_emb_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if cat_emb_dim > 0
            else None
        )
        self.global_tab_proj = nn.Sequential(
            nn.Linear(global_tab_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_type_emb = nn.Parameter(
            torch.zeros(1, len(self.token_names), hidden_dim)
        )
        nn.init.normal_(self.token_type_emb, std=0.02)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        emb_num: torch.Tensor,
        global_tab: torch.Tensor,
        emb_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = global_tab.size(0)
        if self.n_num > 0:
            h_feats = self.num_feat_proj(emb_num) + self.feat_id_emb[:, : self.n_num, :]
            h_feats = self.feat_act(h_feats)
        else:
            h_feats = None

        group_tokens: list[torch.Tensor] = []
        anatomy_idx = self.group_names.index("anatomy_demographics")

        for g_idx in range(len(self.group_names)):
            col_indices: torch.Tensor = getattr(self, f"_group_idx_{g_idx}")
            if h_feats is not None and col_indices.numel() > 0:
                h_g = h_feats[:, col_indices, :]
                scores = self.group_attn_scorers[g_idx](h_g)
                weights = torch.softmax(scores, dim=1)
                h_attn = (weights * h_g).sum(dim=1)
                h_max = h_g.amax(dim=1)
                tok_g = self.group_projs[g_idx](torch.cat([h_attn, h_max], dim=-1))
            else:
                tok_g = self.empty_group_tokens[g_idx].unsqueeze(0).expand(batch_size, -1)

            if g_idx == anatomy_idx and emb_cat is not None and self.cat_emb_proj is not None:
                tok_g = tok_g + self.cat_emb_proj(emb_cat.float())

            group_tokens.append(tok_g)

        global_token = self.global_tab_proj(global_tab)
        tokens = torch.stack(group_tokens + [global_token], dim=1)
        return self.out_norm(tokens + self.token_type_emb)


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal spatial positional encoding for flattened H*W image patch sequences."""

    def __init__(self, hidden_dim: int, scale: float = 0.2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(
        self, x: torch.Tensor, spatial_hw: Optional[tuple[int, int]] = None
    ) -> torch.Tensor:
        _batch_size, seq_len, dim = x.shape
        if seq_len <= 1 or dim < 4:
            return x
        if spatial_hw is None:
            side = int(math.isqrt(seq_len))
            if side * side == seq_len:
                h, w = side, side
            else:
                h, w = 1, seq_len
        else:
            h, w = spatial_hw

        quarter_dim = max(1, dim // 4)
        device = x.device
        dtype = x.dtype
        div_term = torch.exp(
            torch.arange(quarter_dim, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, quarter_dim))
        )

        y_pos = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=torch.float32)
        x_pos = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing="ij")

        ang_y = (grid_y.reshape(-1, 1) * math.pi) * div_term.unsqueeze(0)
        ang_x = (grid_x.reshape(-1, 1) * math.pi) * div_term.unsqueeze(0)

        pos = torch.cat(
            [torch.sin(ang_y), torch.cos(ang_y), torch.sin(ang_x), torch.cos(ang_x)],
            dim=-1,
        ).to(dtype=dtype)
        if pos.size(-1) < dim:
            pos = torch.nn.functional.pad(pos, (0, dim - pos.size(-1)))
        elif pos.size(-1) > dim:
            pos = pos[:, :dim]
        return x + self.scale * pos.unsqueeze(0)


class _CoAttentionLayer(nn.Module):
    """One layer of symmetric bidirectional cross-attention + intra-tabular self-attn + FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        self.img_norm1 = nn.LayerNorm(hidden_dim)
        self.tab_norm1 = nn.LayerNorm(hidden_dim)
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
        self.tab_self_norm = nn.LayerNorm(hidden_dim)
        self.tab_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        ffn_dim = hidden_dim * max(1, int(ffn_mult))
        self.img_norm2 = nn.LayerNorm(hidden_dim)
        self.img_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.tab_norm2 = nn.LayerNorm(hidden_dim)
        self.tab_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        img_tokens: torch.Tensor,
        tab_tokens: torch.Tensor,
        image_keep_mask: torch.Tensor,
        tab_keep_mask: torch.Tensor,
        interaction_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Mask after LayerNorm as well so LayerNorm bias cannot recreate missing modalities.
        norm_img = self.img_norm1(img_tokens) * image_keep_mask
        norm_tab = self.tab_norm1(tab_tokens) * tab_keep_mask

        img_ctx, img_to_tab_attn = self.image_to_tab(
            query=norm_img,
            key=norm_tab,
            value=norm_tab,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        tab_ctx, tab_to_img_attn = self.tab_to_image(
            query=norm_tab,
            key=norm_img,
            value=norm_img,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        img_tokens = (img_tokens + interaction_mask * self.drop(img_ctx)) * image_keep_mask
        tab_tokens = (tab_tokens + interaction_mask * self.drop(tab_ctx)) * tab_keep_mask

        norm_tab_self = self.tab_self_norm(tab_tokens) * tab_keep_mask
        tab_self_ctx, _ = self.tab_self_attn(
            query=norm_tab_self,
            key=norm_tab_self,
            value=norm_tab_self,
            need_weights=False,
        )
        tab_tokens = (tab_tokens + tab_keep_mask * self.drop(tab_self_ctx)) * tab_keep_mask

        img_ffn_in = self.img_norm2(img_tokens) * image_keep_mask
        img_tokens = (img_tokens + image_keep_mask * self.drop(self.img_ffn(img_ffn_in))) * image_keep_mask

        tab_ffn_in = self.tab_norm2(tab_tokens) * tab_keep_mask
        tab_tokens = (tab_tokens + tab_keep_mask * self.drop(self.tab_ffn(tab_ffn_in))) * tab_keep_mask

        return img_tokens, tab_tokens, img_to_tab_attn, tab_to_img_attn


class BidirectionalCoAttentionFusion(nn.Module):
    """Multi-layer bidirectional co-attention between spatial image patches and semantic tabular tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int = 2,
        ffn_mult: int = 2,
        use_2d_pos_emb: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pos_emb = PositionalEncoding2D(hidden_dim) if use_2d_pos_emb else None
        self.layers = nn.ModuleList(
            [
                _CoAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        pool_hidden = max(4, hidden_dim // 2)
        self.img_pool_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pool_hidden),
            nn.GELU(),
            nn.Linear(pool_hidden, 1),
        )
        self.tab_pool_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pool_hidden),
            nn.GELU(),
            nn.Linear(pool_hidden, 1),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        tab_tokens: torch.Tensor,
        image_keep_mask: Optional[torch.Tensor] = None,
        tab_keep_mask: Optional[torch.Tensor] = None,
        spatial_hw: Optional[tuple[int, int]] = None,
        return_attention: bool = False,
        return_tokens: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]
    ):
        batch_size = image_features.size(0)
        if image_keep_mask is None:
            image_keep_mask = image_features.new_ones((batch_size, 1))
        if tab_keep_mask is None:
            tab_keep_mask = tab_tokens.new_ones((batch_size, 1))

        img_mask_3d = image_keep_mask.to(
            device=image_features.device, dtype=image_features.dtype
        ).reshape(batch_size, 1, 1)
        tab_mask_3d = tab_keep_mask.to(
            device=tab_tokens.device, dtype=tab_tokens.dtype
        ).reshape(batch_size, 1, 1)
        interaction_mask = img_mask_3d * tab_mask_3d

        # Add 2D spatial coords before masking so masked image branches stay strictly zero.
        if self.pos_emb is not None:
            image_features = self.pos_emb(image_features, spatial_hw=spatial_hw)

        img_tokens = image_features * img_mask_3d
        tab_state = tab_tokens * tab_mask_3d

        last_i2t: Optional[torch.Tensor] = None
        last_t2i: Optional[torch.Tensor] = None
        for layer in self.layers:
            img_tokens, tab_state, i2t_w, t2i_w = layer(
                img_tokens=img_tokens,
                tab_tokens=tab_state,
                image_keep_mask=img_mask_3d,
                tab_keep_mask=tab_mask_3d,
                interaction_mask=interaction_mask,
                return_attention=return_attention,
            )
            if return_attention:
                last_i2t = i2t_w
                last_t2i = t2i_w

        img_attn_w = torch.softmax(self.img_pool_score(img_tokens), dim=1)
        tab_attn_w = torch.softmax(self.tab_pool_score(tab_state), dim=1)

        img_mean = img_tokens.mean(dim=1)
        img_pooled = (
            0.5 * img_mean + 0.5 * (img_attn_w * img_tokens).sum(dim=1)
        ) * img_mask_3d.squeeze(-1)
        tab_pooled = (tab_attn_w * tab_state).sum(dim=1) * tab_mask_3d.squeeze(-1)

        fused = torch.cat([img_pooled, tab_pooled], dim=1)
        if not return_attention:
            if return_tokens:
                return fused, img_tokens, None
            return fused

        attn_maps: dict[str, torch.Tensor] = {
            "img_pool_weights": img_attn_w.squeeze(-1),
            "tab_pool_weights": tab_attn_w.squeeze(-1),
        }
        if last_i2t is not None:
            attn_maps["image_to_tab"] = last_i2t
        if last_t2i is not None:
            attn_maps["tab_to_image"] = last_t2i
        if return_tokens:
            return fused, img_tokens, attn_maps
        return fused, attn_maps


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
        fusion_num_layers: Optional[int] = None,
        fusion_ffn_mult: Optional[int] = None,
        use_2d_pos_emb: bool = True,
        tabular_numeric_cols: Optional[list[str]] = None,
        tabular_categorical_cols: Optional[list[str]] = None,
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
        if self.fusion_mode not in {
            "concat",
            "cross_attention",
            "gmu",
            "bidirectional_co_attention",
        }:
            raise ValueError(
                "fusion_mode must be one of: 'concat', 'cross_attention', "
                "'gmu', 'bidirectional_co_attention'"
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
        self.tab_tokenizer: Optional[SemanticGroupTokenizer] = None
        self.bidir_film: Optional[nn.Linear] = None
        if self.modality_mode == "both":
            # uses_image already resolved these values above.
            assert fusion_hidden_dim is not None
            assert fusion_dropout is not None
            if self.fusion_mode == "bidirectional_co_attention":
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
                n_cat = len(cat_cardinalities) if (uses_tabular and use_tabm_tabular and cat_cardinalities) else 0
                n_num = int(n_tab_features) - n_cat
                d_num_emb = int(d_embedding) if (uses_tabular and use_tabm_tabular and bins is not None) else 1
                cat_emb_dim = int(sum(cat_cardinalities)) if n_cat > 0 and cat_cardinalities else 0
                self.tab_tokenizer = SemanticGroupTokenizer(
                    n_num=max(0, n_num),
                    d_num_embedding=d_num_emb,
                    hidden_dim=fusion_hidden_dim,
                    global_tab_dim=effective_tab_dim,
                    dropout=fusion_dropout,
                    numeric_cols=tabular_numeric_cols,
                    categorical_cols=tabular_categorical_cols,
                    cat_emb_dim=cat_emb_dim,
                )
                self.tab_to_fusion = None
                self.cross_modal_fusion = BidirectionalCoAttentionFusion(
                    hidden_dim=fusion_hidden_dim,
                    num_heads=int(cross_attention_heads),
                    dropout=float(cross_attention_dropout),
                    num_layers=int(fusion_num_layers or 2),
                    ffn_mult=int(fusion_ffn_mult or 2),
                    use_2d_pos_emb=bool(use_2d_pos_emb),
                )
                self.bidir_film = nn.Linear(fusion_hidden_dim * 2, effective_tab_dim * 2)
                self.member_to_fusion = (
                    nn.Identity()
                    if effective_tab_dim == fusion_hidden_dim
                    else nn.Linear(effective_tab_dim, fusion_hidden_dim)
                )
                self.member_q_norm = nn.LayerNorm(fusion_hidden_dim)
                self.member_kv_norm = nn.LayerNorm(fusion_hidden_dim)
                self.member_to_image = nn.MultiheadAttention(
                    embed_dim=fusion_hidden_dim,
                    num_heads=int(cross_attention_heads),
                    dropout=float(cross_attention_dropout),
                    batch_first=True,
                )
                self.anchor_tab_gate = nn.Parameter(torch.tensor(1.0))
                self.anchor_img_gate = nn.Parameter(torch.tensor(0.2))
                self.delta_gate = nn.Parameter(torch.tensor(0.15))
                fusion_in_dim = fusion_hidden_dim * 3 + effective_tab_dim
                self.fusion_head = nn.Sequential(
                    nn.Linear(fusion_in_dim, fusion_hidden_dim),
                    nn.LayerNorm(fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(fusion_dropout),
                    nn.Linear(fusion_hidden_dim, 1),
                )
            elif self.fusion_mode == "cross_attention":
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
                self.fusion_head = nn.Sequential(
                    nn.Linear(fusion_in_dim, fusion_hidden_dim),
                    nn.BatchNorm1d(fusion_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(fusion_dropout),
                    nn.Linear(fusion_hidden_dim, 1),
                )
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
        if self.tab_tokenizer is not None:
            self._init_custom_weights(self.tab_tokenizer)
        if getattr(self, "member_to_image", None) is not None:
            self._init_custom_weights(self.member_to_image)
        if getattr(self, "member_to_fusion", None) is not None and isinstance(self.member_to_fusion, nn.Linear):
            self._init_custom_weights(self.member_to_fusion)
        if self.bidir_film is not None:
            nn.init.xavier_uniform_(self.bidir_film.weight, gain=0.2)
            nn.init.constant_(self.bidir_film.bias, 0.0)
        if self.fusion_mode == "bidirectional_co_attention" and self.fusion_head is not None:
            nn.init.xavier_uniform_(self.fusion_head[-1].weight, gain=0.5)
            if getattr(self.fusion_head[-1], "bias", None) is not None:
                nn.init.constant_(self.fusion_head[-1].bias, 0.0)
        if getattr(self, 'tab_to_fusion', None) is not None and isinstance(self.tab_to_fusion, nn.Linear):
            self._init_custom_weights(self.tab_to_fusion)

        if init_bias is not None:
            if self.image_only_head is not None:
                nn.init.constant_(self.image_only_head.bias, init_bias)
            if self.tabular_only_head is not None:
                if hasattr(self.tabular_only_head, "bias") and self.tabular_only_head.bias is not None:
                    nn.init.constant_(self.tabular_only_head.bias, init_bias)
            if self.fusion_head is not None and self.fusion_mode != "bidirectional_co_attention":
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
            current_state = self.state_dict()
            for k, v in tabular_state.items():
                if k.startswith("tabular_encoder.") or (
                    k.startswith("tabular_only_head.")
                    and k in current_state
                    and current_state[k].shape == v.shape
                ):
                    filtered_tab_state[k] = v
            missing, unexpected = self.load_state_dict(filtered_tab_state, strict=False)
            print(
                f"Loaded pretrained tabular weights | matched={len(filtered_tab_state)} | "
                f"missing={len(missing)} | unexpected={len(unexpected)}"
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
        spatial_fusion = fusion_mode in {"cross_attention", "bidirectional_co_attention"}
        create_kwargs = {
            "model_name": backbone_name,
            "num_classes": 0,
            "global_pool": "" if spatial_fusion else "avg",
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

    def _encode_image_branch(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[tuple[int, int]]]:
        """Encode images into branch features, auxiliary logits, and optional spatial HxW."""
        if not self.train_image_backbone:
            with torch.no_grad():
                image_features = self.image_encoder(images)
        else:
            image_features = self.image_encoder(images)
        image_features = image_features.float()

        spatial_hw: Optional[tuple[int, int]] = None
        if self.fusion_mode in {"cross_attention", "bidirectional_co_attention"}:
            if image_features.ndim == 4:
                B, C, H, W = image_features.shape
                spatial_hw = (int(H), int(W))
                image_features = image_features.view(B, C, -1).transpose(1, 2)
            image_features = self.image_head(image_features)
            image_pooled = (
                image_features.mean(dim=1)
                if image_features.ndim == 3
                else image_features
            )
            image_logits = self.image_only_head(image_pooled).squeeze(1)
        else:
            if image_features.ndim > 2:
                image_features = torch.flatten(image_features, 1)
            image_features = self.image_head(image_features)
            image_logits = self.image_only_head(image_features).squeeze(1)
        return image_features, image_logits, spatial_hw

    def forward(
        self,
        images: torch.Tensor,
        tabular: torch.Tensor,
        return_attention: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, torch.Tensor]]
        | torch.Tensor
    ):
        
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
            _, image_logits, _ = self._encode_image_branch(images)
            return image_logits

        # --- BOTH MODALITIES (FUSION) ---
        image_features, image_logits, spatial_hw = self._encode_image_branch(images)

        # Tabular branch
        emb_num: Optional[torch.Tensor] = None
        emb_cat: Optional[torch.Tensor] = None
        if hasattr(self.tabular_encoder, "forward_with_embeddings"):
            tabular_features, emb_num, emb_cat = self.tabular_encoder.forward_with_embeddings(tabular)
        else:
            tabular_features = self.tabular_encoder(tabular)
            if getattr(self, "tab_tokenizer", None) is not None:
                emb_num = tabular[:, : self.tab_tokenizer.n_num].unsqueeze(-1)
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
        attn_maps: Optional[dict[str, torch.Tensor]] = None
        if self.fusion_mode == "bidirectional_co_attention":
            if self.tab_tokenizer is None or self.bidir_film is None:
                raise RuntimeError("bidirectional_co_attention modules are not initialized.")

            global_tab = tabular_features.mean(dim=1) if tabular_ensemble_mode else tabular_features
            if emb_num is None:
                emb_num = tabular[:, : self.tab_tokenizer.n_num].unsqueeze(-1)

            tab_tokens = self.tab_tokenizer(
                emb_num=emb_num,
                global_tab=global_tab,
                emb_cat=emb_cat,
            )
            tab_tokens = tab_tokens * tab_keep_mask.reshape(-1, 1, 1)

            img_tokens = (
                image_features.unsqueeze(1)
                if image_features.ndim == 2
                else image_features
            )
            fused_ctx, refined_img_tokens, attn_maps = self.cross_modal_fusion(
                img_tokens,
                tab_tokens,
                image_keep_mask=image_keep_mask,
                tab_keep_mask=tab_keep_mask,
                spatial_hw=spatial_hw,
                return_attention=return_attention,
                return_tokens=True,
            )

            film_params = self.bidir_film(fused_ctx)
            gamma, beta = film_params.chunk(2, dim=-1)
            interaction_3d = (image_keep_mask * tab_keep_mask).reshape(-1, 1, 1)
            kv_img = self.member_kv_norm(refined_img_tokens) * image_keep_mask.unsqueeze(1)

            tab_logits_anchor = tabular_logits.detach() if self.training else tabular_logits
            img_logits_anchor = image_logits.detach() if self.training else image_logits

            if tabular_ensemble_mode:
                B, K, _D_tab = tabular_features.shape
                masked_tab = tabular_features * tab_keep_mask.unsqueeze(1)
                q_mem = self.member_q_norm(self.member_to_fusion(masked_tab)) * tab_keep_mask.unsqueeze(1)
                member_vis_ctx, _ = self.member_to_image(
                    query=q_mem,
                    key=kv_img,
                    value=kv_img,
                    need_weights=False,
                )
                member_vis_ctx = member_vis_ctx * interaction_3d
                modulated_tab = (
                    masked_tab * (1.0 + torch.tanh(gamma).unsqueeze(1))
                    + beta.unsqueeze(1) * tab_keep_mask.unsqueeze(1)
                )
                fused_members = torch.cat(
                    [
                        fused_ctx.unsqueeze(1).expand(-1, K, -1),
                        member_vis_ctx,
                        modulated_tab,
                    ],
                    dim=-1,
                )
                delta_logits = self.fusion_head(
                    fused_members.reshape(-1, fused_members.size(-1))
                ).reshape(B, K)

                tab_anchor = tab_logits_anchor * tab_keep_mask
                img_anchor = (img_logits_anchor * image_keep_mask.squeeze(1)).unsqueeze(1)
                logits = (
                    self.anchor_tab_gate * tab_anchor
                    + self.anchor_img_gate * img_anchor
                    + self.delta_gate * delta_logits
                )
            else:
                masked_tab = tabular_features * tab_keep_mask
                q_mem = (
                    self.member_q_norm(self.member_to_fusion(masked_tab)).unsqueeze(1)
                    * tab_keep_mask.unsqueeze(1)
                )
                member_vis_ctx, _ = self.member_to_image(
                    query=q_mem,
                    key=kv_img,
                    value=kv_img,
                    need_weights=False,
                )
                member_vis_ctx = (member_vis_ctx * interaction_3d).squeeze(1)
                modulated_tab = (
                    masked_tab * (1.0 + torch.tanh(gamma))
                    + beta * tab_keep_mask
                )
                fused_members = torch.cat([fused_ctx, member_vis_ctx, modulated_tab], dim=-1)
                delta_logits = self.fusion_head(fused_members).squeeze(1)
                tab_anchor = tab_logits_anchor * tab_keep_mask.squeeze(1)
                img_anchor = img_logits_anchor * image_keep_mask.squeeze(1)
                logits = (
                    self.anchor_tab_gate * tab_anchor
                    + self.anchor_img_gate * img_anchor
                    + self.delta_gate * delta_logits
                )

            if return_attention and not self.training:
                return logits, (attn_maps or {})
            if self.training:
                return logits, image_logits, tabular_logits
            return logits
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
