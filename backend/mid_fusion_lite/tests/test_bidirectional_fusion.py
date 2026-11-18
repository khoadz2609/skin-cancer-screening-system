from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import torch
from PIL import Image
from torch import nn

from backend.mid_fusion_lite.config import (
    BASE_NUMERIC_COLS,
    DEFAULT_CATEGORICAL_COLS,
    MidFusionLiteConfig,
    build_ablation_columns,
    build_clinical_feature_groups,
    get_preset_config,
)
from backend.mid_fusion_lite.data import EpochRandomUndersampler
from backend.mid_fusion_lite.model import (
    BidirectionalCoAttentionFusion,
    MidFusionLiteModel,
    SemanticGroupTokenizer,
)
from backend.mid_fusion_lite.runner import run_mid_fusion_cv
from backend.mid_fusion_lite.train import SmoothPAUCLoss


class _TinySpatialImageEncoder(nn.Module):
    num_features = 8

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x)).mean(dim=(-2, -1), keepdim=True).expand(-1, 8, 4, 4)


def _make_tiny_tabm_setup(n_num: int = 12, k: int = 4, d_emb: int = 8):
    torch.manual_seed(0)
    x_num = torch.randn(32, n_num)
    bins = rtdl_num_embeddings.compute_bins(x_num, n_bins=6)
    cat_cards = [3, 4]
    num_cols = [
        "age_approx",
        "clin_size_long_diam_mm",
        "tbp_lv_A",
        "tbp_lv_B",
        "tbp_lv_L",
        "tbp_lv_areaMM2",
        "tbp_lv_norm_border",
        "tbp_lv_norm_color",
        "tbp_lv_areaMM2_patient",
        "ud_loc_tbp_lv_areaMM2",
        "patient_lof_score",
        "clin_size_long_diam_mm_sex",
    ][:n_num]
    cat_cols = ["sex", "anatom_site_general"]
    return bins, cat_cards, num_cols, cat_cols, k, d_emb


class ClinicalGroupMappingTests(unittest.TestCase):
    def test_clinical_group_mapping_covers_all_ablation_columns(self) -> None:
        num_cols, cat_cols = build_ablation_columns(True, True, True, True)
        groups = build_clinical_feature_groups(
            numeric_cols=num_cols,
            n_numeric=len(num_cols),
            categorical_cols=cat_cols,
        )
        all_indices: list[int] = []
        for idx_list in groups.values():
            all_indices.extend(idx_list)

        self.assertEqual(sorted(all_indices), list(range(len(num_cols))))
        self.assertEqual(len(all_indices), len(set(all_indices)))

        col_to_group = {
            num_cols[idx]: gname
            for gname, idxs in groups.items()
            for idx in idxs
        }
        self.assertEqual(col_to_group["clin_size_long_diam_mm"], "morphology_shape")
        self.assertEqual(col_to_group["lesion_size_ratio"], "morphology_shape")
        self.assertEqual(col_to_group["tbp_lv_A"], "color_pigment")
        self.assertEqual(col_to_group["tbp_lv_areaMM2_patient"], "patient_context")
        self.assertEqual(col_to_group["ud_loc_tbp_lv_areaMM2"], "ugly_duckling")
        self.assertEqual(col_to_group["age_approx"], "anatomy_demographics")
        self.assertEqual(col_to_group["clin_size_long_diam_mm_sex"], "interactions_counts")

    def test_fallback_partition_without_column_names(self) -> None:
        groups = build_clinical_feature_groups(numeric_cols=None, n_numeric=10)
        flat = sorted(i for v in groups.values() for i in v)
        self.assertEqual(flat, list(range(10)))


class BidirectionalAttentionTests(unittest.TestCase):
    def test_image_to_tab_attention_is_non_degenerate_and_sensitive(self) -> None:
        torch.manual_seed(42)
        num_cols = [
            "clin_size_long_diam_mm",
            "tbp_lv_norm_border",
            "tbp_lv_A",
            "tbp_lv_H",
            "tbp_lv_areaMM2_patient",
            "ud_loc_tbp_lv_areaMM2",
            "age_approx",
            "clin_size_long_diam_mm_sex",
        ]
        tokenizer = SemanticGroupTokenizer(
            n_num=len(num_cols),
            d_num_embedding=8,
            hidden_dim=16,
            global_tab_dim=12,
            dropout=0.0,
            numeric_cols=num_cols,
            categorical_cols=["sex"],
            cat_emb_dim=4,
            feature_token_dim=16,
        ).eval()
        fusion = BidirectionalCoAttentionFusion(
            hidden_dim=16,
            num_heads=4,
            dropout=0.0,
            num_layers=2,
            ffn_mult=2,
            use_2d_pos_emb=True,
        ).eval()

        emb_num = torch.randn(3, len(num_cols), 8)
        global_tab = torch.randn(3, 12)
        emb_cat = torch.zeros(3, 4)
        emb_cat[:, 1] = 1.0

        tab_tokens = tokenizer(emb_num=emb_num, global_tab=global_tab, emb_cat=emb_cat)
        self.assertEqual(tuple(tab_tokens.shape), (3, 7, 16))

        img_patches = torch.randn(3, 16, 16)
        fused, attn_maps = fusion(
            img_patches,
            tab_tokens,
            spatial_hw=(4, 4),
            return_attention=True,
        )
        self.assertEqual(tuple(fused.shape), (3, 32))
        i2t = attn_maps["image_to_tab"]
        t2i = attn_maps["tab_to_image"]
        self.assertEqual(tuple(i2t.shape), (3, 4, 16, 7))
        self.assertEqual(tuple(t2i.shape), (3, 4, 7, 16))
        torch.testing.assert_close(i2t.sum(dim=-1), torch.ones(3, 4, 16))
        torch.testing.assert_close(t2i.sum(dim=-1), torch.ones(3, 4, 7))
        self.assertGreater(float(i2t.detach().std(dim=-1).mean()), 1e-3)

        # Changing color columns must alter the attention distribution over clinical tokens.
        emb_num_color_perturbed = emb_num.clone()
        emb_num_color_perturbed[:, 2:4, :] += 5.0
        tab_tokens_perturbed = tokenizer(
            emb_num=emb_num_color_perturbed,
            global_tab=global_tab,
            emb_cat=emb_cat,
        )
        _, attn_maps_2 = fusion(
            img_patches,
            tab_tokens_perturbed,
            spatial_hw=(4, 4),
            return_attention=True,
        )
        diff = float((attn_maps_2["image_to_tab"] - i2t).detach().abs().max())
        self.assertGreater(diff, 1e-4)

    def test_stage3_loads_stage1_tabm_and_stage2_image_without_missing_keys(self) -> None:
        bins, cat_cards, num_cols, cat_cols, k, d_emb = _make_tiny_tabm_setup()
        n_tab = len(num_cols) + len(cat_cols)

        stage1 = MidFusionLiteModel(
            modality_mode="tabular_only",
            n_tab_features=n_tab,
            use_tabm_tabular=True,
            tabm_arch_type="tabm",
            tabm_k=k,
            tabm_n_blocks=2,
            tabm_d_block=16,
            tab_dropout=0.0,
            bins=bins,
            d_embedding=d_emb,
            cat_cardinalities=cat_cards,
        )
        with patch.object(
            MidFusionLiteModel,
            "_build_image_encoder",
            return_value=_TinySpatialImageEncoder(),
        ):
            stage2 = MidFusionLiteModel(
                modality_mode="image_only",
                backbone_name="fake",
                fusion_mode="cross_attention",
                fusion_hidden_dim=16,
                fusion_dropout=0.0,
                train_image_backbone=True,
            )
            stage3 = MidFusionLiteModel(
                modality_mode="both",
                backbone_name="fake",
                n_tab_features=n_tab,
                use_tabm_tabular=True,
                tabm_arch_type="tabm",
                tabm_k=k,
                tabm_n_blocks=2,
                tabm_d_block=16,
                tab_dropout=0.0,
                bins=bins,
                d_embedding=d_emb,
                cat_cardinalities=cat_cards,
                fusion_mode="bidirectional_co_attention",
                fusion_hidden_dim=16,
                cross_attention_heads=4,
                cross_attention_dropout=0.0,
                fusion_dropout=0.0,
                fusion_num_layers=2,
                modality_dropout_prob=0.15,
                train_image_backbone=True,
                tabular_numeric_cols=num_cols,
                tabular_categorical_cols=cat_cols,
            )

        stage3.load_unimodal_weights(
            tabular_state=stage1.state_dict(),
            image_state=stage2.state_dict(),
        )
        for key, tensor in stage1.state_dict().items():
            if key.startswith("tabular_encoder.") or key.startswith("tabular_only_head."):
                torch.testing.assert_close(stage3.state_dict()[key], tensor)
        for key, tensor in stage2.state_dict().items():
            if key.startswith(("image_encoder.", "image_head.", "image_only_head.")):
                torch.testing.assert_close(stage3.state_dict()[key], tensor)

    def test_modality_dropout_invariance_and_no_mask_params(self) -> None:
        torch.manual_seed(19)
        bins, cat_cards, num_cols, cat_cols, k, d_emb = _make_tiny_tabm_setup()
        n_tab = len(num_cols) + len(cat_cols)

        with patch.object(
            MidFusionLiteModel,
            "_build_image_encoder",
            return_value=_TinySpatialImageEncoder(),
        ):
            model = MidFusionLiteModel(
                modality_mode="both",
                backbone_name="fake",
                n_tab_features=n_tab,
                use_tabm_tabular=True,
                tabm_arch_type="tabm",
                tabm_k=k,
                tabm_n_blocks=2,
                tabm_d_block=16,
                tab_dropout=0.0,
                bins=bins,
                d_embedding=d_emb,
                cat_cardinalities=cat_cards,
                fusion_mode="bidirectional_co_attention",
                fusion_hidden_dim=16,
                cross_attention_heads=4,
                cross_attention_dropout=0.0,
                fusion_dropout=0.0,
                fusion_num_layers=2,
                modality_dropout_prob=0.5,
                train_image_backbone=True,
                tabular_numeric_cols=num_cols,
                tabular_categorical_cols=cat_cols,
            )

        self.assertFalse(any("mask" in k_name for k_name in model.state_dict()))
        self.assertFalse(any("_group_idx" in k_name for k_name in model.state_dict()))

        model.train(True)
        images = torch.randn(2, 3, 16, 16)
        x_num = torch.randn(2, len(num_cols))
        x_cat = torch.tensor([[1, 2], [2, 1]], dtype=torch.float32)
        tabular = torch.cat([x_num, x_cat], dim=1)

        image_keep = torch.tensor([[0.0], [1.0]])
        tab_keep = torch.tensor([[1.0], [0.0]])

        with patch.object(
            model,
            "_sample_modality_keep_masks",
            return_value=(image_keep, tab_keep),
        ):
            fusion_logits, image_logits, tab_logits = model(images, tabular)

            changed_images = images.clone()
            changed_tabular = tabular.clone()
            changed_images[0] += 1000.0  # Image dropped for sample 0
            changed_tabular[1, : len(num_cols)] += 1000.0  # Tabular dropped for sample 1
            changed_tabular[1, len(num_cols) :] = 0.0
            changed_logits, _, _ = model(changed_images, changed_tabular)

        self.assertEqual(tuple(fusion_logits.shape), (2, k))
        self.assertEqual(tuple(image_logits.shape), (2,))
        self.assertEqual(tuple(tab_logits.shape), (2, k))
        torch.testing.assert_close(fusion_logits, changed_logits)
        fusion_logits.sum().backward()
        self.assertIsNotNone(model.vis_to_tab[0].weight.grad)
        self.assertGreater(float(model.vis_to_tab[0].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.tab_tokenizer.num_feat_proj.weight.grad)
        self.assertGreater(float(model.tab_tokenizer.num_feat_proj.weight.grad.abs().sum()), 0.0)


class SamplingLossAndPipelineTests(unittest.TestCase):
    def test_hard_negative_sampler_and_smooth_pauc_loss(self) -> None:
        targets = np.array([1] * 4 + [0] * 40, dtype=np.int64)
        hard_weights = np.ones(44, dtype=np.float64)
        hard_weights[4] = 5000.0  # Index 4 is strongly hard negative

        sampler = EpochRandomUndersampler(
            targets=targets,
            negative_ratio=1,
            seed=123,
            hard_negative_weights=hard_weights,
            hard_negative_alpha=0.85,
        )
        hard_hits = 0
        easy_hits = 0
        for _ in range(30):
            drawn = list(iter(sampler))
            if 4 in drawn:
                hard_hits += 1
            if 5 in drawn:
                easy_hits += 1
        self.assertGreater(hard_hits, easy_hits)

        loss_fn = SmoothPAUCLoss(margin=0.2, top_neg_fraction=0.5)
        y = torch.tensor([1.0, 0.0, 0.0, 0.0])
        bad_ranking = torch.tensor([[-2.0, -2.0], [3.0, 3.0], [2.0, 2.0], [-1.0, -1.0]], requires_grad=True)
        good_ranking = torch.tensor([[3.0, 3.0], [-3.0, -3.0], [-3.0, -3.0], [-3.0, -3.0]])
        loss_bad = loss_fn(bad_ranking, y)
        loss_good = loss_fn(good_ranking, y)
        self.assertGreater(float(loss_bad.detach()), float(loss_good.detach()))
        loss_bad.backward()
        self.assertTrue(torch.isfinite(bad_ranking.grad).all())

        # Safe zero loss when batch has only negatives
        no_pos_loss = loss_fn(bad_ranking.detach(), torch.zeros(4))
        self.assertEqual(float(no_pos_loss), 0.0)

    def test_end_to_end_stage3_smoke_with_cache(self) -> None:
        rng = np.random.default_rng(7)
        n_train = 80
        n_test = 16

        def _make_df(n: int, prefix: str, with_target: bool) -> pd.DataFrame:
            # Assign varied lesions per patient (1 and 12) with isolated extreme rows so
            # ud_count_location (> 2 SD) and single/multi-lesion context columns all vary.
            pat_ids: list[str] = []
            p_idx = 0
            group_sizes = (1, 12)
            while len(pat_ids) < n:
                sz = group_sizes[p_idx % len(group_sizes)]
                for _ in range(sz):
                    if len(pat_ids) < n:
                        pat_ids.append(f"{prefix}_pat_{p_idx}")
                p_idx += 1

            rows = {
                "isic_id": [f"{prefix}_{i}" for i in range(n)],
                "patient_id": pat_ids,
            }
            if with_target:
                rows["target"] = [1 if i % 4 == 0 else 0 for i in range(n)]
            for col in BASE_NUMERIC_COLS:
                if col == "age_approx":
                    rows[col] = rng.integers(25, 80, size=n).astype(float)
                elif col == "tbp_lv_nevi_confidence":
                    rows[col] = rng.uniform(5.0, 95.0, size=n)
                else:
                    vals = rng.uniform(1.0, 2.0, size=n)
                    vals[::13] += 80.0
                    rows[col] = vals
            for col in DEFAULT_CATEGORICAL_COLS:
                rows[col] = rng.choice(["a", "b"], size=n)
            return pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "train-metadata.csv"
            test_csv = root / "test-metadata.csv"
            sub_csv = root / "sample_submission.csv"
            train_h5 = root / "train-image.hdf5"
            test_h5 = root / "test-image.hdf5"
            cache_dir = root / "tab_cache"

            train_df = _make_df(n_train, "TR", with_target=True)
            test_df = _make_df(n_test, "TE", with_target=False)
            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)
            pd.DataFrame({"isic_id": test_df["isic_id"], "target": 0.5}).to_csv(sub_csv, index=False)

            for df, h5_path in ((train_df, train_h5), (test_df, test_h5)):
                with h5py.File(h5_path, "w") as h5:
                    for isic_id in df["isic_id"]:
                        buf = BytesIO()
                        arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
                        Image.fromarray(arr).save(buf, format="JPEG")
                        h5.create_dataset(isic_id, data=np.frombuffer(buf.getvalue(), dtype=np.uint8))

            num_cols, cat_cols = build_ablation_columns(True, True, True, True)
            common_kwargs = dict(
                preset_name="v2_stage3_smoke",
                n_splits=2,
                fold_indices=[0],
                image_size=16,
                train_batch_size=8,
                valid_batch_size=8,
                num_workers=0,
                use_amp=False,
                use_shades_of_gray=False,
                negative_ratio=2,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                tabm_k=4,
                tabm_n_blocks=1,
                tabm_d_block=16,
                n_bins=6,
                d_embedding=8,
                fusion_hidden_dim=16,
                cross_attention_heads=4,
                fusion_num_layers=1,
                fusion_ffn_mult=2,
                backbone_name="fake",
                train_image_backbone=True,
                train_metadata_path=str(train_csv),
                test_metadata_path=str(test_csv),
                sample_submission_path=str(sub_csv),
                train_hdf5_path=str(train_h5),
                test_hdf5_path=str(test_h5),
                tabular_cache_dir=str(cache_dir),
            )

            with patch.object(
                MidFusionLiteModel,
                "_build_image_encoder",
                return_value=_TinySpatialImageEncoder(),
            ):
                cfg_s1 = MidFusionLiteConfig(
                    run_mode="tabular_only",
                    stage1_epochs=1,
                    output_dir=str(root / "out_s1"),
                    **common_kwargs,
                )
                run_mid_fusion_cv(cfg_s1)

                cfg_s2 = MidFusionLiteConfig(
                    run_mode="image_only",
                    fusion_mode="cross_attention",
                    stage2_epochs=1,
                    stage2_train_backbone=True,
                    output_dir=str(root / "out_s2"),
                    **common_kwargs,
                )
                run_mid_fusion_cv(cfg_s2)

                # Run Stage 3 (must load from tabular cache without re-running FE!)
                cfg_s3 = MidFusionLiteConfig(
                    run_mode="fusion",
                    fusion_mode="bidirectional_co_attention",
                    pretrained_tabular_dir=str(root / "out_s1" / "stage1_tabular"),
                    pretrained_image_dir=str(root / "out_s2" / "stage2_image"),
                    stage3_freeze_epochs=1,
                    stage3_unfreeze_epochs=1,
                    aux_loss_image_weight=0.0,
                    aux_loss_tab_weight=0.0,
                    hard_negative_alpha=0.35,
                    pauc_loss_weight=0.5,
                    tta_steps=4,
                    output_dir=str(root / "out_s3"),
                    **common_kwargs,
                )
                with patch(
                    "backend.mid_fusion_lite.runner.add_feature_engineering_folds",
                    side_effect=AssertionError("Expected tabular cache hit to skip FE"),
                ):
                    res = run_mid_fusion_cv(cfg_s3)

            metrics = json.loads(Path(res["metrics_path"]).read_text(encoding="utf-8"))
            self.assertEqual(metrics["model_config"]["fusion_mode"], "bidirectional_co_attention")
            self.assertIn("cv_auc", metrics)
            self.assertTrue((root / "out_s3" / "mid_fusion_lite_fold_0.pt").is_file())
            self.assertTrue((cache_dir / "fold_0" / "meta.json").is_file())

    def test_v2_preset_exists(self) -> None:
        cfg = get_preset_config("v2_stage3_bidir")
        self.assertEqual(cfg.fusion_mode, "bidirectional_co_attention")
        self.assertEqual(cfg.run_mode, "fusion")


if __name__ == "__main__":
    unittest.main()
