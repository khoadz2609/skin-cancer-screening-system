from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import backend.mid_fusion_lite.runner as runner_module
from backend.mid_fusion_lite.config import (
    MidFusionLiteConfig,
    modality_mode_for_run_mode,
    resolve_run_mode,
)
from backend.mid_fusion_lite.model import (
    CrossModalAttentionFusion,
    GatedMultimodalUnit,
    MidFusionLiteModel,
)
from backend.mid_fusion_lite.runner import (
    _missing_stage3_checkpoint_error,
    _resolve_pretrained_checkpoint,
)


class _EnsembleEncoder(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.k, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x.unsqueeze(1) + offsets


class _EnsembleHead(nn.Module):
    def __init__(self, dim: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeImageEncoder(nn.Module):
    num_features = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeStage2Model(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.weight.expand(images.size(0))


def _mask_only_model(probability: float, modality_mode: str = "both") -> MidFusionLiteModel:
    model = MidFusionLiteModel.__new__(MidFusionLiteModel)
    nn.Module.__init__(model)
    model.modality_mode = modality_mode
    model.modality_dropout_prob = probability
    nn.Module.train(model, True)
    return model


def _lightweight_ensemble_model(mode: str, dim: int = 4, k: int = 3) -> MidFusionLiteModel:
    """Build the forward graph without constructing a timm or TabM backbone."""
    model = MidFusionLiteModel.__new__(MidFusionLiteModel)
    nn.Module.__init__(model)
    model.modality_mode = "both"
    model.modality_dropout_prob = 1.0
    model.fusion_mode = mode
    model.keep_tabm_ensemble = True
    model.train_image_backbone = True

    model.image_encoder = nn.Identity()
    model.image_head = nn.Identity()
    model.image_only_head = nn.Linear(dim, 1)
    model.tabular_encoder = _EnsembleEncoder(k)
    model.tabular_only_head = _EnsembleHead(dim, k)

    if mode == "cross_attention":
        model.tab_to_fusion = nn.Linear(dim, dim)
        model.cross_modal_fusion = CrossModalAttentionFusion(dim, num_heads=2, dropout=0.0)
        fusion_dim = dim * 2
    elif mode == "gmu":
        model.tab_to_fusion = nn.Linear(dim, dim)
        model.cross_modal_fusion = GatedMultimodalUnit(dim, dropout=0.0)
        fusion_dim = dim
    else:
        model.tab_to_fusion = None
        model.cross_modal_fusion = None
        fusion_dim = dim * 2

    model.fusion_head = nn.Linear(fusion_dim, 1)
    nn.Module.train(model, True)
    return model


class ModalityMaskSamplingTests(unittest.TestCase):
    def test_probability_zero_keeps_both(self) -> None:
        model = _mask_only_model(0.0)
        image_keep, tab_keep = model._sample_modality_keep_masks(
            128, torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.equal(image_keep, torch.ones_like(image_keep)))
        self.assertTrue(torch.equal(tab_keep, torch.ones_like(tab_keep)))

    def test_probability_one_drops_exactly_one_and_is_balanced(self) -> None:
        torch.manual_seed(7)
        model = _mask_only_model(1.0)
        image_keep, tab_keep = model._sample_modality_keep_masks(
            10_000, torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.all(image_keep + tab_keep == 1))
        image_drop_rate = float((image_keep == 0).float().mean())
        self.assertGreater(image_drop_rate, 0.47)
        self.assertLess(image_drop_rate, 0.53)

    def test_eval_and_unimodal_modes_keep_both(self) -> None:
        for modality_mode in ("both", "image_only", "tabular_only"):
            with self.subTest(modality_mode=modality_mode):
                model = _mask_only_model(1.0, modality_mode=modality_mode)
                if modality_mode == "both":
                    nn.Module.train(model, False)
                image_keep, tab_keep = model._sample_modality_keep_masks(
                    32, torch.device("cpu"), torch.float32
                )
                self.assertTrue(torch.all(image_keep == 1))
                self.assertTrue(torch.all(tab_keep == 1))


class CheckpointResolutionTests(unittest.TestCase):
    def test_resolves_actual_zero_indexed_checkpoint_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "uploaded_output"
            nested.mkdir()
            tab_path = nested / "tabm_fold_0.pt"
            image_path = nested / "image_only_fold_0.pt"
            tab_path.touch()
            image_path.touch()

            self.assertEqual(
                _resolve_pretrained_checkpoint(str(root), 0, "tabular"),
                tab_path,
            )
            self.assertEqual(
                _resolve_pretrained_checkpoint(str(root), 0, "image"),
                image_path,
            )

    def test_stage3_missing_checkpoint_message_is_actionable(self) -> None:
        error = _missing_stage3_checkpoint_error(
            "/kaggle/input/missing", fold_index=0, branch="tabular"
        )
        message = str(error)
        self.assertIn("tabm_fold_0.pt", message)
        self.assertIn("will not train", message)


class ModeSpecificConfigurationTests(unittest.TestCase):
    def test_run_mode_is_the_single_canonical_selector(self) -> None:
        expected_modalities = {
            "tabular_only": "tabular_only",
            "image_only": "image_only",
            "fusion": "both",
            "progressive": "both",
            "end_to_end": "both",
        }
        for run_mode, expected_modality in expected_modalities.items():
            with self.subTest(run_mode=run_mode):
                config = MidFusionLiteConfig(
                    run_mode=run_mode,
                    # Contradictory legacy values must not override run_mode.
                    modality_mode="tabular_only",
                    use_staged_pretraining=True,
                    stage_to_run="stage1",
                )
                self.assertEqual(resolve_run_mode(config), run_mode)
                self.assertEqual(
                    modality_mode_for_run_mode(run_mode), expected_modality
                )

    def test_legacy_runner_fields_still_map_to_run_mode(self) -> None:
        legacy_cases = [
            ({"modality_mode": "tabular_only"}, "tabular_only"),
            ({"modality_mode": "image_only"}, "image_only"),
            ({"modality_mode": "both"}, "end_to_end"),
            (
                {"use_staged_pretraining": True, "stage_to_run": "stage3"},
                "fusion",
            ),
            (
                {"use_staged_pretraining": True, "stage_to_run": "all"},
                "progressive",
            ),
        ]
        for updates, expected in legacy_cases:
            with self.subTest(updates=updates):
                self.assertEqual(
                    resolve_run_mode(MidFusionLiteConfig(**updates)), expected
                )

    def test_tabular_only_mlp_does_not_require_image_or_fusion_parameters(self) -> None:
        model = MidFusionLiteModel(
            n_tab_features=5,
            use_tabm_tabular=False,
            tab_hidden_dim=8,
            tab_dropout=0.0,
            modality_mode="tabular_only",
        ).eval()
        with torch.no_grad():
            logits = model(torch.empty(3, 0), torch.randn(3, 5))
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertIsNone(model.image_encoder)
        self.assertIsNone(model.fusion_head)

    def test_tabular_only_tabm_does_not_require_mlp_or_image_parameters(self) -> None:
        model = MidFusionLiteModel(
            n_tab_features=5,
            use_tabm_tabular=True,
            tabm_arch_type="tabm",
            tabm_k=3,
            tabm_n_blocks=1,
            tabm_d_block=8,
            tab_dropout=0.0,
            modality_mode="tabular_only",
        ).eval()
        with torch.no_grad():
            logits = model(torch.empty(2, 0), torch.randn(2, 5))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_image_only_does_not_require_tabular_or_attention_parameters(self) -> None:
        with patch.object(
            MidFusionLiteModel,
            "_build_image_encoder",
            return_value=_FakeImageEncoder(),
        ):
            model = MidFusionLiteModel(
                backbone_name="fake",
                fusion_hidden_dim=4,
                fusion_mode="concat",
                fusion_dropout=0.0,
                modality_mode="image_only",
            ).eval()
        with torch.no_grad():
            logits = model(torch.randn(3, 4), torch.empty(3, 0))
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertIsNone(model.tabular_encoder)
        self.assertIsNone(model.fusion_head)

    def test_missing_relevant_parameter_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_tab_features is required"):
            MidFusionLiteModel(
                use_tabm_tabular=False,
                tab_hidden_dim=8,
                tab_dropout=0.0,
                modality_mode="tabular_only",
            )

    def test_stage2_returns_its_own_history_and_score(self) -> None:
        config = MidFusionLiteConfig(
            run_mode="image_only",
            stage2_epochs=1,
            save_checkpoints=False,
            backbone_weights_path=None,
        )
        expected_state = {"weight": torch.tensor(1.0)}
        expected_history = [{"epoch": 1, "valid_pauc": 0.42}]
        with (
            patch.object(runner_module, "MidFusionLiteModel", _FakeStage2Model),
            patch.object(
                runner_module,
                "fit_one_fold",
                return_value=(expected_state, expected_history, 0.42),
            ),
        ):
            state, history, score = runner_module._run_stage2_image(
                fold_index=0,
                config=config,
                train_loader=object(),
                valid_loader=object(),
                device=torch.device("cpu"),
                pos_weight=1.0,
                x_train_shape=1,
            )

        self.assertEqual(state, expected_state)
        self.assertEqual(history, expected_history)
        self.assertEqual(score, 0.42)


class FusionMaskingTests(unittest.TestCase):
    def test_cross_attention_missing_branch_is_invariant_and_zero(self) -> None:
        torch.manual_seed(11)
        fusion = CrossModalAttentionFusion(8, num_heads=2, dropout=0.0).eval()
        image = torch.randn(5, 3, 8)
        tab_a = torch.randn(5, 8)
        tab_b = tab_a + 1000.0
        image_keep = torch.ones(5, 1)
        tab_drop = torch.zeros(5, 1)

        out_a = fusion(image, tab_a, image_keep, tab_drop)
        out_b = fusion(image, tab_b, image_keep, tab_drop)
        torch.testing.assert_close(out_a, out_b)
        torch.testing.assert_close(out_a[:, 8:], torch.zeros_like(out_a[:, 8:]))

        image_b = image + 1000.0
        image_drop = torch.zeros(5, 1)
        tab_keep = torch.ones(5, 1)
        out_a = fusion(image, tab_a, image_drop, tab_keep)
        out_b = fusion(image_b, tab_a, image_drop, tab_keep)
        torch.testing.assert_close(out_a, out_b)
        torch.testing.assert_close(out_a[:, :8], torch.zeros_like(out_a[:, :8]))

    def test_gmu_missing_branch_is_invariant_despite_linear_bias(self) -> None:
        torch.manual_seed(13)
        fusion = GatedMultimodalUnit(8, dropout=0.0).eval()
        with torch.no_grad():
            fusion.image_transform.bias.fill_(2.0)
            fusion.tab_transform.bias.fill_(3.0)

        image = torch.randn(5, 8)
        tab = torch.randn(5, 8)
        ones = torch.ones(5, 1)
        zeros = torch.zeros(5, 1)

        out_a = fusion(image, tab, ones, zeros)
        out_b = fusion(image, tab + 1000.0, ones, zeros)
        torch.testing.assert_close(out_a, out_b)

        out_a = fusion(image, tab, zeros, ones)
        out_b = fusion(image + 1000.0, tab, zeros, ones)
        torch.testing.assert_close(out_a, out_b)

    def test_all_modes_support_shared_patient_mask_for_tabm_ensemble(self) -> None:
        torch.manual_seed(17)
        batch_size, k, dim = 2, 3, 4
        image_keep = torch.tensor([[0.0], [1.0]])
        tab_keep = torch.tensor([[1.0], [0.0]])

        for mode in ("concat", "gmu", "cross_attention"):
            with self.subTest(mode=mode):
                model = _lightweight_ensemble_model(mode, dim=dim, k=k)
                image_shape = (batch_size, 2, dim) if mode == "cross_attention" else (batch_size, dim)
                images = torch.randn(*image_shape, requires_grad=True)
                tabular = torch.randn(batch_size, dim, requires_grad=True)

                with patch.object(
                    model,
                    "_sample_modality_keep_masks",
                    return_value=(image_keep, tab_keep),
                ):
                    fusion_logits, image_logits, tab_logits = model(images, tabular)

                    changed_images = images.detach().clone()
                    changed_tabular = tabular.detach().clone()
                    changed_images[0] += 1000.0  # Image is dropped for patient 0.
                    changed_tabular[1] += 1000.0  # Tabular is dropped for patient 1.
                    changed_logits, _, _ = model(changed_images, changed_tabular)

                self.assertEqual(tuple(fusion_logits.shape), (batch_size, k))
                self.assertEqual(tuple(image_logits.shape), (batch_size,))
                self.assertEqual(tuple(tab_logits.shape), (batch_size, k))
                torch.testing.assert_close(fusion_logits, changed_logits)

                # Auxiliary predictions must use the clean, unmasked features.
                expected_tab_logits = model.tabular_only_head(
                    model.tabular_encoder(tabular)
                ).squeeze(-1)
                torch.testing.assert_close(tab_logits, expected_tab_logits)

                fusion_logits.sum().backward()
                self.assertIsNotNone(model.fusion_head.weight.grad)

    def test_masks_do_not_add_checkpoint_parameters(self) -> None:
        for fusion in (
            CrossModalAttentionFusion(8, num_heads=2, dropout=0.0),
            GatedMultimodalUnit(8, dropout=0.0),
        ):
            self.assertFalse(any("mask" in key for key in fusion.state_dict()))


if __name__ == "__main__":
    unittest.main()
