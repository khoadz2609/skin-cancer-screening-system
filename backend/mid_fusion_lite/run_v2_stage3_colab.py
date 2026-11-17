"""Run Stage 3 experiments with BidirectionalCoAttentionFusion reusing Stage 1 (TabM) & Stage 2 (EffNet-B0)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mid_fusion_lite.config import MidFusionLiteConfig, get_preset_config  # noqa: E402
from mid_fusion_lite.runner import run_mid_fusion_cv  # noqa: E402


EXPERIMENT_PRESETS = {
    "arch_only": {
        "pauc_loss_weight": 0.0,
        "hard_negative_alpha": 0.0,
        "tta_steps": 1,
    },
    "arch_pauc_hnm": {
        "pauc_loss_weight": 0.5,
        "hard_negative_alpha": 0.35,
        "tta_steps": 1,
    },
    "arch_pauc_hnm_tta": {
        "pauc_loss_weight": 0.5,
        "hard_negative_alpha": 0.35,
        "tta_steps": 4,
    },
}


def _find_dataset_root(explicit_data_dir: Optional[str]) -> Path:
    candidates: list[Path] = []
    if explicit_data_dir:
        candidates.append(Path(explicit_data_dir))
    candidates.extend(
        [
            Path("/content/isic2024_data"),
            Path("/content/data"),
            REPO_ROOT / "data",
            Path.cwd() / "data",
            Path("/kaggle/input/isic-2024-challenge"),
            Path("/kaggle/input/competitions/isic-2024-challenge"),
        ]
    )
    for cand in candidates:
        if (cand / "train-metadata.csv").is_file() and (cand / "train-image.hdf5").is_file():
            return cand

    for search_base in (Path("/content"), Path("/kaggle/input")):
        if not search_base.exists():
            continue
        for root, _dirs, files in os.walk(search_base):
            if "train-metadata.csv" in files and "train-image.hdf5" in files:
                return Path(root)

    raise FileNotFoundError(
        "Could not locate ISIC 2024 dataset files (train-metadata.csv, train-image.hdf5). "
        "Pass --data-dir pointing to the folder containing ISIC 2024 files."
    )


def _resolve_optional_file(root: Path, primary: str, fallbacks: list[Path]) -> Path:
    candidate = root / primary
    if candidate.is_file():
        return candidate
    for fb in fallbacks:
        if fb.is_file():
            return fb
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default="arch_only",
        help="Controlled Stage-3 experiment variant.",
    )
    parser.add_argument("--data-dir", default=None, help="Directory with ISIC 2024 CSV/HDF5 files.")
    parser.add_argument(
        "--pretrained-tabular-dir",
        default="/content/isic_weights/tabm",
        help="Directory or {fold}/{fold_1} template for Stage-1 TabM checkpoints.",
    )
    parser.add_argument(
        "--pretrained-image-dir",
        default=str(BACKEND_DIR / "backbone_image"),
        help="Directory or template for Stage-2 image checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for Stage-3 checkpoints, OOF predictions, and metrics.json.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/content/tabular_cache_isic2024",
        help="Fold tabular feature/bin cache directory to avoid repeating 610-col FE.",
    )
    parser.add_argument("--folds", default="0,1", help="Comma-separated fold indices (default: 0,1).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--stage3a-epochs", type=int, default=4)
    parser.add_argument("--stage3b-epochs", type=int, default=6)
    args = parser.parse_args()

    dataset_root = _find_dataset_root(args.data_dir)
    train_meta = dataset_root / "train-metadata.csv"
    test_meta = _resolve_optional_file(
        dataset_root,
        "test-metadata.csv",
        [REPO_ROOT / "tests" / "test-metadata.csv"],
    )
    sample_sub = _resolve_optional_file(
        dataset_root,
        "sample_submission.csv",
        [REPO_ROOT / "tests" / "sample_submission.csv"],
    )
    train_h5 = dataset_root / "train-image.hdf5"
    test_h5 = _resolve_optional_file(
        dataset_root,
        "test-image.hdf5",
        [REPO_ROOT / "tests" / "test-image.hdf5"],
    )

    fold_indices = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    out_dir = args.output_dir or f"/content/fusion_v2_outputs/{args.exp}"

    base_cfg: MidFusionLiteConfig = get_preset_config("v2_stage3_bidir")
    exp_overrides = EXPERIMENT_PRESETS[args.exp]
    cfg = base_cfg.with_updates(
        fold_indices=fold_indices,
        train_metadata_path=str(train_meta),
        test_metadata_path=str(test_meta),
        sample_submission_path=str(sample_sub),
        train_hdf5_path=str(train_h5),
        test_hdf5_path=str(test_h5),
        pretrained_tabular_dir=args.pretrained_tabular_dir,
        pretrained_image_dir=args.pretrained_image_dir,
        tabular_cache_dir=args.cache_dir,
        output_dir=out_dir,
        train_batch_size=int(args.batch_size),
        valid_batch_size=int(args.valid_batch_size),
        num_workers=int(args.num_workers),
        stage3_freeze_epochs=int(args.stage3a_epochs),
        stage3_unfreeze_epochs=int(args.stage3b_epochs),
        **exp_overrides,
    )

    print("=" * 72)
    print(f"STAGE-3 V2 EXPERIMENT: {args.exp}")
    print(f"Dataset root : {dataset_root}")
    print(f"TabM weights : {cfg.pretrained_tabular_dir}")
    print(f"Img weights  : {cfg.pretrained_image_dir}")
    print(f"Cache dir    : {cfg.tabular_cache_dir}")
    print(f"Folds        : {cfg.fold_indices}")
    print(f"Output dir   : {cfg.output_dir}")
    print("=" * 72)

    results = run_mid_fusion_cv(cfg)
    print(
        f"[DONE] exp={args.exp} | OOF AUROC={results['cv_auc']:.6f} | "
        f"OOF pAUC={results['cv_pauc']:.6f}"
    )


if __name__ == "__main__":
    main()
