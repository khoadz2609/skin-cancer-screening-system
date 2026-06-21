"""Ablation 2 giai doan: sang loc nhanh → xac nhan top candidates.

Giai doan 1 (Screen): 1 fold, 3 epochs, 120 steps  → loai bo cau hinh te
Giai doan 2 (Confirm): 3 folds, 5 epochs, full steps → xac nhan top N

Usage:
    python quick_ablation.py                          # Chay tat ca, chi Phase 1
    python quick_ablation.py --phase 1                # Chi Phase 1 (sang loc)
    python quick_ablation.py --phase 2 --top 3        # Phase 1 + Phase 2 (top 3)
    python quick_ablation.py --only baseline,combo_simple  # Chi 2 thi nghiem
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import sys
import os
from pathlib import Path

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

try:
    from .config import MidFusionLiteConfig
    from .runner import run_mid_fusion_cv
except ImportError:
    from config import MidFusionLiteConfig
    from runner import run_mid_fusion_cv


# =====================================================================
# CONFIG TEMPLATES
# =====================================================================

def _get_data_paths() -> dict:
    is_kaggle = Path('/kaggle/input').exists()
    if is_kaggle:
        # Trong notebook, id của dataset có thể là 'isic-2024-challenge'
        comp_root = Path('/kaggle/input/isic-2024-challenge')
        if not comp_root.exists():
            # Hoặc đôi khi Kaggle lưu dưới tên 'competitions'
            comp_root = Path('/kaggle/input/competitions/isic-2024-challenge')
        
        return dict(
            train_metadata_path=str(comp_root / 'train-metadata.csv'),
            test_metadata_path=str(comp_root / 'test-metadata.csv'),
            sample_submission_path=str(comp_root / 'sample_submission.csv'),
            train_hdf5_path=str(comp_root / 'train-image.hdf5'),
            test_hdf5_path=str(comp_root / 'test-image.hdf5'),
        )
    return {}


def _screen_base() -> dict:
    """Phase 1: nhanh, chi de loai bo cau hinh te ro rang."""
    base = dict(
        preset_name="screen",
        n_splits=3,
        fold_indices=[0],
        image_size=128,
        train_batch_size=32,
        valid_batch_size=64,
        epochs=3,
        patience=3,
        lr=3e-4,
        weight_decay=1e-4,
        negative_ratio=6,
        backbone_name="efficientnet_b0",
        tab_hidden_dim=128,
        fusion_hidden_dim=192,
        fusion_mode="gmu",
        tab_dropout=0.1,
        fusion_dropout=0.2,
        train_image_backbone=False,
        use_pretrained_backbone=False,
        max_train_steps_per_epoch=120,
        max_valid_steps=60,
        show_predict_progress=False,
        save_checkpoints=False,
        num_workers=0,
        output_dir="./mid_fusion_lite_outputs/ablation_screen",
    )
    base.update(_get_data_paths())
    return base


def _confirm_base() -> dict:
    """Phase 2: cham hon nhung dang tin cay hon."""
    base = dict(
        preset_name="confirm",
        n_splits=3,
        fold_indices=None,       # chay tat ca 3 folds
        image_size=160,
        train_batch_size=32,
        valid_batch_size=64,
        epochs=5,
        patience=3,
        lr=3e-4,
        weight_decay=1e-4,
        negative_ratio=6,
        backbone_name="efficientnet_b0",
        tab_hidden_dim=128,
        fusion_hidden_dim=192,
        fusion_mode="gmu",
        tab_dropout=0.1,
        fusion_dropout=0.2,
        train_image_backbone=False,
        use_pretrained_backbone=False,
        max_train_steps_per_epoch=None,  # full steps
        max_valid_steps=None,            # full validation
        show_predict_progress=False,
        save_checkpoints=False,
        num_workers=0,
        output_dir="./mid_fusion_lite_outputs/ablation_confirm",
    )
    base.update(_get_data_paths())
    return base


# =====================================================================
# EXPERIMENTS — Them/sua o day
# =====================================================================

EXPERIMENTS = {
    # --- Baseline ---
    "baseline": {},

    # --- Modality isolation ---
    "mod_tabular_only": {"modality_mode": "tabular_only"},
    "mod_image_only": {"modality_mode": "image_only"},

    # --- Auxiliary loss weights ---
    "aux_img0.1_tab0.1": {"aux_loss_image_weight": 0.1, "aux_loss_tab_weight": 0.1},
    "aux_img0.0_tab0.0": {"aux_loss_image_weight": 0.0, "aux_loss_tab_weight": 0.0},
    "aux_img0.2_tab0.3": {"aux_loss_image_weight": 0.2, "aux_loss_tab_weight": 0.3},

    # --- Modality dropout ---
    "mod_drop_0.05": {"modality_dropout_prob": 0.05},
    "mod_drop_0.00": {"modality_dropout_prob": 0.0},

    # --- Negative ratio ---
    "neg_ratio_4": {"negative_ratio": 4},
    "neg_ratio_8": {"negative_ratio": 8},

    # --- Combo ---
    "combo_simple": {
        "aux_loss_image_weight": 0.1,
        "aux_loss_tab_weight": 0.1,
        "modality_dropout_prob": 0.05,
    },
    "combo_minimal": {
        "aux_loss_image_weight": 0.0,
        "aux_loss_tab_weight": 0.0,
        "modality_dropout_prob": 0.0,
    },
}


# =====================================================================
# RUNNER
# =====================================================================

def _run_one(name: str, overrides: dict, base: dict) -> tuple[str, float, float]:
    cfg_kwargs = {**base, **overrides}
    cfg_kwargs["output_dir"] = f"{base['output_dir']}/{name}"
    config = MidFusionLiteConfig(**cfg_kwargs)

    t0 = time.time()
    try:
        result = run_mid_fusion_cv(config)
        pauc = result["cv_pauc"]
        elapsed = time.time() - t0
        return name, pauc, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f">>> {name}: FAILED ({e})")
        return name, float("nan"), elapsed
    finally:
        gc.collect()


def _print_table(
    results: list[tuple[str, float, float]],
    phase_label: str,
    baseline_pauc: float | None = None,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {phase_label}")
    print(f"{'=' * 72}")
    print(f"  {'#':<4} {'Experiment':<25} {'pAUC':>10} {'Time':>8} {'vs Baseline':>14}")
    print(f"  {'-' * 63}")

    sorted_results = sorted(
        results,
        key=lambda x: -x[1] if not math.isnan(x[1]) else float("-inf"),
    )

    for rank, (name, pauc, elapsed) in enumerate(sorted_results, 1):
        delta_str = ""
        if baseline_pauc is not None and not math.isnan(pauc) and not math.isnan(baseline_pauc):
            diff = pauc - baseline_pauc
            sign = "+" if diff >= 0 else ""
            delta_str = f"{sign}{diff:.6f}"

        is_baseline = " <<<" if name == "baseline" else ""
        pauc_str = f"{pauc:.6f}" if not math.isnan(pauc) else "   FAILED"
        print(f"  {rank:<4} {name:<25} {pauc_str:>10} {elapsed:>7.0f}s {delta_str:>14}{is_baseline}")

    print(f"{'=' * 72}")


def run_phase1(
    experiments: dict[str, dict],
) -> list[tuple[str, float, float]]:
    """Phase 1: Sang loc nhanh."""
    base = _screen_base()
    results: list[tuple[str, float, float]] = []
    total = len(experiments)

    print("\n" + "=" * 72)
    print(f"  PHASE 1 — SCREEN: {total} experiments × 1 fold × 3 epochs × 120 steps")
    print(f"  Muc dich: loai bo cau hinh te, giu lai top candidates")
    print("=" * 72)

    for idx, (name, overrides) in enumerate(experiments.items(), 1):
        print(f"\n--- [{idx}/{total}] {name} ---")
        if overrides:
            print(f"    Overrides: {overrides}")
        result = _run_one(name, overrides, base)
        results.append(result)
        _, pauc, elapsed = result
        pauc_str = f"{pauc:.6f}" if not math.isnan(pauc) else "FAILED"
        print(f">>> {name}: pAUC = {pauc_str} ({elapsed:.0f}s)")

    baseline_pauc = None
    for name, pauc, _ in results:
        if name == "baseline":
            baseline_pauc = pauc
            break

    _print_table(results, "PHASE 1 RESULTS (Screen)", baseline_pauc)
    return results


def run_phase2(
    experiments: dict[str, dict],
    top_names: list[str],
) -> list[tuple[str, float, float]]:
    """Phase 2: Xac nhan top candidates bang 3 folds, full steps."""
    base = _confirm_base()

    # Luon bao gom baseline
    selected = {"baseline": experiments.get("baseline", {})}
    for name in top_names:
        if name != "baseline" and name in experiments:
            selected[name] = experiments[name]

    total = len(selected)
    results: list[tuple[str, float, float]] = []

    print("\n" + "=" * 72)
    print(f"  PHASE 2 — CONFIRM: {total} experiments × 3 folds × 5 epochs × full steps")
    print(f"  Muc dich: xac nhan ket qua dang tin cay voi nhieu folds")
    print(f"  Candidates: {list(selected.keys())}")
    print("=" * 72)

    for idx, (name, overrides) in enumerate(selected.items(), 1):
        print(f"\n--- [{idx}/{total}] {name} ---")
        if overrides:
            print(f"    Overrides: {overrides}")
        result = _run_one(name, overrides, base)
        results.append(result)
        _, pauc, elapsed = result
        pauc_str = f"{pauc:.6f}" if not math.isnan(pauc) else "FAILED"
        print(f">>> {name}: pAUC = {pauc_str} ({elapsed:.0f}s)")

    baseline_pauc = None
    for name, pauc, _ in results:
        if name == "baseline":
            baseline_pauc = pauc
            break

    _print_table(results, "PHASE 2 RESULTS (Confirm — 3 folds, full steps)", baseline_pauc)
    return results


def _save_results(results: list[tuple[str, float, float]], path: str) -> None:
    """Luu ket qua ra JSON de xem lai sau."""
    out = [{"name": n, "pauc": p, "time_s": t} for n, p, t in results]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Results saved to: {path}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation 2 giai doan cho Mid Fusion Lite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_ablation.py                    # Phase 1 only (nhanh)
  python quick_ablation.py --phase 2 --top 3  # Phase 1 + confirm top 3
  python quick_ablation.py --only baseline,combo_simple
        """,
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="1 = chi sang loc nhanh, 2 = sang loc + xac nhan (default: 1)",
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="So luong top candidates dua vao Phase 2 (default: 3)",
    )
    parser.add_argument(
        "--only", default=None,
        help="Chi chay mot so thi nghiem (comma-separated names)",
    )
    args = parser.parse_args()

    # Filter experiments
    exps = dict(EXPERIMENTS)
    if args.only:
        selected = [n.strip() for n in args.only.split(",")]
        exps = {k: v for k, v in exps.items() if k in selected}
        if not exps:
            print(f"Khong tim thay thi nghiem nao. Co san: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        if "baseline" not in exps and "baseline" in EXPERIMENTS:
            exps = {"baseline": EXPERIMENTS["baseline"], **exps}

    # Phase 1
    phase1_results = run_phase1(exps)
    _save_results(phase1_results, "./mid_fusion_lite_outputs/ablation_screen/results.json")

    # Phase 2 (optional)
    if args.phase >= 2:
        valid_results = [(n, p, t) for n, p, t in phase1_results if not math.isnan(p)]
        sorted_by_pauc = sorted(valid_results, key=lambda x: -x[1])
        top_names = [name for name, _, _ in sorted_by_pauc[: args.top]]

        print(f"\n>>> Top {args.top} tu Phase 1: {top_names}")
        phase2_results = run_phase2(exps, top_names)
        _save_results(phase2_results, "./mid_fusion_lite_outputs/ablation_confirm/results.json")

    print("\nDone!")
