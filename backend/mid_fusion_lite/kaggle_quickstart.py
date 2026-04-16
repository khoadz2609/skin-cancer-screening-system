from __future__ import annotations

import argparse

from runner import run_with_preset

# Usage in Kaggle notebook:
# 1) Copy this folder into /kaggle/working/...
# 2) Run: !python /kaggle/working/isic-2024/mid_fusion_lite/kaggle_quickstart.py --preset quick


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick launcher for Mid Fusion Lite")
    parser.add_argument(
        "--preset", default="quick", choices=["smoke", "quick", "full_lite"]
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--folds", default=None, help="Comma-separated fold ids, e.g. 0,1"
    )
    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    if args.folds is not None:
        overrides["fold_indices"] = [
            int(x.strip()) for x in args.folds.split(",") if x.strip()
        ]

    result = run_with_preset(args.preset, **overrides)
    print("CV pAUC:", result["cv_pauc"])
    print("Submission:", result["submission_path"])


if __name__ == "__main__":
    main()
