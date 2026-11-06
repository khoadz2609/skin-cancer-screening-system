"""Verify that the local model artifacts required by the web demo exist.

This script intentionally uses only the Python standard library so it can be
run before installing the ML dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def expected_artifacts() -> dict[str, list[Path]]:
    return {
        "fusion checkpoints": [
            ROOT / "backend" / "models" / f"mid_fusion_lite_fold_{fold}.pt"
            for fold in range(5)
        ]
        + [ROOT / "backend" / "models" / "metrics.json"],
        "image-only checkpoints": [
            ROOT / "backend" / "backbone_image" / f"image_only_fold_{fold}.pt"
            for fold in range(5)
        ]
        + [ROOT / "backend" / "backbone_image" / "metrics.json"],
        "fusion preprocessors (optional)": [
            ROOT / "backend" / "preprocessor_cache" / f"preprocessor_kaggle_top1_fusion_fold_{fold}.pkl"
            for fold in range(5)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-preprocessors",
        action="store_true",
        help="Fail when the cached fusion preprocessors are missing.",
    )
    args = parser.parse_args()

    groups = expected_artifacts()
    failed = False
    for group, paths in groups.items():
        optional = group.endswith("(optional)") and not args.require_preprocessors
        missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
        if missing and not optional:
            failed = True
            print(f"[MISSING] {group}")
            for path in missing:
                print(f"  - {path.relative_to(ROOT)}")
        elif missing:
            print(f"[WARN] {group} not found; fusion inference may need train-metadata.csv to recreate them.")
        else:
            print(f"[OK] {group}")

    if failed:
        print("\nDownload the checkpoints described in README.md, then run this command again.")
        return 1

    print("\nAll required model artifacts are available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
