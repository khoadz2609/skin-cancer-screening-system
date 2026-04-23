from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch


def download_backbone_weights(backbone_name: str, output_path: str | None = None) -> Path:
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=0,
        global_pool="avg",
    )

    if output_path is None or str(output_path).strip() == "":
        output_file = Path(f"{backbone_name}_backbone.pth")
    else:
        output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_file)
    print(f"Saved backbone weights: {output_file.resolve()}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download timm pretrained backbone and save local .pth state_dict"
    )
    parser.add_argument(
        "--backbone-name",
        default="convnextv2_nano",
        help="timm model name, e.g. convnextv2_nano, efficientnet_b0, swinv2_small_window8_256",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pth path. Default: <backbone-name>_backbone.pth",
    )
    args = parser.parse_args()

    download_backbone_weights(backbone_name=args.backbone_name, output_path=args.output)


if __name__ == "__main__":
    main()
