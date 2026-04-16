from __future__ import annotations

from io import BytesIO

import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def get_image_transforms(image_size: int) -> dict[str, A.Compose]:
    train_tfms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.06,
                scale_limit=0.08,
                rotate_limit=20,
                border_mode=0,
                p=0.6,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )
    valid_tfms = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )
    return {"train": train_tfms, "valid": valid_tfms}


class HDF5ImageTabularDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        tabular: np.ndarray,
        hdf5_path: str,
        targets: np.ndarray | None = None,
        transforms: A.Compose | None = None,
    ) -> None:
        self.ids = np.asarray(ids).astype(str)
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.targets = (
            None if targets is None else np.asarray(targets, dtype=np.float32)
        )
        self.hdf5_path = hdf5_path
        self.transforms = transforms
        self._h5: h5py.File | None = None

        if len(self.ids) != len(self.tabular):
            raise ValueError("ids and tabular must have the same length")
        if self.targets is not None and len(self.targets) != len(self.ids):
            raise ValueError("targets and ids must have the same length")

    def __len__(self) -> int:
        return len(self.ids)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _read_image(self, isic_id: str) -> np.ndarray:
        raw_obj = self._get_h5()[isic_id][()]
        if isinstance(raw_obj, np.ndarray):
            binary = raw_obj.tobytes()
        else:
            binary = bytes(raw_obj)

        image = Image.open(BytesIO(binary)).convert("RGB")
        return np.asarray(image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        isic_id = self.ids[index]
        image = self._read_image(isic_id)
        tabular = self.tabular[index]

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        sample: dict[str, torch.Tensor | str] = {
            "image": image,
            "tabular": torch.from_numpy(tabular),
            "isic_id": isic_id,
        }

        if self.targets is not None:
            sample["target"] = torch.tensor(self.targets[index], dtype=torch.float32)
        return sample

    def __del__(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def make_balanced_sampler(
    targets: np.ndarray,
    negative_ratio: int,
    seed: int,
) -> WeightedRandomSampler | None:
    labels = np.asarray(targets, dtype=np.int64)
    pos_mask = labels == 1
    neg_mask = labels == 0

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    weights = np.zeros(len(labels), dtype=np.float64)
    weights[pos_mask] = 1.0
    weights[neg_mask] = (n_pos * max(1, negative_ratio)) / max(1, n_neg)

    num_samples = int(n_pos * (1 + max(1, negative_ratio)))
    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
