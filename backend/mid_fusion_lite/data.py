from __future__ import annotations

from io import BytesIO

import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


def shades_of_gray(img_rgb: np.ndarray, power: int = 6, gamma: float | None = None) -> np.ndarray:
    """Apply Shades of Gray color constancy algorithm to normalize lighting."""
    img_dtype = img_rgb.dtype
    img_float = img_rgb.astype(np.float32)
    
    if gamma is not None:
        img_float = np.power(img_float, gamma)
        
    img_power = np.power(img_float, power)
    illum = np.power(np.mean(img_power, axis=(0, 1)), 1.0 / power)
    
    illum_norm = np.sqrt(np.sum(np.square(illum)))
    if illum_norm > 0:
        illum = illum / illum_norm
        
    mean_gray = 1.0 / np.sqrt(3.0)
    
    # Scale channels
    scale = mean_gray / (illum + 1e-8)
    img_normalized = img_float * scale[None, None, :]
    
    # Clip and cast
    return np.clip(img_normalized, 0, 255).astype(img_dtype)


def get_image_transforms(image_size: int) -> dict[str, A.Compose]:
    train_tfms = A.Compose(
        [  # type: ignore[arg-type]
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
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )
    valid_tfms = A.Compose(
        [  # type: ignore[arg-type]
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
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
        cache_size: int = 1000,
        use_shades_of_gray: bool = False,
    ) -> None:
        self.ids = np.asarray(ids).astype(str)
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.targets = (
            None if targets is None else np.asarray(targets, dtype=np.float32)
        )
        self.hdf5_path = hdf5_path
        self.transforms = transforms
        self._h5: h5py.File | None = None
        self.cache_size = max(1, cache_size)
        self.image_cache: dict[str, np.ndarray] = {}
        self.use_shades_of_gray = use_shades_of_gray

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
        # Check cache first (cache hit: 0.1ms vs miss: 5-10ms)
        if isic_id in self.image_cache:
            return self.image_cache[isic_id]

        raw_obj = self._get_h5()[isic_id][()]  # type: ignore[index]
        if isinstance(raw_obj, np.ndarray):
            binary = raw_obj.tobytes()
        else:
            binary = bytes(raw_obj)  # type: ignore[arg-type]

        image = Image.open(BytesIO(binary)).convert("RGB")
        image_array = np.asarray(image)

        if self.use_shades_of_gray:
            image_array = shades_of_gray(image_array)

        # Cache image if space available (LRU-like strategy)
        if len(self.image_cache) < self.cache_size:
            self.image_cache[isic_id] = image_array

        return image_array

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
        self.image_cache.clear()  # Clear cache on cleanup
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def make_balanced_sampler(
    targets: np.ndarray,
    negative_ratio: int,
    seed: int,
    hard_negative_weights: np.ndarray | None = None,
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

    target_neg_sum = n_pos * max(1, negative_ratio)

    if hard_negative_weights is not None:
        neg_w = np.asarray(hard_negative_weights, dtype=np.float64)[neg_mask]
        # Normalize the custom weights so their sum equals the desired total negative weight
        if neg_w.sum() > 0:
            neg_w = (neg_w / neg_w.sum()) * target_neg_sum
        else:
            neg_w = target_neg_sum / max(1, n_neg)
        weights[neg_mask] = neg_w
    else:
        weights[neg_mask] = target_neg_sum / max(1, n_neg)

    num_samples = int(n_pos * (1 + max(1, negative_ratio)))
    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),  # type: ignore[arg-type]
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
