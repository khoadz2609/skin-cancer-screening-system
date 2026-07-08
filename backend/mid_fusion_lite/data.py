from __future__ import annotations

import os
from io import BytesIO

import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, Sampler


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
            # === GEOMETRIC — safe for ABCD criteria (Perez et al. 2018) ===
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,  # Reduced from 0.85
            ),
            # === COLOR — most important augmentation group for dermoscopy ===
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.75,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5,
            ),
            A.CLAHE(clip_limit=4.0, p=0.5),  # Reduced from 0.7
            # === NOISE/BLUR — mild robustness, reduced probability ===
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(),
                ],
                p=0.3,  # Reduced from 0.7
            ),
            # REMOVED: OpticalDistortion, GridDistortion, ElasticTransform
            # Reason: distort_limit=1.0 destroys Border and Structures (ABCD)
            # and 2k images/epoch is insufficient to learn through heavy distortion
            # === REGULARIZATION ===
            A.Resize(image_size, image_size),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(1, max(2, int(image_size * 0.12))),  # Reduced from 0.25
                hole_width_range=(1, max(2, int(image_size * 0.12))),   # Reduced from 0.25
                p=0.4,  # Reduced from 0.7
            ),
            A.Normalize(
                mean=(0.4815, 0.4578, 0.4082),
                std=(0.2686, 0.2613, 0.2758),
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
                mean=(0.4815, 0.4578, 0.4082),
                std=(0.2686, 0.2613, 0.2758),
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
        modality_mode: str = "both",
        image_size: int = 160,
    ) -> None:
        self.ids = np.asarray(ids).astype(str)
        self.tabular = np.asarray(tabular, dtype=np.float32)
        self.targets = (
            None if targets is None else np.asarray(targets, dtype=np.float32)
        )
        self.hdf5_path = hdf5_path
        self.transforms = transforms
        self._h5: h5py.File | None = None
        self._pid = os.getpid()
        self.cache_size = max(1, cache_size)
        self.image_cache: dict[str, np.ndarray] = {}
        self.use_shades_of_gray = use_shades_of_gray
        self.modality_mode = str(modality_mode).strip().lower()
        self.image_size = image_size

        if len(self.ids) != len(self.tabular):
            raise ValueError("ids and tabular must have the same length")
        if self.targets is not None and len(self.targets) != len(self.ids):
            raise ValueError("targets and ids must have the same length")

    def __len__(self) -> int:
        return len(self.ids)

    def _get_h5(self) -> h5py.File:
        current_pid = os.getpid()
        if self._h5 is None or current_pid != self._pid:
            self._h5 = h5py.File(self.hdf5_path, "r")
            self._pid = current_pid
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
        tabular = self.tabular[index]

        if self.modality_mode == "tabular_only":
            # Return a fast dummy zero tensor to bypass expensive image loading and resizing
            image = torch.empty(0, dtype=torch.float32)
        else:
            image = self._read_image(isic_id)
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


class EpochRandomUndersampler(Sampler):
    """
    Custom PyTorch Sampler that dynamically draws a new random subset of negative
    indices at the start of every epoch, ensuring the model sees a varied negative
    distribution over time without replacement within a single epoch.
    """
    def __init__(self, targets: np.ndarray, negative_ratio: float, seed: int):
        labels = np.asarray(targets, dtype=np.int64)
        self.pos_idx = np.where(labels == 1)[0].tolist()
        self.neg_idx = np.where(labels == 0)[0].tolist()
        
        self.n_pos = len(self.pos_idx)
        n_neg_sampled = int(self.n_pos * negative_ratio)
        self.n_neg_sampled = min(n_neg_sampled, len(self.neg_idx))
        
        self.base_seed = seed
        self.epoch = 0

    def __iter__(self):
        # We vary the seed per epoch to ensure different samples
        rng = np.random.default_rng(self.base_seed + self.epoch)
        self.epoch += 1
        
        if self.n_pos == 0 or len(self.neg_idx) == 0:
            return iter([])

        sampled_neg_idx = rng.choice(self.neg_idx, size=self.n_neg_sampled, replace=False).tolist()
        
        keep_idx = self.pos_idx + sampled_neg_idx
        rng.shuffle(keep_idx)
        return iter(keep_idx)

    def __len__(self):
        if self.n_pos == 0 or len(self.neg_idx) == 0:
            return 0
        return self.n_pos + self.n_neg_sampled


def make_balanced_sampler(
    targets: np.ndarray,
    negative_ratio: float,
    seed: int,
    hard_negative_weights: np.ndarray | None = None,
) -> Sampler | None:
    """Returns an EpochRandomUndersampler for dynamically resampling negative class."""
    sampler = EpochRandomUndersampler(targets, negative_ratio, seed)
    if len(sampler) == 0:
        return None
    return sampler

