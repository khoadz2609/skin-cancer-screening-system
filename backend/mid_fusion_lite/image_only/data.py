import os
from io import BytesIO
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def shades_of_gray(img_rgb: np.ndarray, power: int = 6, gamma: float | None = None) -> np.ndarray:
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
    scale = mean_gray / (illum + 1e-8)
    img_normalized = img_float * scale[None, None, :]
    return np.clip(img_normalized, 0, 255).astype(img_dtype)

def get_image_transforms(image_size: int) -> dict[str, A.Compose]:
    train_tfms = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=20, border_mode=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ])
    valid_tfms = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
        ToTensorV2(),
    ])
    return {"train": train_tfms, "valid": valid_tfms}

class StandaloneImageDataset(Dataset):
    def __init__(
        self,
        ids: np.ndarray,
        image_dir: str | None = None,
        hdf5_path: str | None = None,
        targets: np.ndarray | None = None,
        transforms: A.Compose | None = None,
        use_shades_of_gray: bool = False,
        cache_size: int = 1000,
    ) -> None:
        self.ids = np.asarray(ids).astype(str)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.hdf5_path = Path(hdf5_path) if hdf5_path is not None else None
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        self.transforms = transforms
        self.use_shades_of_gray = use_shades_of_gray
        self.cache_size = cache_size
        self.image_cache: dict[str, np.ndarray] = {}
        self._h5: h5py.File | None = None

        if self.image_dir is None and self.hdf5_path is None:
            raise ValueError("Either image_dir or hdf5_path must be specified.")

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _read_image(self, isic_id: str) -> np.ndarray:
        if isic_id in self.image_cache:
            return self.image_cache[isic_id]

        image_array = None

        # 1. Try reading from directory first
        if self.image_dir is not None and self.image_dir.exists():
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                p = self.image_dir / f"{isic_id}{ext}"
                if p.exists():
                    try:
                        image = Image.open(p).convert("RGB")
                        image_array = np.asarray(image)
                        break
                    except Exception as e:
                        print(f"Error reading image file {p}: {e}")

        # 2. Try HDF5 if directory check failed
        if image_array is None and self.hdf5_path is not None and self.hdf5_path.exists():
            try:
                h5_file = self._get_h5()
                if isic_id in h5_file:
                    raw_obj = h5_file[isic_id][()]
                    if isinstance(raw_obj, np.ndarray):
                        binary = raw_obj.tobytes()
                    else:
                        binary = bytes(raw_obj)
                    image = Image.open(BytesIO(binary)).convert("RGB")
                    image_array = np.asarray(image)
            except Exception as e:
                print(f"Error reading image {isic_id} from HDF5 {self.hdf5_path}: {e}")

        if image_array is None:
            print(f"Warning: Image {isic_id} not found in folder or HDF5. Using white placeholder.")
            image_array = np.ones((224, 224, 3), dtype=np.uint8) * 255

        if self.use_shades_of_gray:
            image_array = shades_of_gray(image_array)

        if len(self.image_cache) < self.cache_size:
            self.image_cache[isic_id] = image_array

        return image_array

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        isic_id = self.ids[index]
        image_array = self._read_image(isic_id)

        if self.transforms is not None:
            image_tensor = self.transforms(image=image_array)["image"]
        else:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        sample: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "isic_id": isic_id,
        }

        if self.targets is not None:
            sample["target"] = torch.tensor(self.targets[index], dtype=torch.float32)

        return sample

    def __del__(self) -> None:
        self.image_cache.clear()
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
    target_neg_sum = n_pos * max(1, negative_ratio)
    weights[neg_mask] = target_neg_sum / max(1, n_neg)

    num_samples = int(n_pos * (1 + max(1, negative_ratio)))
    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
