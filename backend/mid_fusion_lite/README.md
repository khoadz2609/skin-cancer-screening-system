# Mid Fusion Lite (Kaggle-first)

This folder contains a self-contained multimodal pipeline for ISIC 2024:
- image branch (EfficientNet-B0 via timm)
- tabular branch (TABM by default, with MLP fallback)
- mid-fusion head

It is designed for quick experiments on Kaggle with lightweight presets.

## Files in this folder

- `config.py`: presets and path resolution
- `tabular.py`: fold-wise tabular preprocessing (leak-safe)
- `data.py`: HDF5 image + tabular dataset, augmentations, balanced sampler
- `model.py`: Mid Fusion Lite model
- `train.py`: training loop, evaluation, pAUC metric
- `runner.py`: CV training + OOF + submission export

## Presets

- `smoke`: 1 fold only, very fast sanity run (EfficientNet-B0)
- `quick`: 3 folds, lightweight model for rapid validation (EfficientNet-B0)
- `full_lite`: 5 folds, still lightweight but stronger than quick (EfficientNet-B0)

## Kaggle usage (Notebook)

```python
import sys
sys.path.append('/kaggle/working/isic-2024/mid_fusion_lite')

from runner import run_with_preset

result = run_with_preset('quick')
print('CV pAUC:', result['cv_pauc'])
print('Submission file:', result['submission_path'])
```

If you copy this folder somewhere else, append that folder path to `sys.path`.

## Local usage (CLI)

```bash
python -m mid_fusion_lite.runner --preset quick
python -m mid_fusion_lite.runner --preset smoke --folds 0 --epochs 2
python -m mid_fusion_lite.runner --preset full_lite --use-data-parallel true --dp-device-ids 0,1
```

## Notes

- Split is done with `StratifiedGroupKFold` on `patient_id`.
- Tabular feature engineering now mirrors the TabM standalone notebook (engineered numerical features, patient-normalized features, per-patient category counts, and numeric-category interaction stats).
- Tabular preprocessing (scaler + one-hot) is fitted only on train fold.
- Fold-wise QuantileTransformer is applied on tabular matrix when `use_tabm_tabular=True` and `use_quantile_transform=True`.
- Train sampler limits negatives per epoch using `negative_ratio` for faster iterations.
- Validation always uses full validation fold (no resampling).
- TABM is the default tabular encoder (`use_tabm_tabular=True` in `config.py`).
- Multi-GPU is supported via `torch.nn.DataParallel` when `use_data_parallel=True` and at least 2 CUDA devices are visible.
- Limit DataParallel GPUs with `data_parallel_device_ids` (for example `[0, 1]`), or leave it as `None` to use all visible GPUs.
