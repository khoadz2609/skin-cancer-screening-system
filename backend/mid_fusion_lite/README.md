# Mid Fusion Lite (Kaggle-first)

This folder contains a self-contained multimodal pipeline for ISIC 2024:
- image branch (light timm backbone)
- tabular branch (scaled numeric + one-hot categorical)
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

- `smoke`: 1 fold only, very fast sanity run
- `quick`: 3 folds, lightweight model for rapid validation
- `full_lite`: 5 folds, still lightweight but stronger than quick

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
```

## Notes

- Split is done with `StratifiedGroupKFold` on `patient_id`.
- Tabular preprocessing is fitted only on train fold.
- Train sampler limits negatives per epoch using `negative_ratio` for faster iterations.
- Validation always uses full validation fold (no resampling).
