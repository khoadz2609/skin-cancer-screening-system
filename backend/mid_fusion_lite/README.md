# Mid Fusion Lite (Kaggle-first)

This folder contains a self-contained multimodal pipeline for ISIC 2024:
- image branch (EfficientNet-B0 via timm)
- tabular branch (TABM by default, with MLP fallback)
- mid-fusion head (concat or cross-modal attention)

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
python -m mid_fusion_lite.runner --preset full_lite
```

## Download backbone weights (offline Kaggle use)

```bash
python download_weight.py --backbone-name convnextv2_nano --output convnextv2_nano_backbone.pth
```

Then upload the `.pth` file to your Kaggle dataset and set `BACKBONE_WEIGHTS_PATH` to that file path.

## Notes

- Split is done with `StratifiedGroupKFold` on `patient_id`.
- Tabular feature engineering now includes richer cross-context statistics: patient-normalized, patient-location-normalized, patient-location-simple-normalized, and patient-site-normalized features.
- Added Top-solution style features for tabular branch: anatomical z-scores, area aggregates (`tbp_lv_areaMM2_patient`, `tbp_lv_areaMM2_bp`), ugly-duckling location/percentile features, and additional morphology/color composites.
- Optional heavy tabular options are available in config: patient-level LOF feature extraction (`use_patient_lof_features`) and null-importance-style numeric filtering (`use_null_importance_filter`).
- Tabular preprocessing (scaler + one-hot) is fitted only on train fold.
- Fold-wise QuantileTransformer is applied on tabular matrix when `use_tabm_tabular=True` and `use_quantile_transform=True`.
- Train sampler limits negatives per epoch using `negative_ratio` for faster iterations.
- Validation always uses full validation fold (no resampling).
- TABM is the default tabular encoder (`use_tabm_tabular=True` in `config.py`).
- You can enable cross-modal attention with `fusion_mode='cross_attention'`.
