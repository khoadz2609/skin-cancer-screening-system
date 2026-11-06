import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path to import BEST_WITH_CONCAT
sys.path.append(str(Path(__file__).parent))

from mid_fusion_lite.config import get_preset_config, BASE_NUMERIC_COLS
from mid_fusion_lite.tabular_engineering_02 import add_feature_engineering
from mid_fusion_lite.tabular_preprocessor_04 import FoldTabularPreprocessor, select_available_columns
from sklearn.model_selection import StratifiedGroupKFold

CACHE_DIR = Path(__file__).parent / "preprocessor_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_preprocessor_cache_path(preset_name: str, fold_idx: int) -> Path:
    return CACHE_DIR / f"preprocessor_{preset_name}_fold_{fold_idx}.pkl"

def load_or_create_preprocessors(preset_name: str, n_splits: int, fold_indices: list[int]) -> dict[int, FoldTabularPreprocessor]:
    """Load cached preprocessors or fit them on the training metadata and cache them."""
    preprocessors = {}
    missing_folds = []
    
    # Check what is missing
    for idx in fold_indices:
        cache_path = get_preprocessor_cache_path(preset_name, idx)
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    preprocessors[idx] = pickle.load(f)
                print(f"Loaded cached preprocessor for preset '{preset_name}' fold {idx}")
            except Exception as e:
                print(f"Failed to load cached preprocessor for preset '{preset_name}' fold {idx}: {e}")
                missing_folds.append(idx)
        else:
            missing_folds.append(idx)
            
    if not missing_folds:
        return preprocessors
        
    print(f"Fitting missing preprocessors for folds {missing_folds} of preset '{preset_name}'...")
    
    # Find train-metadata.csv
    # Try different locations
    candidates = [
        Path(__file__).parent.parent.parent / "data" / "train-metadata.csv",
        Path(__file__).parent.parent / "data" / "train-metadata.csv",
        Path(__file__).parent.parent / "train-metadata.csv",
        Path("data/train-metadata.csv"),
    ]
    train_metadata_path = None
    for c in candidates:
        if c.exists():
            train_metadata_path = c
            break
            
    if train_metadata_path is None:
        raise FileNotFoundError(
            f"Could not find train-metadata.csv to fit the preprocessors. Checked: {[str(c) for c in candidates]}"
        )
        
    print(f"Reading training metadata from {train_metadata_path}...")
    
    # We load only the required columns to save memory
    config = get_preset_config(preset_name if preset_name != "smoke_test" else "smoke")
    
    # Override configurations to match the trained checkpoints
    if preset_name == "full_lite":
        config = config.with_updates(
            image_size=256,
            tabm_k=16,
            tabm_n_blocks=3,
            tabm_d_block=256,
            tab_dropout=0.15,
            backbone_name="efficientnet_b0",
            tab_hidden_dim=192,
            fusion_hidden_dim=320,
        )
    
    requested_numeric_cols = set(config.numeric_cols)
    requested_numeric_cols.update(BASE_NUMERIC_COLS)
    
    requested_cols = {
        config.id_col,
        config.target_col,
        config.group_col,
        "tbp_lv_nevi_confidence",
        *config.categorical_cols,
        *requested_numeric_cols
    }
    
    # Read CSV
    train_df = pd.read_csv(train_metadata_path, usecols=lambda c: c in requested_cols)
    
    print("Applying feature engineering to training metadata...")
    # Create a dummy test df because add_feature_engineering expects both train and test
    dummy_test_df = train_df.iloc[:2].copy()
    
    train_df_engineered, _ = add_feature_engineering(
        train_df=train_df,
        test_df=dummy_test_df,
        group_col=config.group_col,
        categorical_cols=config.categorical_cols,
    )
    
    del train_df
    import gc
    gc.collect()
    
    # Get available numeric/categorical columns
    numeric_cols, categorical_cols = select_available_columns(
        train_df=train_df_engineered,
        test_df=train_df_engineered,
        numeric_candidates=config.numeric_cols,
        categorical_candidates=config.categorical_cols,
    )
    
    y = train_df_engineered[config.target_col].astype(np.int64).to_numpy()
    groups = train_df_engineered[config.group_col].to_numpy()
    
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    folds = list(splitter.split(train_df_engineered, y, groups))
    
    # Fit the missing preprocessors
    for fold_index in missing_folds:
        if fold_index >= len(folds):
            print(f"Warning: requested fold_index {fold_index} is out of bounds for n_splits {n_splits}")
            continue
            
        print(f"Fitting preprocessor for Fold {fold_index}...")
        train_idx, _ = folds[fold_index]
        fold_train_df = train_df_engineered.iloc[train_idx].reset_index(drop=True)
        
        # Ensure fold_train_df has all categories present in the entire dataset by appending one representative row for each unique category value
        extra_rows = []
        for col in categorical_cols:
            if col in train_df_engineered.columns:
                unique_cat_rows = train_df_engineered.groupby(col, observed=False).first().reset_index()
                extra_rows.append(unique_cat_rows)
        if extra_rows:
            extra_df = pd.concat(extra_rows, ignore_index=True)
            extra_df = extra_df[train_df_engineered.columns]
            fold_train_df = pd.concat([fold_train_df, extra_df], ignore_index=True)
        
        preprocessor = FoldTabularPreprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            use_quantile_transform=(
                bool(config.use_quantile_transform) and bool(config.use_tabm_tabular)
            ),
            quantile_noise_std=float(config.quantile_noise_std),
            quantile_random_state=config.seed + fold_index + 1,
            categorical_as_codes=bool(config.use_tabm_tabular),
        )
        preprocessor.fit(fold_train_df)
        
        # Compute bins if using TabM
        if getattr(config, "use_tabm_tabular", False) and preprocessor.n_numeric_out > 0:
            import torch
            import rtdl_num_embeddings
            import gc
            
            gc.collect()
            num_x = preprocessor._transform_numeric(fold_train_df)
            x_train_tensor = torch.tensor(num_x, dtype=torch.float32)
            del num_x
            gc.collect()
            
            print(f"Computing bins for Fold {fold_index}...")
            raw_bins = rtdl_num_embeddings.compute_bins(
                x_train_tensor,
                n_bins=config.n_bins
            )
            bins = []
            for b in raw_bins:
                b_wide = b.clone()
                if len(b_wide) > 1:
                    span = b_wide[-1] - b_wide[0]
                    margin = max(1e-3, float(0.1 * span))
                    b_wide[0] -= margin
                    b_wide[-1] += margin
                bins.append(b_wide)
            preprocessor.bins = bins
        else:
            preprocessor.bins = None
        
        # Save cache
        cache_path = get_preprocessor_cache_path(preset_name, fold_index)
        with open(cache_path, "wb") as f:
            pickle.dump(preprocessor, f)
            
        print(f"Saved fitted preprocessor to {cache_path}")
        preprocessors[fold_index] = preprocessor
        
        # Free memory before next iteration
        del fold_train_df
        gc.collect()
        
    return preprocessors
