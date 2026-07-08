import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedGroupKFold

def create_folds_and_save():
    # 1. Define paths for Kaggle
    input_csv = '/kaggle/input/competitions/isic-2024-challenge/train-metadata.csv'
    output_dir = '/kaggle/working'
    output_csv = os.path.join(output_dir, 'train_metadata_with_folds.csv')
    
    # Local PC fallback
    if not os.path.exists(input_csv):
        print(f"[Warning] {input_csv} not found. Checking local data folders...")
        local_candidates = [
            'data/train-metadata.csv',
            '../data/train-metadata.csv',
            'train-metadata.csv'
        ]
        for path in local_candidates:
            if os.path.exists(path):
                input_csv = path
                break
        else:
            input_csv = 'train-metadata.csv'
        output_dir = '.'
        output_csv = 'train_metadata_with_folds.csv'

    print(f"Reading train metadata from: {input_csv} ...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} does not exist. Stopping.")
        return

    # Configurations
    n_splits = 5
    seed = 42
    target_col = "target"
    group_col = "patient_id"

    # Fill NaN patient IDs (important for patient-level grouping)
    if df[group_col].isnull().any():
        df[group_col] = df[group_col].fillna('unknown_patient')
        
    y = df[target_col].astype(np.int64).to_numpy()
    groups = df[group_col].to_numpy()
    df['fold'] = -1
    
    # StratifiedGroupKFold by patient
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    print("Splitting folds by patient using StratifiedGroupKFold (seed=42)...")
    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(df, y, groups)):
        df.loc[valid_idx, 'fold'] = fold_idx

    # Print summary of each fold
    for fold_idx in range(n_splits):
        val_fold = df[df['fold'] == fold_idx]
        print(f"Fold {fold_idx}: Total={len(val_fold)}, Positive={val_fold[target_col].sum()}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    print(f"Saving output to: {output_csv} ...")
    df.to_csv(output_csv, index=False)
    print("SUCCESS! train_metadata_with_folds.csv is created.")

if __name__ == "__main__":
    create_folds_and_save()
