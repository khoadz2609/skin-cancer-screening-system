import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

# Add parent directory of 'mid_fusion_lite' to import modules if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from mid_fusion_lite.image_only.model import StandaloneImageModel
from mid_fusion_lite.image_only.data import StandaloneImageDataset, get_image_transforms, make_balanced_sampler
from mid_fusion_lite.image_only.train import fit_one_fold, evaluate, predict_loader, partial_auc_score

def get_args():
    parser = argparse.ArgumentParser(description="Standalone Image Model Training & Inference Pipeline")
    parser.add_argument("--metadata-csv", type=str, required=True, help="Path to metadata CSV file")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing raw JPEG/PNG images")
    parser.add_argument("--hdf5-path", type=str, default=None, help="Path to HDF5 file containing images")
    parser.add_argument("--output-dir", type=str, default="./image_only_outputs", help="Directory to save output checkpoints and logs")
    
    # Model configuration
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", help="Timm backbone model name")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution for image resizing")
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained ImageNet weights")
    parser.add_argument("--weights-path", type=str, default=None, help="Load local model checkpoint (.pt)")
    parser.add_argument("--backbone-weights-path", type=str, default=None, help="Path to pre-downloaded local backbone weights (.pth/.pt)")
    
    # Training hyperparameters
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default="train", help="Mode: train a new model or run prediction")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision training (AMP)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--negative-ratio", type=int, default=4, help="Negative sampler ratio (for balanced sampling)")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Read CSV
    print(f"Loading metadata from {args.metadata_csv}...")
    df = pd.read_csv(args.metadata_csv)
    
    # Check essential columns
    id_col = "isic_id"
    target_col = "target"
    group_col = "patient_id"
    
    if id_col not in df.columns:
        raise ValueError(f"CSV metadata must contain '{id_col}' column.")
        
    if args.mode == "train":
        if target_col not in df.columns:
            raise ValueError(f"CSV metadata must contain '{target_col}' column for training.")
        if group_col not in df.columns:
            print(f"Warning: '{group_col}' not found. Grouping folds randomly.")
            df[group_col] = df[id_col]
            
        # Stratified Group K-Fold split
        print(f"Splitting dataset into {args.folds} folds...")
        sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        df["fold"] = -1
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df[target_col], groups=df[group_col])):
            df.loc[val_idx, "fold"] = fold
            
        oof_preds = np.zeros(len(df), dtype=np.float32)
        history_all = {}
        fold_scores = []
        
        transforms = get_image_transforms(args.image_size)
        
        for fold in range(args.folds):
            print(f"\n=================== FOLD {fold} ===================")
            train_df = df[df["fold"] != fold].reset_index(drop=True)
            valid_df = df[df["fold"] == fold].reset_index(drop=True)
            
            # Setup Datasets
            train_dataset = StandaloneImageDataset(
                ids=train_df[id_col].to_numpy(),
                image_dir=args.image_dir,
                hdf5_path=args.hdf5_path,
                targets=train_df[target_col].to_numpy(),
                transforms=transforms["train"]
            )
            
            valid_dataset = StandaloneImageDataset(
                ids=valid_df[id_col].to_numpy(),
                image_dir=args.image_dir,
                hdf5_path=args.hdf5_path,
                targets=valid_df[target_col].to_numpy(),
                transforms=transforms["valid"]
            )
            
            # Balanced Sampler for imbalanced dataset
            sampler = make_balanced_sampler(
                targets=train_df[target_col].to_numpy(),
                negative_ratio=args.negative_ratio,
                seed=args.seed + fold
            )
            
            # Setup Dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=0,
                pin_memory=True
            )
            
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            # Initialize Model
            print(f"Initializing {args.backbone} model...")
            model = StandaloneImageModel(
                backbone_name=args.backbone,
                use_pretrained=args.pretrained,
                weights_path=args.weights_path,
                backbone_weights_path=args.backbone_weights_path
            )
            model = model.to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            pos_weight = 1.0  # Loss weight computed if needed
            
            # Fit
            best_state, fold_history, best_score = fit_one_fold(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=args.epochs,
                patience=args.patience,
                pos_weight=pos_weight,
                use_amp=args.use_amp
            )
            
            # Save Checkpoint
            checkpoint_path = output_dir / f"image_only_model_fold_{fold}.pt"
            torch.save(best_state, checkpoint_path)
            print(f"Saved best fold checkpoint to {checkpoint_path}")
            
            # Load best weights and predict OOF
            model.load_state_dict(best_state)
            fold_val_preds = predict_loader(model=model, loader=valid_loader, device=device, use_amp=args.use_amp)
            
            val_indices = df[df["fold"] == fold].index
            oof_preds[val_indices] = fold_val_preds
            
            # Calculate validation metric
            valid_stats = evaluate(
                model=model,
                loader=valid_loader,
                criterion=torch.nn.BCEWithLogitsLoss(),
                device=device,
                use_amp=args.use_amp
            )
            print(f"Fold {fold} Best pAUC: {best_score:.5f} | Final pAUC: {valid_stats['pauc']:.5f}")
            
            fold_scores.append(best_score)
            history_all[f"fold_{fold}"] = fold_history
            
        # Overall OOF Evaluation
        oof_pauc = partial_auc_score(df[target_col].to_numpy(), oof_preds)
        print(f"\n==========================================")
        print(f"Overall OOF partial AUC (pAUC): {oof_pauc:.5f}")
        print(f"Mean Fold pAUC: {np.mean(fold_scores):.5f}")
        
        # Save OOF predictions
        df["pred"] = oof_preds
        oof_csv_path = output_dir / "oof_predictions.csv"
        df[[id_col, target_col, "fold", "pred"]].to_csv(oof_csv_path, index=False)
        print(f"Saved OOF predictions to {oof_csv_path}")
        
        # Save summary metrics
        metrics = {
            "overall_oof_pauc": oof_pauc,
            "mean_fold_pauc": np.mean(fold_scores),
            "fold_scores": fold_scores,
            "args": vars(args)
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
    elif args.mode == "predict":
        print("Running prediction mode...")
        transforms = get_image_transforms(args.image_size)
        
        dataset = StandaloneImageDataset(
            ids=df[id_col].to_numpy(),
            image_dir=args.image_dir,
            hdf5_path=args.hdf5_path,
            transforms=transforms["valid"]
        )
        
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # Initialize Model
        model = StandaloneImageModel(
            backbone_name=args.backbone,
            weights_path=args.weights_path,
            backbone_weights_path=args.backbone_weights_path
        )
        model = model.to(device)
        
        print(f"Generating predictions...")
        preds = predict_loader(model=model, loader=loader, device=device, use_amp=args.use_amp, progress_name="prediction", log_interval=50)
        
        df["pred"] = preds
        sub_csv_path = output_dir / "submission_image_only.csv"
        df[[id_col, "pred"]].to_csv(sub_csv_path, index=False)
        print(f"Saved predictions to {sub_csv_path}")

if __name__ == "__main__":
    main()
