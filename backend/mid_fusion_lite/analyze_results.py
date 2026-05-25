#!/usr/bin/env python
"""
Script phân tích chi tiết kết quả từ Kaggle run.
Usage: python analyze_results.py --output-dir <path-to-outputs>
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve,
    auc as pr_auc,
)


def analyze_metrics(output_dir: str = "./mid_fusion_lite_outputs"):
    """Phân tích chi tiết metrics từ run."""
    output_path = Path(output_dir)
    
    print("\n" + "="*80)
    print("📊 DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load metrics
    metrics_file = output_path / "metrics.json"
    if not metrics_file.exists():
        print(f"❌ metrics.json not found at {metrics_file}")
        return
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Load OOF predictions
    oof_file = output_path / "oof_predictions.csv"
    if not oof_file.exists():
        print(f"❌ oof_predictions.csv not found at {oof_file}")
        return
    
    oof_df = pd.read_csv(oof_file)
    
    print("\n1️⃣ OVERALL CV METRICS")
    print("-" * 80)
    print(f"CV pAUC:          {metrics.get('cv_pauc', 'N/A')}")
    print(f"Number of folds:  {len(metrics.get('fold_metrics', []))}")
    
    # Fold-by-fold analysis
    print("\n2️⃣ FOLD-BY-FOLD BREAKDOWN")
    print("-" * 80)
    
    fold_metrics = metrics.get('fold_metrics', [])
    for fold in fold_metrics:
        print(f"Fold {fold['fold']}: pAUC={fold['best_valid_pauc']:.4f} | "
              f"Train={fold['n_train']:,} | Valid={fold['n_valid']:,} | "
              f"Epochs={fold['epochs_ran']}")
    
    # Variance analysis
    pauc_values = [f['best_valid_pauc'] for f in fold_metrics]
    print(f"\npAUC Statistics:")
    print(f"  Mean:   {np.mean(pauc_values):.4f}")
    print(f"  Std:    {np.std(pauc_values):.4f}")
    print(f"  Min:    {np.min(pauc_values):.4f}")
    print(f"  Max:    {np.max(pauc_values):.4f}")
    print(f"  Range:  {np.max(pauc_values) - np.min(pauc_values):.4f} ⚠️ (high variance suggests instability)")
    
    # Prediction analysis
    print("\n3️⃣ PREDICTION DISTRIBUTION")
    print("-" * 80)
    
    targets = oof_df['target'].values
    preds = oof_df['pred'].values
    
    print(f"Total samples: {len(preds):,}")
    print(f"  Negative (0): {(targets == 0).sum():,} ({100*(targets==0).mean():.1f}%)")
    print(f"  Positive (1): {(targets == 1).sum():,} ({100*(targets==1).mean():.1f}%)")
    
    print(f"\nPrediction ranges:")
    print(f"  Overall: {preds.min():.4f} - {preds.max():.4f}")
    print(f"  Negative (0): mean={preds[targets==0].mean():.4f}, "
          f"std={preds[targets==0].std():.4f}")
    print(f"  Positive (1): mean={preds[targets==1].mean():.4f}, "
          f"std={preds[targets==1].std():.4f}")
    
    sep_score = preds[targets==1].mean() - preds[targets==0].mean()
    print(f"\n  Separation score (pos_mean - neg_mean): {sep_score:.4f}")
    print(f"  ℹ️  Good separation should be > 0.3")
    
    if sep_score < 0.1:
        print(f"  ⚠️  WARNING: Very poor separation! Model predictions almost identical for both classes")
    elif sep_score < 0.2:
        print(f"  🟠 CONCERN: Weak separation - predictions not confident enough")
    else:
        print(f"  ✅ Acceptable separation")
    
    # Overlap analysis
    neg_preds = preds[targets == 0]
    pos_preds = preds[targets == 1]
    
    overlap = ((neg_preds >= pos_preds.mean() - 0.1) & (neg_preds <= pos_preds.mean() + 0.1)).sum()
    overlap_pct = 100 * overlap / len(neg_preds)
    print(f"\nPrediction overlap:")
    print(f"  {overlap:,} negative predictions near positive mean ({overlap_pct:.1f}%)")
    print(f"  ℹ️  Lower is better (less confusion)")
    
    # AUROC vs pAUC comparison
    auroc = roc_auc_score(targets, preds)
    print(f"\n4️⃣ AUROC VS pAUC COMPARISON")
    print("-" * 80)
    print(f"AUROC (full ROC):     {auroc:.4f}")
    print(f"pAUC (partial ROC):   {metrics['cv_pauc']:.4f}")
    print(f"Ratio (pAUC/AUROC):   {metrics['cv_pauc']/auroc:.4f}")
    
    if auroc > 0.90 and metrics['cv_pauc'] < 0.20:
        print(f"\n⚠️  HIGH CONCERN: Large gap indicates model is good at general classification")
        print(f"    but poor at high-sensitivity detection (which is critical for medical imaging)")
        print(f"\n    This suggests:")
        print(f"    1. Model overly conservative with positive predictions")
        print(f"    2. Imbalance handling is too aggressive")
        print(f"    3. Threshold might be miscalibrated")
    
    # Threshold analysis
    print(f"\n5️⃣ THRESHOLD ANALYSIS")
    print("-" * 80)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"{'Threshold':<12} {'TPR':<10} {'FPR':<10} {'Specificity':<12}")
    print("-" * 44)
    
    for thresh in thresholds:
        pred_binary = (preds >= thresh).astype(int)
        tp = ((pred_binary == 1) & (targets == 1)).sum()
        fp = ((pred_binary == 1) & (targets == 0)).sum()
        tn = ((pred_binary == 0) & (targets == 0)).sum()
        fn = ((pred_binary == 0) & (targets == 1)).sum()
        
        tpr = tp / max(tp + fn, 1)  # True positive rate (recall)
        fpr = fp / max(fp + tn, 1)  # False positive rate
        tnr = tn / max(tn + fp, 1)  # True negative rate (specificity)
        
        print(f"{thresh:<12.1f} {tpr:<10.4f} {fpr:<10.4f} {tnr:<12.4f}")
    
    print(f"\nℹ️  Goal: High TPR while keeping FPR low")
    print(f"    pAUC @ 80% TPR is the official metric - requires minimal FPR")
    
    return {
        'cv_pauc': metrics['cv_pauc'],
        'auroc': auroc,
        'separation': sep_score,
        'n_samples': len(preds),
    }


def suggest_improvements():
    """Suggest concrete improvements."""
    print("\n" + "="*80)
    print("💡 RECOMMENDED IMPROVEMENTS (Priority Order)")
    print("="*80)
    
    print("""
1️⃣  REDUCE IMBALANCE HANDLING AGGRESSIVENESS (15 min, HIGH impact)
    Current: negative_ratio=10, pos_weight=~10x
    Reason: This double-penalizes negative samples
    Try:
      - negative_ratio: 5 (instead of 10)
      - pos_weight: sqrt(n_neg/n_pos) instead of n_neg/n_pos
      - Or sample_weight-based balancing instead of pos_weight
    
    Test: Run 1 fold with new config, compare pAUC
    Expected impact: +0.02-0.05 pAUC

2️⃣  INCREASE TRAINING STABILITY (10 min, MEDIUM impact)
    Current: patience=3, some folds stop at epoch 8-13
    Try:
      - patience: 10 (instead of 3)
      - Or patience: min(5, epochs // 2)
    
    Test: Monitor if folds train full 20 epochs
    Expected impact: +0.01-0.03 pAUC

3️⃣  IMPROVE THRESHOLD CALIBRATION (20 min, MEDIUM impact)
    Current: Uses default threshold=0.5
    Try:
      - Find optimal threshold that maximizes pAUC @ 80% TPR
      - Different threshold per fold (post-hoc calibration)
      - Use validation set to pick threshold
    
    Test: Calculate pAUC with optimal thresholds
    Expected impact: +0.02-0.08 pAUC

4️⃣  TRY FOCAL LOSS (1 hour, MEDIUM impact)
    Current: BCEWithLogitsLoss with pos_weight
    Try: Focal Loss (alpha=0.25, gamma=2.0)
    Reason: Dynamically downweights easy samples, focuses on hard ones
    
    Expected impact: +0.01-0.04 pAUC

5️⃣  ARCHITECTURE TUNING (2-4 hours, LOW-MEDIUM impact)
    Current: convnextv2_nano + TABM + cross-attention
    Try:
      - Increase model capacity if underfitting
      - Switch fusion_mode from 'cross_attention' to 'concat' (simpler)
      - Selective unfreezing of image backbone layers
      - Different LR for image vs tabular branch
    
    Expected impact: +0.01-0.03 pAUC

6️⃣  FEATURE ENGINEERING REVIEW (1 hour, MEDIUM impact)
    Current: Many engineered features enabled
    Try:
      - Check feature importance → drop noisy features
      - Reduce feature space to keep signal-to-noise high
      - Or disable feature engineering to focus on images
    
    Expected impact: +0.01-0.02 pAUC
""")


def main():
    parser = argparse.ArgumentParser(description="Analyze Mid-Fusion Lite results")
    parser.add_argument("--output-dir", default="./mid_fusion_lite_outputs",
                       help="Path to output directory")
    args = parser.parse_args()
    
    results = analyze_metrics(args.output_dir)
    suggest_improvements()
    
    print("\n" + "="*80)
    print("📌 NEXT STEPS")
    print("="*80)
    print("""
1. Run diagnostic with optimal changes:
   python analyze_results.py  # See current baseline
   
2. Edit config.py or create a new preset with improvements
   
3. Test with smoke preset first:
   python runner.py --preset smoke --folds 0 --epochs 3
   
4. If improvement > 0.01 pAUC, commit and test full
   python runner.py --preset full_lite
""")


if __name__ == "__main__":
    main()
