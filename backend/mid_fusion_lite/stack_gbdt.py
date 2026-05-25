#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from .config import (
        BASE_NUMERIC_COLS,
        MidFusionLiteConfig,
        get_preset_config,
        resolve_data_paths,
    )
    from .tabular import FoldTabularPreprocessor, add_feature_engineering, select_available_columns
    from .train import partial_auc_score
except ImportError:
    from config import (
        BASE_NUMERIC_COLS,
        MidFusionLiteConfig,
        get_preset_config,
        resolve_data_paths,
    )
    from tabular import FoldTabularPreprocessor, add_feature_engineering, select_available_columns
    from train import partial_auc_score


def _read_csv_with_fallback(path: str, requested_cols: set[str]) -> pd.DataFrame:
    header_cols = list(pd.read_csv(path, nrows=0).columns)
    selected_cols = [col for col in header_cols if col in requested_cols]

    read_kwargs: dict[str, Any] = {"low_memory": True}
    if selected_cols:
        read_kwargs["usecols"] = selected_cols

    try:
        return pd.read_csv(path, **read_kwargs)
    except pd.errors.ParserError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(
            f"Parser out-of-memory while reading {path}. "
            "Retrying with python engine."
        )
        retry_kwargs = dict(read_kwargs)
        retry_kwargs["engine"] = "python"
        return pd.read_csv(path, **retry_kwargs)


def _filter_supported_config_updates(
    config: MidFusionLiteConfig,
    updates: dict[str, Any],
) -> dict[str, Any]:
    if not updates:
        return updates

    supported_keys = {field.name for field in dataclass_fields(type(config))}
    filtered = {k: v for k, v in updates.items() if k in supported_keys}
    dropped = sorted(set(updates) - set(filtered))
    if dropped:
        print("Ignoring unsupported config overrides: " + ", ".join(dropped))
    return filtered


def _extract_oof_predictions(oof_path: Path, id_col: str) -> pd.DataFrame:
    oof_df = pd.read_csv(oof_path)
    if "oof_pred" in oof_df.columns:
        pred_col = "oof_pred"
    elif "pred" in oof_df.columns:
        pred_col = "pred"
    else:
        raise KeyError(
            "OOF file must contain 'oof_pred' (preferred) or 'pred' column."
        )

    used_col = "used_in_cv" if "used_in_cv" in oof_df.columns else None
    keep_cols = [id_col, pred_col]
    if used_col is not None:
        keep_cols.append(used_col)

    oof_df = oof_df[keep_cols].copy()
    oof_df = oof_df.rename(columns={pred_col: "fusion_pred"})
    if used_col is None:
        oof_df["used_in_cv"] = 1
    else:
        oof_df["used_in_cv"] = oof_df["used_in_cv"].fillna(0).astype(int)
    return oof_df


def _extract_test_predictions(submission_path: Path, id_col: str) -> pd.DataFrame:
    submission_df = pd.read_csv(submission_path)
    if "target" not in submission_df.columns:
        raise KeyError("Submission file must contain 'target' column.")
    return submission_df[[id_col, "target"]].rename(columns={"target": "fusion_pred"})


def run_gbdt_stack(
    preset_name: str,
    output_dir: str,
    stack_output_dir: str,
    feature_mode: str,
    gbdt_params: dict[str, Any],
    **override_kwargs,
) -> dict[str, Any]:
    config = get_preset_config(preset_name)
    if override_kwargs:
        filtered_updates = _filter_supported_config_updates(config, override_kwargs)
        if filtered_updates:
            config = config.with_updates(**filtered_updates)

    config = resolve_data_paths(config)

    oof_path = Path(output_dir) / "oof_predictions.csv"
    submission_path = Path(output_dir) / "submission_mid_fusion_lite.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")

    requested_numeric_cols = set(config.numeric_cols)
    if bool(config.use_feature_engineering):
        requested_numeric_cols.update(BASE_NUMERIC_COLS)

    requested_train_cols = {
        config.id_col,
        config.target_col,
        config.group_col,
        *config.categorical_cols,
        *requested_numeric_cols,
    }
    requested_test_cols = {
        config.id_col,
        *config.categorical_cols,
        *requested_numeric_cols,
    }
    if bool(config.use_feature_engineering):
        requested_test_cols.add(config.group_col)

    train_df = _read_csv_with_fallback(
        path=config.train_metadata_path,
        requested_cols=requested_train_cols,
    )
    test_df = _read_csv_with_fallback(
        path=config.test_metadata_path,
        requested_cols=requested_test_cols,
    )

    if (
        config.id_col not in train_df.columns
        or config.target_col not in train_df.columns
        or config.group_col not in train_df.columns
    ):
        raise KeyError("Train metadata must contain id, target, and group columns.")
    if config.id_col not in test_df.columns:
        raise KeyError("Test metadata must contain id column.")

    if bool(config.use_feature_engineering):
        train_df, test_df = add_feature_engineering(
            train_df=train_df,
            test_df=test_df,
            group_col=config.group_col,
            categorical_cols=config.categorical_cols,
            use_contextual_group_norm_features=bool(
                config.use_contextual_group_norm_features
            ),
            use_ugly_duckling_features=bool(config.use_ugly_duckling_features),
            use_anatomical_zscore_features=bool(config.use_anatomical_zscore_features),
            use_area_aggregation_features=bool(config.use_area_aggregation_features),
            use_patient_lof_features=bool(config.use_patient_lof_features),
            patient_lof_min_group_size=int(config.patient_lof_min_group_size),
        )

    numeric_cols, categorical_cols = select_available_columns(
        train_df=train_df,
        test_df=test_df,
        numeric_candidates=config.numeric_cols,
        categorical_candidates=config.categorical_cols,
    )

    oof_df = _extract_oof_predictions(oof_path=oof_path, id_col=config.id_col)
    train_df = train_df.merge(oof_df, on=config.id_col, how="left")
    if train_df["fusion_pred"].isna().any():
        raise ValueError("Missing OOF predictions after merge with train metadata.")

    test_pred_df = _extract_test_predictions(
        submission_path=submission_path, id_col=config.id_col
    )
    test_df = test_df.merge(test_pred_df, on=config.id_col, how="left")
    if test_df["fusion_pred"].isna().any():
        raise ValueError("Missing submission predictions after merge with test metadata.")

    numeric_cols = list(dict.fromkeys(list(numeric_cols) + ["fusion_pred"]))

    feature_mode = feature_mode.strip().lower()
    if feature_mode == "oof-only":
        numeric_cols = ["fusion_pred"]
        categorical_cols = []
    elif feature_mode == "numeric-only":
        categorical_cols = []
    elif feature_mode != "full":
        raise ValueError("feature_mode must be one of: oof-only, numeric-only, full")

    used_mask = train_df["used_in_cv"].astype(int) == 1
    train_df = train_df.loc[used_mask].reset_index(drop=True)

    y = train_df[config.target_col].astype(np.int64).to_numpy()
    groups = train_df[config.group_col].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    folds = list(splitter.split(train_df, y, groups))

    if config.fold_indices is None:
        selected_folds = list(range(config.n_splits))
    else:
        selected_folds = list(config.fold_indices)
    selected_folds_set = set(selected_folds)
    print(f"Selected folds: {sorted(selected_folds_set)}")

    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    oof_mask = np.zeros(len(train_df), dtype=bool)
    test_pred_sum = np.zeros(len(test_df), dtype=np.float32)
    n_test_ensembles = 0
    fold_metrics: list[dict[str, Any]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(folds):
        if fold_index not in selected_folds_set:
            continue

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

        preprocessor = FoldTabularPreprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            use_quantile_transform=False,
            quantile_noise_std=float(config.quantile_noise_std),
            quantile_random_state=config.seed + fold_index + 1,
            use_null_importance_filter=False,
            target_col=config.target_col,
        )
        preprocessor.fit(fold_train_df)

        x_train = preprocessor.transform(fold_train_df)
        x_valid = preprocessor.transform(fold_valid_df)
        x_test = preprocessor.transform(test_df)

        y_train = fold_train_df[config.target_col].astype(np.int64).to_numpy()
        y_valid = fold_valid_df[config.target_col].astype(np.int64).to_numpy()

        n_pos = max(float(y_train.sum()), 1.0)
        n_neg = max(float(len(y_train) - y_train.sum()), 1.0)
        pos_weight = n_neg / n_pos
        sample_weight = np.where(y_train == 1, pos_weight, 1.0)

        model = HistGradientBoostingClassifier(
            learning_rate=float(gbdt_params["learning_rate"]),
            max_depth=gbdt_params["max_depth"],
            max_iter=int(gbdt_params["max_iter"]),
            max_leaf_nodes=gbdt_params["max_leaf_nodes"],
            min_samples_leaf=int(gbdt_params["min_samples_leaf"]),
            l2_regularization=float(gbdt_params["l2_regularization"]),
            random_state=config.seed + fold_index,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)

        valid_pred = model.predict_proba(x_valid)[:, 1]
        fold_pauc = partial_auc_score(y_valid, valid_pred)

        oof_pred[valid_idx] = valid_pred
        oof_mask[valid_idx] = True

        test_pred_sum += model.predict_proba(x_test)[:, 1]
        n_test_ensembles += 1

        fold_info = {
            "fold": fold_index,
            "fold_valid_pauc": float(fold_pauc),
            "n_train": int(len(fold_train_df)),
            "n_valid": int(len(fold_valid_df)),
        }
        fold_metrics.append(fold_info)
        print("Fold summary:", fold_info)

    if n_test_ensembles == 0:
        raise RuntimeError("No folds were selected for stacking.")

    test_pred = test_pred_sum / float(n_test_ensembles)

    if not np.any(oof_mask):
        cv_pauc = float("nan")
    else:
        cv_pauc = partial_auc_score(y[oof_mask], oof_pred[oof_mask])

    stack_dir = Path(stack_output_dir)
    stack_dir.mkdir(parents=True, exist_ok=True)

    oof_stack_df = train_df[[config.id_col, config.target_col, "fusion_pred"]].copy()
    oof_stack_df["gbdt_pred"] = oof_pred
    oof_stack_df["used_in_cv"] = oof_mask.astype(int)

    submission_df = pd.DataFrame(
        {config.id_col: test_df[config.id_col], "target": test_pred}
    )

    metrics_payload = {
        "preset": config.preset_name,
        "feature_mode": feature_mode,
        "cv_pauc": float(cv_pauc),
        "n_test_ensembles": int(n_test_ensembles),
        "fold_metrics": fold_metrics,
        "gbdt_params": gbdt_params,
    }

    oof_path_out = stack_dir / "oof_predictions_stack_gbdt.csv"
    submission_path_out = stack_dir / "submission_stack_gbdt.csv"
    metrics_path_out = stack_dir / "metrics_stack_gbdt.json"

    oof_stack_df.to_csv(oof_path_out, index=False)
    submission_df.to_csv(submission_path_out, index=False)
    metrics_path_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"Stacked CV pAUC: {cv_pauc:.6f}")
    print(f"Saved OOF: {oof_path_out}")
    print(f"Saved submission: {submission_path_out}")
    print(f"Saved metrics: {metrics_path_out}")

    return {
        "cv_pauc": float(cv_pauc),
        "fold_metrics": fold_metrics,
        "oof_path": str(oof_path_out),
        "submission_path": str(submission_path_out),
        "metrics_path": str(metrics_path_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GBDT stacker on Mid Fusion OOF")
    parser.add_argument("--preset", default="quick", choices=["smoke", "quick", "full_lite"])
    parser.add_argument(
        "--mid-fusion-output-dir",
        default="./mid_fusion_lite_outputs/quick",
        help="Path to Mid Fusion Lite outputs containing OOF/submission files",
    )
    parser.add_argument(
        "--stack-output-dir",
        default="./mid_fusion_lite_outputs/stack_gbdt",
        help="Output directory for stacked predictions",
    )
    parser.add_argument(
        "--feature-mode",
        default="numeric-only",
        choices=["oof-only", "numeric-only", "full"],
        help="Feature set for stacker (default: numeric-only)",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    args = parser.parse_args()

    gbdt_params = {
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "l2_regularization": args.l2_regularization,
    }

    run_gbdt_stack(
        preset_name=args.preset,
        output_dir=args.mid_fusion_output_dir,
        stack_output_dir=args.stack_output_dir,
        feature_mode=args.feature_mode,
        gbdt_params=gbdt_params,
    )


if __name__ == "__main__":
    main()
