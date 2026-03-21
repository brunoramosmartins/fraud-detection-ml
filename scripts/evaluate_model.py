"""End-to-end model evaluation on the validation split.

Loads the latest trained model, rebuilds the temporal train/val split,
and prints all cost-sensitive and classification metrics.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --sample 10000
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.data.loader import load_full_training_dataset
from src.data.split import temporal_train_val_split
from src.features.pipeline import build_features
from src.models.metrics import (
    approve_all_baseline_loss,
    expected_loss,
    precision_fpr_at_threshold,
    threshold_sweep,
)
from src.utils.config import C_FP, DATA_PATH, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest trained model.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use a random sample of N rows from the full dataset (faster evaluation).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold (default: use model metadata).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Find the latest model artifact ──────────────────────────────────
    model_files = sorted(MODELS_DIR.glob("gb_v1_*.pkl"))
    model_files = [f for f in model_files if "_meta" not in f.name]
    if not model_files:
        print("ERROR: No model artifact found in artifacts/models/")
        sys.exit(1)

    model_path = model_files[-1]
    meta_path = model_path.with_name(model_path.stem + "_meta.json")

    print(f"Model: {model_path.name}")
    model = joblib.load(model_path)

    # Load metadata
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    feature_list = meta.get("feature_list")
    metrics_meta = meta.get("metrics", {})
    saved_threshold = float(metrics_meta.get("best_threshold", 0.5))

    threshold = args.threshold if args.threshold is not None else saved_threshold
    print(f"Threshold: {threshold:.4f}")
    print()

    # ── 2. Load data and split ─────────────────────────────────────────────
    print("Loading dataset...")
    df = load_full_training_dataset(DATA_PATH)

    if args.sample and args.sample < len(df):
        print(f"Sampling {args.sample:,} rows from {len(df):,} total")
        df = df.sample(n=args.sample, random_state=42)

    X_train_raw, X_val_raw, y_train, y_val, val_amount = temporal_train_val_split(
        df,
        split_quantile=0.8,
        target_col="isFraud",
        time_col="TransactionDT",
        amount_col="TransactionAmt",
    )

    print(f"Train: {len(y_train):,} rows ({y_train.sum():,} fraud)")
    print(f"Val:   {len(y_val):,} rows ({y_val.sum():,} fraud)")
    print()

    # ── 3. Build features and predict ──────────────────────────────────────
    X_val, used_features = build_features(X_val_raw)
    if feature_list:
        X_val = X_val[feature_list].fillna(0.0)

    proba = model.predict_proba(X_val)[:, 1]

    # ── 4. Cost-sensitive metrics ──────────────────────────────────────────
    print("=" * 60)
    print("COST-SENSITIVE METRICS (EML Framework)")
    print("=" * 60)

    baseline_loss = approve_all_baseline_loss(y_val, val_amount)
    model_loss = expected_loss(y_val, proba, val_amount, C_FP, threshold)
    reduction = baseline_loss - model_loss

    print(f"Baseline loss (approve-all): ${baseline_loss:,.2f}")
    print(f"Model loss (t={threshold:.4f}):   ${model_loss:,.2f}")
    pct = reduction / baseline_loss * 100
    print(f"Loss reduction:              ${reduction:,.2f} ({pct:.1f}%)")
    print()

    # ── 5. Threshold sweep ─────────────────────────────────────────────────
    best_t, best_loss, thresholds_arr, losses = threshold_sweep(
        y_val, proba, val_amount, C_FP
    )
    print(f"Optimal threshold (sweep):   {best_t:.4f}")
    print(f"Best possible loss:          ${best_loss:,.2f}")
    max_reduction = baseline_loss - best_loss
    max_pct = max_reduction / baseline_loss * 100
    print(f"Max loss reduction:          ${max_reduction:,.2f} ({max_pct:.1f}%)")
    print()

    # ── 6. Classification metrics ──────────────────────────────────────────
    print("=" * 60)
    print(f"CLASSIFICATION METRICS @ threshold={threshold:.4f}")
    print("=" * 60)

    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    precision, fpr = precision_fpr_at_threshold(y_val, proba, threshold)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
    print(f"Precision:  {precision:.4f} ({tp:,} real fraud in {tp + fp:,} alerts)")
    print(f"Recall:     {recall:.4f} ({tp:,} of {tp + fn:,} fraud cases captured)")
    print(f"FPR:        {fpr:.4f} ({fp:,} of {fp + tn:,} legit flagged)")
    print()

    # ── 7. Ranking metrics ─────────────────────────────────────────────────
    from sklearn.metrics import average_precision_score, roc_auc_score

    roc_auc = roc_auc_score(y_val, proba)
    pr_auc = average_precision_score(y_val, proba)

    print(f"ROC-AUC:    {roc_auc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print()

    # ── 8. Full classification report ──────────────────────────────────────
    print("=" * 60)
    print("SKLEARN CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_val, y_pred, target_names=["Legitimate", "Fraud"]))

    # ── 9. Top feature importances ─────────────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = feature_list or used_features
        top_n = min(15, len(feat_names))
        indices = np.argsort(importances)[::-1][:top_n]

        print("=" * 60)
        print(f"TOP {top_n} FEATURE IMPORTANCES")
        print("=" * 60)
        for rank, idx in enumerate(indices, 1):
            print(f"  {rank:2d}. {feat_names[idx]:<25s} {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
