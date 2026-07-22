"""SHAP analysis for the served v2 model (Phase 9, Issue #27).

Uses LightGBM's built-in TreeSHAP (``pred_contrib=True``) — exact SHAP
values for tree ensembles with no extra dependency (the ``shap`` package
needs numba, unavailable on newer Pythons, and adds nothing for GBDTs).

Outputs, computed on a sample of the temporal holdout:

1. global importance — top features by mean |SHAP|;
2. block-level attribution — mean |SHAP| aggregated by the ADR-006
   feature blocks, quantifying what each serving-feasible block
   contributes to the served model's decisions;
3. a per-prediction explanation for the highest-scored transaction
   (closes limitation #6 in docs/13: "no per-prediction explanations").

SHAP values are in log-odds space (the LightGBM raw score); the final
column returned by ``pred_contrib`` is the expected value (base rate).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import yaml

from src.data.loader import load_full_training_dataset
from src.data.split import temporal_train_val_split
from src.utils.config import DATA_PATH, MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TreeSHAP analysis of a served model.")
    parser.add_argument("--model-glob", default="lgbm_v2_*.pkl")
    parser.add_argument("--config", default="configs/model_lgbm_v2.yml")
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def feature_block(name: str) -> str:
    """Map an engineered feature name to its ADR-006 block."""
    if name in ("tx_hour", "tx_dow", "amt_log1p", "amt_cents"):
        return "time/amount (EXP-003)"
    if name.endswith("_norm"):
        return "D-normalization (EXP-003)"
    if name.endswith("_le"):
        return "categorical label enc (EXP-002)"
    if name.endswith("_freq"):
        return "categorical freq enc (EXP-002)"
    return "numeric base (v1)"


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model_files = sorted(MODELS_DIR.glob(args.model_glob))
    if not model_files:
        raise FileNotFoundError(f"No artifact matching {args.model_glob} in {MODELS_DIR}")
    model_path = model_files[-1]
    artifact = joblib.load(model_path)
    builder, clf = artifact.steps[0][1], artifact.steps[-1][1]
    print(f"Artifact : {model_path.name}")

    df = load_full_training_dataset(DATA_PATH)
    _, X_val_raw, _, y_val, _ = temporal_train_val_split(
        df,
        split_quantile=config.get("split_quantile", 0.8),
        target_col="isFraud",
        time_col="TransactionDT",
        amount_col="TransactionAmt",
    )
    X_val = builder.transform(X_val_raw)
    y = y_val.to_numpy()

    rng = np.random.default_rng(42)
    n = min(args.sample_size, len(X_val))
    idx = rng.choice(len(X_val), size=n, replace=False)
    X_s, y_s = X_val.iloc[idx], y[idx]
    print(f"Holdout  : {len(X_val):,} rows | SHAP sample: {n:,}")

    contrib = clf.booster_.predict(X_s, pred_contrib=True)
    shap_vals, base = contrib[:, :-1], contrib[0, -1]
    features = list(X_s.columns)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    # 1 — global importance
    order = np.argsort(mean_abs)[::-1]
    print(f"\n[1] Top {args.top_n} features by mean |SHAP| (log-odds)")
    for i in order[: args.top_n]:
        print(f"    {features[i]:<28} {mean_abs[i]:.4f}  [{feature_block(features[i])}]")

    # 2 — block-level attribution (the ADR-006 boundary, quantified)
    blocks: dict = {}
    for f, v in zip(features, mean_abs):
        b = feature_block(f)
        blocks.setdefault(b, {"total": 0.0, "n_features": 0})
        blocks[b]["total"] += float(v)
        blocks[b]["n_features"] += 1
    total = sum(b["total"] for b in blocks.values())
    print("\n[2] Attribution by ADR-006 feature block")
    print(f"{'block':<32} {'share':>7} {'n_feat':>7} {'mean|SHAP|/feat':>16}")
    for name, b in sorted(blocks.items(), key=lambda kv: -kv[1]["total"]):
        share = b["total"] / total
        per_feat = b["total"] / b["n_features"]
        print(f"{name:<32} {share:>6.1%} {b['n_features']:>7d} {per_feat:>16.4f}")
        b["share"] = share

    # 3 — per-prediction explanation: highest-scored transaction in sample
    proba_s = clf.predict_proba(X_s)[:, 1]
    top = int(np.argmax(proba_s))
    row_contrib = shap_vals[top]
    row_order = np.argsort(np.abs(row_contrib))[::-1][:8]
    print(f"\n[3] Highest-scored sampled transaction "
          f"(proba={proba_s[top]:.4f}, label={int(y_s[top])}, base log-odds={base:+.3f})")
    for i in row_order:
        print(f"    {features[i]:<28} {row_contrib[i]:+.3f}  (value={X_s.iloc[top, i]})")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": model_path.name,
        "sample_size": int(n),
        "base_log_odds": float(base),
        "top_features": [
            {"feature": features[i], "mean_abs_shap": float(mean_abs[i]),
             "block": feature_block(features[i])}
            for i in order[: args.top_n]
        ],
        "block_attribution": blocks,
        "example": {
            "proba": float(proba_s[top]),
            "label": int(y_s[top]),
            "contributions": [
                {"feature": features[i], "shap": float(row_contrib[i])}
                for i in row_order
            ],
        },
    }
    out_dir = PROJECT_ROOT / "artifacts" / "monitoring" / "shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shap_{model_path.stem}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
