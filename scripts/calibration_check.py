"""Calibration check for the served model (Phase 9, Issue #26).

Loads the deployed artifact, scores the temporal validation split with the
exact training-time configuration, and reports:

1. sanity metrics (must reproduce the training run);
2. calibration quality — Brier score, ECE, reliability table;
3. a FINE threshold sweep for the EML operating point (the training sweep's
   default grid starts at 0.01, which can clip the optimum for a
   well-separated model);
4. a leak-free isotonic-calibration test: fit on the earlier half of the
   validation window, evaluate raw vs calibrated on the later half.

The decision (adopt isotonic or not) is reported, not applied: the served
artifact is only replaced by an explicit retrain/repackage step.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.data.loader import load_full_training_dataset
from src.data.split import temporal_train_val_split
from src.models.metrics import (
    expected_calibration_error,
    reliability_table,
    threshold_sweep,
)
from src.utils.config import DATA_PATH, MODELS_DIR

FINE_GRID = np.concatenate(
    [np.arange(0.001, 0.010, 0.0005), np.arange(0.010, 0.1005, 0.0025)]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration check for a served model.")
    parser.add_argument("--model-glob", default="lgbm_v2_*.pkl")
    parser.add_argument("--config", default="configs/model_lgbm_v2.yml")
    parser.add_argument("--n-bins", type=int, default=10)
    return parser.parse_args()


def print_reliability(tbl: dict) -> None:
    print(f"{'bin':>22} {'count':>8} {'mean_pred':>10} {'observed':>10}")
    for lo, hi, n, mp, obs in zip(
        tbl["bin_lower"], tbl["bin_upper"], tbl["count"],
        tbl["mean_predicted"], tbl["observed_rate"],
    ):
        print(f"[{lo:9.5f}, {hi:9.5f}] {n:>8d} {mp:>10.5f} {obs:>10.5f}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    c_fp = float(config.get("c_fp", 5.0))

    model_files = sorted(MODELS_DIR.glob(args.model_glob))
    if not model_files:
        raise FileNotFoundError(f"No artifact matching {args.model_glob} in {MODELS_DIR}")
    model_path = model_files[-1]
    meta = json.loads(
        model_path.with_name(model_path.stem + "_meta.json").read_text(encoding="utf-8")
    )
    print(f"Artifact : {model_path.name}")

    df = load_full_training_dataset(DATA_PATH)
    _, X_val_raw, _, y_val, val_amount = temporal_train_val_split(
        df,
        split_quantile=config.get("split_quantile", 0.8),
        target_col="isFraud",
        time_col="TransactionDT",
        amount_col="TransactionAmt",
    )
    y = y_val.to_numpy()
    amount = val_amount.to_numpy()

    artifact = joblib.load(model_path)
    proba = artifact.predict_proba(X_val_raw[meta["feature_list"]])[:, 1]

    # 1 — sanity: must reproduce the training-run holdout AUC
    auc = roc_auc_score(y, proba)
    print(f"\n[1] Sanity — holdout ROC-AUC: {auc:.6f} "
          f"(training run: {meta['metrics']['roc_auc']:.6f})")

    # 2 — calibration quality
    brier = brier_score_loss(y, proba)
    ece = expected_calibration_error(y, proba, n_bins=args.n_bins)
    print(f"\n[2] Calibration — Brier: {brier:.6f} | ECE ({args.n_bins} quantile bins): {ece:.6f}")
    print_reliability(reliability_table(y, proba, n_bins=args.n_bins))

    # 3 — fine EML threshold sweep (training grid starts at 0.01)
    best_t, best_loss, grid, losses = threshold_sweep(y, proba, amount, c_fp, FINE_GRID)
    coarse_loss = losses[np.isclose(grid, 0.01)][0]
    print(f"\n[3] Fine sweep — best threshold: {best_t:.4f} | EML: {best_loss:,.0f}")
    print(f"    at 0.01 (training grid edge): EML {coarse_loss:,.0f} "
          f"({coarse_loss - best_loss:+,.0f} vs fine optimum)")
    order = np.argsort(losses)[:5]
    for i in order:
        print(f"    t={grid[i]:.4f}  EML={losses[i]:,.0f}")

    # 4 — isotonic test: fit on earlier half, evaluate on later half
    dt = X_val_raw["TransactionDT"].to_numpy()
    cut = np.quantile(dt, 0.5)
    a, b = dt < cut, dt >= cut
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(proba[a], y[a])
    proba_b_cal = iso.predict(proba[b])

    rows = {}
    for name, p_b in (("raw", proba[b]), ("isotonic", proba_b_cal)):
        t_b, loss_b, _, _ = threshold_sweep(y[b], p_b, amount[b], c_fp, FINE_GRID)
        rows[name] = {
            "brier": brier_score_loss(y[b], p_b),
            "ece": expected_calibration_error(y[b], p_b, n_bins=args.n_bins),
            "best_threshold": t_b,
            "best_eml": loss_b,
        }
    print("\n[4] Isotonic (fit on earlier val half, evaluated on later half)")
    print(f"{'':>10} {'Brier':>10} {'ECE':>10} {'best_t':>8} {'best EML':>12}")
    for name, r in rows.items():
        print(f"{name:>10} {r['brier']:>10.6f} {r['ece']:>10.6f} "
              f"{r['best_threshold']:>8.4f} {r['best_eml']:>12,.0f}")
    eml_gain = rows["raw"]["best_eml"] - rows["isotonic"]["best_eml"]
    rel = eml_gain / rows["raw"]["best_eml"]
    print(f"    isotonic EML gain: {eml_gain:+,.0f} ({rel:+.2%})")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact": model_path.name,
        "holdout_auc": auc,
        "brier": brier,
        "ece": ece,
        "fine_sweep": {"best_threshold": best_t, "best_eml": best_loss,
                       "eml_at_0.01": float(coarse_loss)},
        "isotonic_holdout_halves": rows,
    }
    out_dir = PROJECT_ROOT / "artifacts" / "monitoring" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{model_path.stem}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
