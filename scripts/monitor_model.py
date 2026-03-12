import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path so that `src` is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.metrics import approve_all_baseline_loss, expected_loss
from src.utils.drift import compute_psi

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor model drift and performance.")
    parser.add_argument("--reference-path", required=True, help="CSV with reference data.")
    parser.add_argument(
        "--predictions-path",
        required=True,
        help="Predictions CSV produced by simulate_transactions.py.",
    )
    parser.add_argument(
        "--psi-threshold", type=float, default=0.2, help="PSI threshold for drift warning."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ref_path = Path(args.reference_path)
    pred_path = Path(args.predictions_path)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    ref_df = pd.read_csv(ref_path)
    pred_df = pd.read_csv(pred_path)

    # Drift report: compute PSI for each numeric feature present in both datasets.
    drift_report: dict = {}
    for col in ["TransactionAmt"]:
        if col in ref_df.columns and col in pred_df.columns:
            psi_val = compute_psi(ref_df[col].values, pred_df[col].values)
            drift_report[col] = psi_val
            level = "WARNING" if psi_val > args.psi_threshold else "OK"
            logger.info("PSI[%s]=%.4f  [%s]", col, psi_val, level)

    # Performance report: requires ground-truth labels in the predictions log.
    performance_report: dict = {}
    if "isFraud" in pred_df.columns and "fraud_probability" in pred_df.columns:
        y_true = pred_df["isFraud"].fillna(0).astype(int).values
        proba = pred_df["fraud_probability"].values
        amount = (
            pred_df["TransactionAmt"].values
            if "TransactionAmt" in pred_df.columns
            else np.zeros(len(pred_df))
        )
        baseline = approve_all_baseline_loss(y_true, amount)
        loss_at_05 = expected_loss(y_true, proba, amount, c_fp=5.0, threshold=0.5)
        performance_report = {
            "fraud_rate": float(y_true.mean()),
            "baseline_loss": baseline,
            "loss_at_05": loss_at_05,
            "expected_loss_reduction_at_05": baseline - loss_at_05,
        }
        logger.info("Performance: fraud_rate=%.4f  loss_reduction=%.2f", y_true.mean(), baseline - loss_at_05)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    drift_dir = Path("artifacts") / "monitoring" / "drift"
    perf_dir = Path("artifacts") / "monitoring" / "performance"
    drift_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    drift_path = drift_dir / f"drift_report_{ts}.json"
    perf_path = perf_dir / f"perf_report_{ts}.json"

    drift_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")
    perf_path.write_text(json.dumps(performance_report, indent=2), encoding="utf-8")

    logger.info("Saved drift report to %s", drift_path)
    logger.info("Saved performance report to %s", perf_path)


if __name__ == "__main__":
    main()
