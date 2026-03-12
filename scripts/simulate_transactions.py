import argparse
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

# Ensure project root is on sys.path so that `src` is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_full_training_dataset
from src.utils.config import DATA_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate transaction scoring requests.")
    parser.add_argument("--api-url", default="http://localhost:8000/predict")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sanitize_record(rec: dict) -> dict:
    """Replace NaN / ±inf float values with None so the payload is valid JSON."""
    return {
        k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
        for k, v in rec.items()
    }


def send_batch(api_url: str, batch_df: pd.DataFrame) -> List[dict]:
    records = [_sanitize_record(r) for r in batch_df.to_dict(orient="records")]
    response = requests.post(api_url, json={"transactions": records}, timeout=30)
    response.raise_for_status()
    return response.json()["predictions"]


def main() -> None:
    args = parse_args()

    df = load_full_training_dataset(DATA_PATH)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path("artifacts") / "monitoring" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{timestamp}.csv"

    logs: List[dict] = []
    start_idx = 0

    for batch_idx in range(args.max_batches):
        if start_idx >= len(df):
            break

        end_idx = start_idx + args.batch_size
        batch = df.iloc[start_idx:end_idx].copy()

        try:
            predictions = send_batch(args.api_url, batch)
        except requests.RequestException as exc:
            logger.error("Batch %d failed: %s — skipping.", batch_idx + 1, exc)
            start_idx = end_idx
            continue

        now = datetime.now(timezone.utc).isoformat()
        for row, pred in zip(batch.itertuples(index=False), predictions):
            logs.append(
                {
                    "timestamp": now,
                    "TransactionID": getattr(row, "TransactionID", None),
                    "fraud_probability": pred["fraud_probability"],
                    "fraud_flag": pred["fraud_flag"],
                    "isFraud": getattr(row, "isFraud", None),
                    "TransactionAmt": getattr(row, "TransactionAmt", None),
                }
            )

        logger.info("Batch %d: scored %d transactions.", batch_idx + 1, len(predictions))
        start_idx = end_idx
        time.sleep(args.sleep_seconds)

    if logs:
        pd.DataFrame(logs).to_csv(output_path, index=False)
        logger.info("Saved %d prediction records to %s", len(logs), output_path)
    else:
        logger.warning("No logs generated — dataset may be empty or all batches failed.")


if __name__ == "__main__":
    main()
