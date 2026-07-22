import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so that `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.training_pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate automated model retraining.")
    parser.add_argument("--config", required=True, help="YAML config for training pipeline.")
    parser.add_argument("--drift-report", required=True, help="Path to drift report JSON.")
    parser.add_argument(
        "--psi-threshold", type=float, default=0.2, help="PSI threshold to trigger retraining."
    )
    parser.add_argument(
        "--model-name",
        default="lgbm",
        choices=["lr", "rf", "gb", "lgbm"],
        help="Model to retrain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    drift_report = json.loads(Path(args.drift_report).read_text(encoding="utf-8"))
    max_psi = max(drift_report.values()) if drift_report else 0.0

    if max_psi <= args.psi_threshold:
        print(f"Max PSI {max_psi:.3f} below threshold {args.psi_threshold:.3f}; no retraining.")
        return

    print(f"Max PSI {max_psi:.3f} above threshold {args.psi_threshold:.3f}; triggering retraining.")

    config_path = Path(args.config)
    result = run_training_pipeline(
        model_name=args.model_name,
        config_path=config_path,
        dataset_version="ieee-cis-retrained",
    )
    print("Retraining finished.")
    print("New model path:", result["model_path"])
    print("Run metadata:", result["run_path"])


if __name__ == "__main__":
    main()


