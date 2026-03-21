import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so that `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.training_pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection model.")
    parser.add_argument("--model", required=True, choices=["lr", "rf", "gb"])
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--dataset-version", default="ieee-cis-original")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    result = run_training_pipeline(
        model_name=args.model,
        config_path=config_path,
        dataset_version=args.dataset_version,
    )
    print("Training finished.")
    print("Metrics:", result["metrics"])
    print("Model saved to:", result["model_path"])
    print("Run metadata:", result["run_path"])


if __name__ == "__main__":
    main()

