import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

from src.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)


def save_model_artifact(
    model: Any,
    metadata: Dict[str, Any],
    model_name: str,
    version: str,
) -> Tuple[Path, Path]:
    """
    Serialize model and metadata under artifacts/models.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_dir = MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{model_name}_{version}_{timestamp}"
    model_path = model_dir / f"{base_name}.pkl"
    meta_path = model_dir / f"{base_name}_meta.json"

    try:
        joblib.dump(model, model_path)
    except OSError as exc:
        raise OSError(f"Failed to write model artifact to {model_path}: {exc}") from exc

    try:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
    except OSError as exc:
        raise OSError(f"Failed to write metadata to {meta_path}: {exc}") from exc

    logger.info("Saved artifact: %s", model_path.name)
    return model_path, meta_path


def load_model_artifact(model_path: Path) -> Any:
    """Load a serialized model artifact."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    except OSError as exc:
        raise OSError(f"Failed to load model from {model_path}: {exc}") from exc

