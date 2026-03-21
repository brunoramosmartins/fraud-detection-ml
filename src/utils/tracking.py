import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import RUNS_DIR

logger = logging.getLogger(__name__)


def start_run(model_name: str, config_path: str) -> Dict[str, Any]:
    """Create a new run record for experiment tracking.

    Returns a mutable dict that accumulates metrics and artifact paths
    throughout the training pipeline.  Call :func:`end_run` to persist
    the record to disk.

    Parameters
    ----------
    model_name : str
        Identifier of the model being trained (e.g., ``"gb"``).
    config_path : str
        Path to the YAML configuration file used for this run.

    Returns
    -------
    Dict[str, Any]
        Run record with ``run_id``, timestamps, and empty containers
        for metrics and artifacts.
    """
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run = {
        "run_id": run_id,
        "model_name": model_name,
        "config_file": config_path,
        "start_time": datetime.utcnow().isoformat(),
        "metrics": {},
        "artifact_paths": [],
    }
    return run


def log_metrics(run: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Merge *metrics* into the run record (in place)."""
    run["metrics"].update(metrics)


def log_artifacts(run: Dict[str, Any], artifact_paths: List[str]) -> None:
    """Append *artifact_paths* to the run record (in place)."""
    run.setdefault("artifact_paths", []).extend(artifact_paths)


def end_run(run: Dict[str, Any]) -> Path:
    """Finalize the run and persist it as JSON to ``artifacts/runs/``."""
    run["end_time"] = datetime.utcnow().isoformat()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"run_{run['run_id']}.json"
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to persist run record to {path}: {exc}") from exc
    logger.info("Run %s saved to %s", run["run_id"], path)
    return path

