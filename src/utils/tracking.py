import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.utils.config import RUNS_DIR


def start_run(model_name: str, config_path: str) -> Dict[str, Any]:
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
    run["metrics"].update(metrics)


def log_artifacts(run: Dict[str, Any], artifact_paths: List[str]) -> None:
    run.setdefault("artifact_paths", []).extend(artifact_paths)


def end_run(run: Dict[str, Any]) -> Path:
    run["end_time"] = datetime.utcnow().isoformat()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"run_{run['run_id']}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return path

