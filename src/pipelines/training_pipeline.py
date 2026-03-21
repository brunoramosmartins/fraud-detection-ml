from pathlib import Path
from typing import Any, Dict

import yaml

from src.data.loader import load_full_training_dataset
from src.data.schema import validate_schema
from src.data.split import temporal_train_val_split
from src.features.pipeline import build_features
from src.models.artifacts import save_model_artifact
from src.models.factory import get_model
from src.models.metrics import compute_classification_metrics
from src.utils import config as cfg
from src.utils.tracking import end_run, log_artifacts, log_metrics, start_run


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training_pipeline(
    model_name: str,
    config_path: Path,
    dataset_version: str = "ieee-cis-original",
) -> Dict[str, Any]:
    """Execute the full training pipeline: load, validate, split, train, evaluate, persist.

    Parameters
    ----------
    model_name : str
        Model identifier passed to :func:`get_model` (``"lr"``, ``"rf"``, or ``"gb"``).
    config_path : Path
        Path to the YAML config with hyperparameters and feature set.
    dataset_version : str
        Label recorded in run metadata for traceability.

    Returns
    -------
    Dict[str, Any]
        Paths to the saved model, metadata, and run record, plus computed metrics.
    """
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    run = start_run(model_name=model_name, config_path=str(config_path))
    run["dataset_version"] = dataset_version

    df = load_full_training_dataset(cfg.DATA_PATH)
    validate_schema(df)

    X_train_raw, X_val_raw, y_train, y_val, val_amount = temporal_train_val_split(
        df,
        split_quantile=config.get("split_quantile", 0.8),
        target_col="isFraud",
        time_col="TransactionDT",
        amount_col="TransactionAmt",
    )

    X_train, feature_list = build_features(X_train_raw, feature_set=config.get("feature_set", "v1"))
    X_val, _ = build_features(X_val_raw, feature_set=config.get("feature_set", "v1"))

    model = get_model(model_name, config=config.get("models", {}))
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_val)[:, 1]

    c_fp = config.get("c_fp", cfg.C_FP)
    metrics = compute_classification_metrics(
        y_true=y_val,
        proba=val_proba,
        amount=val_amount,
        c_fp=c_fp,
    )
    log_metrics(run, metrics)

    metadata = {
        "model_name": model_name,
        "feature_set": config.get("feature_set", "v1"),
        "feature_list": feature_list,
        "dataset_version": dataset_version,
        "config_file": str(config_path),
        "metrics": metrics,
    }
    model_path, meta_path = save_model_artifact(
        model=model,
        metadata=metadata,
        model_name=model_name,
        version=config.get("version", "v1"),
    )
    log_artifacts(run, [str(model_path), str(meta_path)])

    run_path = end_run(run)

    result = {
        "metrics": metrics,
        "model_path": str(model_path),
        "metadata_path": str(meta_path),
        "run_path": str(run_path),
    }
    return result

