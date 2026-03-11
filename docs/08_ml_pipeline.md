## 1. Overview

This document describes the Phase 4 ML engineering pipeline for the fraud detection project.
The goal is to move from notebook-centric experimentation to a modular, reproducible ML system.

The pipeline mirrors production-style workflows: data ingestion, schema validation, feature building,
model training, cost-sensitive evaluation, artifact versioning, experiment tracking, and automated tests.

## 2. Directory Structure

- `src/`
  - `data/`: data ingestion and splitting (`loader.py`, `schema.py`, `split.py`)
  - `features/`: feature registry and feature pipeline (`feature_registry.py`, `pipeline.py`)
  - `models/`: metrics, model factory, artifact serialization (`metrics.py`, `factory.py`, `artifacts.py`)
  - `pipelines/`: orchestration of the training pipeline (`training_pipeline.py`)
  - `utils/`: configuration and experiment tracking (`config.py`, `tracking.py`)
- `configs/`: YAML configuration files for models and pipeline (e.g. `model_gb_v1.yml`)
- `scripts/`: CLI entrypoints (e.g. `train_model.py`)
- `artifacts/`: serialized models and run metadata
- `tests/`: unit tests for metrics and, later, data loading and features

## 3. Data Flow

1. **Train script** (`scripts/train_model.py`) parses CLI arguments and calls the training pipeline.
2. **Training pipeline** (`src/pipelines/training_pipeline.py`) loads a YAML config and orchestrates the run:
   - Load and merge raw data via `src/data/loader.py`.
   - Validate basic schema via `src/data/schema.py`.
   - Apply a temporal train/validation split via `src/data/split.py`.
   - Build feature matrices using `src/features/pipeline.py` and the feature registry.
   - Instantiate a model via `src/models/factory.py` and fit on training data.
   - Compute cost-sensitive metrics (Expected Monetary Loss, ROC-AUC, PR-AUC, etc.) via `src/models/metrics.py`.
   - Serialize the trained model and metadata via `src/models/artifacts.py`.
   - Log run metadata and metrics via `src/utils/tracking.py`.

## 4. Feature Registry and Pipeline

- `feature_registry.py` maintains a simple registry of feature sets (e.g. `FEATURE_SETS['v1']`).
  - For the initial version, the numeric feature list is inferred from a reference dataframe,
    excluding `TransactionID`, `TransactionDT`, and `isFraud`, mirroring Phase 3.
  - The inferred list is cached in the registry for reuse.
- `pipeline.py` exposes:
  - `build_features(df, feature_set='v1')`:
    - selects the ordered feature list from the registry,
    - applies simple imputations (`fillna(0.0)`),
    - returns the feature matrix `X` and the feature list.

Scaling is handled inside the model pipeline (for logistic regression) using scikit-learn pipelines.

## 5. Data Ingestion and Schema Validation

- `loader.py`:
  - `load_full_training_dataset(base_path)` loads `train_transaction.csv` and `train_identity.csv`
    and merges them on `TransactionID` with a left join.
- `schema.py`:
  - Validates presence of key columns (`TransactionID`, `TransactionDT`, `TransactionAmt`, `isFraud`).
  - Enforces no missing values in these core columns.
  - Performs simple dtype and target distribution sanity checks.
- `split.py`:
  - `temporal_train_val_split` performs an 80/20-style temporal split on `TransactionDT`,
    returning raw train/validation frames and the validation `TransactionAmt` series for cost computation.

## 6. Model Training and Metrics

- `factory.py` implements `get_model(model_name, config)`:
  - `"lr"`: logistic regression with a `StandardScaler` in a scikit-learn `Pipeline`.
  - `"rf"`: RandomForestClassifier with conservative depth and leaf settings.
  - `"gb"`: GradientBoostingClassifier with the hyperparameters aligned to the Phase 3 notebook.
- `metrics.py` contains:
  - Expected Monetary Loss computation (`expected_loss`, `approve_all_baseline_loss`).
  - Threshold sweep and optimal threshold selection (`threshold_sweep`).
  - ROC-AUC, PR-AUC, FDR, FPR computation and a `compute_classification_metrics` helper
    that returns all metrics in a single dictionary.

## 7. Training Pipeline and Experiment Tracking

- `training_pipeline.py` exposes `run_training_pipeline(model_name, config_path, dataset_version)`:
  - Loads config YAML (e.g. `configs/model_gb_v1.yml`).
  - Orchestrates data loading, validation, split, feature building, model training, and evaluation.
  - Logs metrics and artifact paths through `src/utils/tracking.py`.
- `tracking.py`:
  - Provides a lightweight experiment tracking mechanism writing JSON files to `artifacts/runs/`.
  - Each run stores:
    - `run_id`, `model_name`, `config_file`, `dataset_version`,
      timestamps, `metrics`, and `artifact_paths`.

## 8. Artifact Management

- `artifacts.py`:
  - Persists trained models (via `joblib`) and JSON metadata under `artifacts/models/`.
  - Metadata includes:
    - `model_name`, `feature_set`, `feature_list`, `dataset_version`,
      `config_file`, and the evaluation `metrics`.
  - This enables reproducibility of the trained artifacts and links them to their configuration.

## 9. CLI Training Script

- `scripts/train_model.py`:
  - CLI interface:
    - `--model {lr, rf, gb}`
    - `--config path/to/config.yml`
    - `--dataset-version` (optional)
  - Calls `run_training_pipeline` and prints final metrics and artifact paths.

Example:

```bash
python scripts/train_model.py --model gb --config configs/model_gb_v1.yml
```

## 10. Tests

- `tests/test_metrics.py`:
  - Sanity tests for Expected Monetary Loss, baseline loss, and threshold sweep behavior.
- Future tests (to be expanded as the project evolves):
  - `tests/test_data_loading.py`: data ingestion and schema validation.
  - `tests/test_features.py`: feature registry and pipeline behavior.
  - `tests/test_training_pipeline.py`: end-to-end smoke test on a small synthetic dataset.

## 11. Reproducibility and Extensions

- Core reproducibility is achieved through:
  - Centralized configuration in YAML files.
  - Fixed random state in models and config.
  - Versioned artifacts and run metadata under `artifacts/`.
- Future extensions:
  - More explicit dataset versioning via a dataset catalog.
  - Integration with full-fledged experiment tracking tools (e.g. MLflow).
  - Additional pipelines for calibration, monitoring, and deployment simulation.

