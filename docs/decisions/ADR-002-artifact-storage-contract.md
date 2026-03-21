# ADR-002 — Artifact Storage Contract and Metadata Schema

## Status

Accepted

## Context

The trained model must be persisted in a way that the scoring API can load
it reliably and that future training runs do not silently overwrite the
deployed model. Additionally, the API needs to know which feature columns
to expect at inference time, in which order, and what decision threshold
was selected — information that is only known after training.

Options considered:
1. Save only the `.pkl` file; hardcode feature list and threshold in the API.
2. Save `.pkl` + companion `_meta.json` with feature list, threshold, and
   metrics; use a file naming convention that includes timestamp.
3. Use an external model registry (MLflow, Weights & Biases).

## Decision

Option 2: `.pkl` + `_meta.json` companion file with a timestamp naming
convention.

**Naming:** `{model}_{version}_{YYYYMMDD_HHMMSS}.pkl` and
`{model}_{version}_{YYYYMMDD_HHMMSS}_meta.json`.

**Metadata schema:**
```json
{
  "model_name": "gb",
  "feature_set": "v1",
  "feature_list": ["TransactionAmt", "card1", ...],
  "dataset_version": "ieee-cis-original",
  "config_file": "configs/model_gb_v1.yml",
  "metrics": {
    "roc_auc": 0.861,
    "best_threshold": 0.02,
    ...
  }
}
```

The API loads the lexicographically latest `.pkl` matching `gb_v1_*.pkl`
at startup.

## Consequences

**Positive:**
- The `feature_list` in `_meta.json` is the training–inference contract.
  The API reads it and enforces it exactly, preventing silent schema drift.
- Multiple versions coexist without collision. Rolling back means pointing
  to an earlier artifact — no retraining required.
- Run metadata in `artifacts/runs/` links run ID, metrics, artifact paths,
  and timestamps, providing lightweight experiment tracking without an
  external service.
- The `_meta.json` is human-readable and can be inspected to understand any
  deployed model without running code.

**Negative:**
- "Latest file alphabetically" as the deployment mechanism is brittle.
  A file with a future timestamp (e.g., from a system clock error) would
  be selected incorrectly. A production system would use an explicit
  "deployed" flag or a model registry with promotion workflows.
- No access control or immutability guarantee. Any process with filesystem
  access can overwrite artifacts. A production artifact store would be
  append-only with checksums.
- The binary `.pkl` format is not portable across Python versions or
  scikit-learn major versions. A production deployment would test artifact
  compatibility before promotion.
