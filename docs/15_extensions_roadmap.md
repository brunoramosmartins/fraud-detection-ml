# Extensions Roadmap

This document describes concrete next steps to evolve the system toward
production readiness. Each extension is described with its motivation,
the specific gap it addresses, implementation sketch, and the signal it
would add to a technical portfolio.

Extensions are grouped by theme and ordered within each group by
implementation complexity (simplest first).

---

## 1. Evaluation and Robustness

### 1.1 Temporal Cross-Validation

**Gap:** The model is evaluated on a single validation window (the last 20%
of data by time). A single window may not be representative if there are
seasonal fraud patterns or campaign-specific spikes.

**Extension:** Implement multiple non-overlapping temporal folds:
```
Fold 1: train on [t0, t60%), validate on [t60%, t70%)
Fold 2: train on [t0, t70%), validate on [t70%, t80%)
Fold 3: train on [t0, t80%), validate on [t80%, t90%)
```
Report mean ± std of ROC-AUC and EML across folds.

**Signal:** demonstrates awareness that a single train/test split produces
a high-variance metric estimate and how to mitigate it in time-series settings.

---

### 1.2 Confidence Intervals via Bootstrap

**Gap:** All reported metrics (ROC-AUC 0.861, EML reduction 58.7%) are
point estimates. Interviewers may ask: "how confident are you in these numbers?"

**Extension:** Bootstrap the validation set (1000 resamples with replacement),
compute ROC-AUC and EML for each resample, report 95th percentile intervals.

**Signal:** distinguishes a practitioner who understands statistical uncertainty
from one who reports numbers without qualification.

---

### 1.3 Calibration Analysis

**Gap:** The model's predicted probabilities are used directly as fraud
scores. Poorly calibrated probabilities make EML optimization unreliable —
a model that outputs 0.8 when true fraud probability is 0.3 will select
a wrong threshold.

**Extension:** Compute a reliability diagram (calibration curve) and
Expected Calibration Error (ECE). Apply Platt scaling or isotonic regression
if calibration is poor.

**Signal:** calibration is a prerequisite for valid threshold optimization.
Demonstrating awareness of this is a differentiator.

---

## 2. Modeling

### 2.1 LightGBM / XGBoost Comparison

**Gap:** The model uses `sklearn.ensemble.GradientBoostingClassifier`, which
is slower than purpose-built GBDT implementations. The trade-off analysis
(`docs/12_trade_off_analysis.md`, Trade-off 1) notes this but does not
provide empirical evidence.

**Extension:** Train LightGBM and XGBoost with equivalent hyperparameters,
compare ROC-AUC, PR-AUC, EML, and training time on the same temporal split.
Log results to `artifacts/runs/`.

**Signal:** demonstrates ability to run controlled experiments and make
model selection decisions based on empirical comparison rather than
assumption.

---

### 2.2 SHAP-Based Explainability

**Gap:** The model is a black box. Fraud decisions cannot be explained per
transaction. This is a regulatory gap under LGPD (Brazil) and similar
regulations.

**Extension:** Add `shap.TreeExplainer` for the deployed GradientBoosting
model. Generate:
- Global: mean absolute SHAP values as feature importance (replaces
  `feature_importances_` in notebook Section 4).
- Local: waterfall chart for the highest-risk transaction in the
  validation set.
- API extension: optional `?explain=true` parameter on `/predict` that
  returns top-5 SHAP features per transaction.

**Signal:** explainability is a first-class production requirement in
financial services. Implementing it correctly (TreeExplainer, not
KernelExplainer) and integrating it into the API boundary is a strong signal.

---

### 2.3 Feature Selection via Permutation Importance

**Gap:** The model uses 380 features. Many are likely noise or redundant.
Reducing the feature set can improve inference latency, reduce drift surface
area, and improve model interpretability.

**Extension:** Compute permutation importance on the validation set for the
top 50 features. Retrain with the top 50 only. Compare EML and inference
latency. Update the feature registry to support named feature subsets.

**Signal:** feature engineering is the highest-leverage lever in tabular
ML. Demonstrating systematic feature selection and its trade-offs is a
senior-level signal.

---

## 3. Operations and Monitoring

### 3.1 Full-Feature PSI Monitoring

**Gap:** The current `monitor_model.py` computes PSI for `TransactionAmt`
only. The model was trained on 380 features; a drift in any of them could
degrade performance invisibly.

**Extension:** Compute PSI for all numeric features in `feature_list`.
Aggregate to a composite drift score: `max_psi`, `mean_psi`, and count
of features with PSI > 0.2. Persist a structured report per monitoring run.

**Signal:** demonstrates understanding that monitoring one proxy feature
is insufficient and that monitoring scope must match training scope.

---

### 3.2 Performance Monitoring at Optimal Threshold

**Gap:** The monitoring script computes EML at `threshold=0.5`. The
deployed threshold is 0.02. Monitoring at the wrong threshold measures
a configuration that isn't running in production.

**Extension:** Read `threshold` from `_meta.json` and compute EML,
precision, recall, and FPR at the deployed threshold. Add a time-series
plot of these metrics across monitoring runs.

---

### 3.3 Asynchronous Transaction Queue

**Gap:** The current scoring architecture is synchronous: the caller waits
for the model response before proceeding. This blocks at high throughput.

**Extension:** Replace the direct HTTP call with a message queue pattern:
transactions are published to a queue (Redis Streams or a mock using
`asyncio.Queue`); a consumer goroutine scores them in batches and publishes
results to an output queue. The simulation script publishes to the input
queue instead of calling the API directly.

**Signal:** demonstrates knowledge of the async scoring pattern that
underlies high-throughput fraud systems. Even a local `asyncio.Queue`
mock is sufficient to demonstrate the design.

---

## 4. Deployment and Infrastructure

### 4.1 Docker Compose with API + Monitoring Service

**Gap:** The `Dockerfile` packages the API only. The simulation and
monitoring scripts run in a separate environment.

**Extension:** Define a `docker-compose.yml` with three services:
- `api`: the scoring service on port 8000.
- `simulator`: runs `simulate_transactions.py` once and exits.
- `monitor`: runs `monitor_model.py` against the simulation output.

Add a shared volume for `artifacts/monitoring/`.

**Signal:** demonstrates knowledge of multi-service container orchestration,
which is the entry point for Kubernetes-based deployments.

---

### 4.2 CI Pipeline with GitHub Actions

**Gap:** There is no automated test or lint enforcement. Tests pass
locally but could break silently on a new contributor's changes.

**Extension:** Add `.github/workflows/ci.yml` running on every push to
any branch:
```yaml
- python -m pip install -r requirements-dev.txt
- python -m pytest tests/ -v
- python -m flake8 src/ app/ scripts/ --max-line-length=100
```

**Signal:** CI is a minimum viable practice for any codebase that will
receive contributions. Demonstrating it signals professional engineering
hygiene.

---

### 4.3 Model Registry with Promotion Workflow

**Gap:** The API loads "the most recent `.pkl` file alphabetically." There
is no concept of a promoted model, a staged model, or a rollback path.

**Extension:** Implement a minimal model registry as a JSON index file
at `artifacts/models/registry.json`:
```json
{
  "deployed": "gb_v1_20260311_155010",
  "staged":   "gb_v1_20260312_180803",
  "history": [...]
}
```
The API reads `registry.json` to identify the deployed model. Promotion
requires an explicit script call, not just a newer file timestamp.

**Signal:** model registry is a core MLOps primitive. Implementing even
a filesystem-based version demonstrates understanding of the concept and
its purpose.

---

## 5. Data Engineering

### 5.1 Typed Feature Store Interface

**Gap:** Features are computed by `build_features()` which calls
`get_feature_list()` and applies `fillna(0.0)`. The schema of the feature
space (column names, dtypes, expected ranges) exists only implicitly in the
training pipeline.

**Extension:** Define a typed feature schema in `src/data/feature_schema.py`:
```python
FEATURE_SCHEMA: dict[str, FeatureSpec] = {
    "TransactionAmt": FeatureSpec(dtype=float, fill=0.0, range=(0, None)),
    "card1": FeatureSpec(dtype=float, fill=0.0, range=(1, 18396)),
    ...
}
```
Validate the training data and inference payload against this schema.
Generate documentation from it automatically.

**Signal:** a feature schema is the contract between data engineering and
ML engineering. Implementing it prevents training-serving skew and makes
the feature space auditable.

---

## Summary

| Extension | Effort | Impact | Priority |
|---|---|---|---|
| CI with GitHub Actions | Low | High | 1 |
| Full-feature PSI monitoring | Low | High | 2 |
| Calibration analysis | Low | Medium | 3 |
| SHAP explainability | Medium | High | 4 |
| Temporal cross-validation | Medium | High | 5 |
| LightGBM comparison | Medium | Medium | 6 |
| Performance monitoring at deployed threshold | Low | Medium | 7 |
| Docker Compose multi-service | Medium | Medium | 8 |
| Model registry with promotion | Medium | Medium | 9 |
| Bootstrap confidence intervals | Low | Medium | 10 |
| Async transaction queue | High | High | 11 |
| Typed feature store | High | High | 12 |
| Feature selection | Medium | Medium | 13 |
