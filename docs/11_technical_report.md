# Technical Report: Fraud Detection ML System

## 1. System Architecture Overview

The system is structured around six cooperating components with explicit contracts between them:

```
Raw Data → Training Pipeline → Artifact Store → Scoring API → Monitoring → Retraining Trigger
```

Each component has a single responsibility and communicates with adjacent components through stable interfaces — file-based for offline components (artifact store, run metadata) and HTTP for the online serving layer.

**Training Pipeline** (`src/pipelines/training_pipeline.py`): orchestrates data loading, schema validation, temporal split, feature construction, model training, metric computation, and artifact serialization. Accepts a YAML config file that controls the model type, feature set, and split ratio. The entire pipeline is deterministic given a config and dataset.

**Artifact Store** (`artifacts/`): stores versioned model objects (`.pkl`), companion metadata (`.json` with `feature_list`, `threshold`, and evaluation metrics), and run records. The naming convention `{model}_{version}_{timestamp}.pkl` allows multiple versions to coexist without collision. The API always loads the lexicographically latest artifact matching the deployment pattern.

**Scoring API** (`app/main.py`): a FastAPI service that enforces the feature contract from training metadata. On startup, it loads the model and its `feature_list` from the artifact store into `app.state`. At inference time, it validates that all required columns are present in the payload, selects them in the exact training order, applies `fillna(0.0)`, and runs `predict_proba`. Missing features raise `HTTP 422`; model absence raises `HTTP 503`.

**Monitoring** (`src/utils/drift.py`, `scripts/monitor_model.py`): computes Population Stability Index between a reference distribution and the current prediction log. Outputs timestamped JSON reports to `artifacts/monitoring/`.

**Retraining Trigger** (`scripts/retrain_model.py`): reads a drift report, computes `max_psi`, and conditionally invokes the training pipeline.

---

## 2. Modeling Decisions

### Why Gradient Boosting over Logistic Regression and Random Forest

Three models were trained and evaluated on the same cost-sensitive framework: Logistic Regression (LR), Random Forest (RF), and Gradient Boosting (GB). The evaluation criterion was Expected Monetary Loss at the cost-optimal threshold, not ROC-AUC or accuracy.

GB achieved the lowest expected loss and the highest PR-AUC (0.409 vs lower values for LR and RF). The PR-AUC is the preferred ranking metric in this domain because the positive class is rare (~3.5%) and we care about precision–recall trade-offs, not rank order across the full threshold range.

LR was not selected despite its interpretability advantage. At a dataset scale of ~590k transactions with 380 numeric features, including many Vesta-engineered anonymized variables with non-linear distributions, a linear boundary cannot capture the structure needed to separate fraud from legitimate activity with sufficient precision.

RF was not selected because, in ensemble boosted models, the sequential correction mechanism tends to reduce residual errors on difficult cases — which in fraud detection are the unusual fraud patterns. RF builds trees independently, which limits its ability to concentrate capacity on hard-to-classify transactions.

GB remains the choice as long as: (1) latency requirements allow batch or near-real-time scoring (GB inference is sequential across estimators), (2) interpretability constraints do not mandate a simpler model, and (3) the cost differential over RF justifies the additional training time.

### Temporal Validation

All models are evaluated on a temporal hold-out: the 80th quantile of `TransactionDT` is used as the split boundary. Training data is everything before the cutoff; validation is everything after. This prevents the model from "seeing the future" and ensures that evaluation reflects the distribution shift that occurs over time in real transaction data.

Random splitting is explicitly prohibited. In a temporal dataset, a random split leaks future patterns into the training set and produces optimistically inflated metrics that do not generalize to deployment.

---

## 3. Evaluation Framework

The primary evaluation metric is **Expected Monetary Loss (EML)**:

```
EML(threshold) = Σ(false_negatives × transaction_amount) + Σ(false_positives × c_fp)
```

Where `c_fp = 5.0` represents the fixed operational cost of a false positive (a legitimate transaction incorrectly flagged). This cost includes analyst review time or the cost of a customer authentication challenge.

The **approve-all baseline** computes EML at threshold=1.0, where all transactions are approved and all fraud becomes a false negative:

```
baseline_loss = Σ(fraud_transaction_amounts)
```

The baseline on the validation set is **$609,934**. The served model (v2) reduces this to **$174,832** — a 71.3% reduction (v1: $251,945, 58.7%).

**Threshold selection** is performed via a discrete sweep. The original grid covered [0.01, 0.99] in 0.01 increments; the Phase 9 calibration analysis showed this clips the optimum for a well-separated model, so the grid now extends down to 0.001 in fine steps. On this dataset, `threshold* = 0.003` for v2 (v1 operated at 0.02; at the clipped 0.01 the v2 model leaves ~$18k of avoidable loss on the table).

The low optimal threshold reflects two structural properties of the problem: (1) severe class imbalance means the model assigns low absolute probabilities even to fraud cases, so the threshold must be correspondingly low to capture them; and (2) the asymmetric cost structure (fraud loss >> c_fp) incentivizes aggressive flagging to minimize missed fraud.

**Supporting metrics** recorded at the optimal threshold (v1 kept as the before/after comparison):

| Metric | v1 (sklearn GB) | v2 (LightGBM, served) | Interpretation |
|---|---|---|---|
| ROC-AUC | 0.861 | **0.930** | Strong discrimination across all thresholds |
| PR-AUC | 0.409 | **0.629** | Precision–recall trade-off for rare positives |
| Precision at threshold | 0.101 | **0.180** | Fraction of flagged transactions that are real fraud |
| FPR at threshold | 0.259 | **0.138** | Fraction of legitimate transactions flagged |

Note: in `src/models/metrics.py`, the stored metric `precision` is `TP/(TP+FP)` — the fraction of flagged transactions that are actual fraud.

---

## 4. Feature Engineering

Feature construction lives in `src/features/pipeline.py` and has two generations:

**v1 — `build_features()`** (stateless):

1. Calls `get_feature_list(df, feature_set="v1")` which returns all numeric columns excluding `isFraud`, `TransactionID`, and `TransactionDT`.
2. Selects those columns from the dataframe and applies `fillna(0.0)`.
3. Returns `(X, feature_list)` where `feature_list` is the ordered list used for this training run.

**v2 — `FeatureBuilderV2`** (stateful fit/transform, the served path since Phase 9): fits label/frequency-encoding tables on the train partition only, freezes them, and builds 494 engineered features (numeric base + categorical encodings + email split + time/amount + D-normalization) from 432 raw columns. The fitted builder is serialized *inside* the model artifact as the first step of a sklearn `Pipeline`, so `predict_proba` takes raw request columns. Which feature blocks were allowed to cross the research-to-serving boundary — and why UID aggregations were not — is documented in ADR-006.

The `feature_list` is serialized in the model metadata JSON. At inference time, the API reads this list from `_meta.json` and enforces it as a hard contract — the payload must contain exactly those columns (for v2, the raw input columns).

**Imputation strategy**: v1 applied `fillna(0.0)` uniformly — a deliberate simplification with known implications (see `docs/12_trade_off_analysis.md`). v2 removes imputation entirely: LightGBM routes missing values natively at each split, which keeps "missing" distinct from a legitimate zero. The API respects this via the `imputation: "native"` metadata flag (it must not zero-fill v2 payloads).

**Feature groups** in the dataset:

| Group | Columns | Count | Type |
|---|---|---|---|
| Transaction amount | `TransactionAmt` | 1 | Raw |
| Card identifiers | `card1–card5` | 5 | Encoded |
| Address | `addr1, addr2` | 2 | Encoded |
| Distance | `dist1, dist2` | 2 | Numeric |
| Count features | `C1–C14` | 14 | Behavioral |
| Time deltas | `D1–D15` | 15 | Temporal |
| Vesta features | `V1–V339` | 339 | Anonymized |
| Identity | `id_01–id_32` | 14 | Device/network |

Total: 392 raw features, 380 after removing non-numeric columns.

---

## 5. Scoring API Design

### Startup and State Management

Model loading is performed once at startup via a `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_deployed_model(app)
    yield
```

State is stored in `app.state` rather than module-level globals. This ensures that the state is scoped to the application instance, making test isolation clean (the test suite replaces `_load_deployed_model` via `monkeypatch` before entering the `TestClient` context manager).

### Feature Contract Enforcement

The API enforces the training feature contract with a hard fail:

```python
missing = [c for c in feature_list if c not in df.columns]
if missing:
    raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")
```

This is a deliberate design decision: the API does not attempt to infer missing features or fill them silently. Silent inference of missing features would allow schema drift to go undetected and produce wrong predictions without raising an error. Fail-fast behavior makes schema drift visible immediately.

### Pydantic v2 Schema

The `Transaction` model uses `model_config = ConfigDict(extra="allow")`, accepting arbitrary additional fields. This allows callers to include all raw fields without the API rejecting unknown columns; the `feature_list` filter ensures only training-validated features reach the model.

### Error Contract

| Condition | HTTP Status |
|---|---|
| Model not loaded on startup | `503 Service Unavailable` |
| Feature list missing from metadata | `503 Service Unavailable` |
| Required feature absent from payload | `422 Unprocessable Entity` |
| Successful inference | `200 OK` |

---

## 6. Drift Monitoring

### PSI Implementation

Population Stability Index is computed in `src/utils/drift.py`. Key design decisions:

**Quantile-based binning**: bins are derived from the reference distribution's quantiles, not uniform intervals. This ensures each bin contains a meaningful fraction of the reference data, avoiding degenerate bins on skewed distributions like `TransactionAmt`.

**Epsilon for zero bins**: when a bin receives zero observations in either distribution, the formula uses `ε = 1e-4` instead of skipping the bin or returning infinity. Skipping zero bins silently underestimates drift — if the current distribution falls entirely outside the reference bins (extreme drift), skipping would return PSI=0. The epsilon ensures this extreme case produces a large, finite PSI value.

**Zero-denominator guard**: if all observations in the current window fall outside the reference bin range, `cur_counts.sum() == 0`, which would produce NaN via division. This is handled explicitly before the proportion calculation:

```python
cur_perc = cur_counts / cur_total if cur_total > 0 else np.zeros(...)
```

### PSI Interpretation

| PSI | Signal |
|---|---|
| < 0.10 | Stable — no meaningful shift detected |
| 0.10 – 0.20 | Slight shift — increase monitoring frequency |
| > 0.20 | Significant drift — consider retraining |

---

## 7. What Would Change in Production

This system simulates production patterns but is not production-ready as built. The key gaps:

**Model registry**: the current artifact store is a directory with a naming convention. A production system would use a centralized model registry (MLflow, Vertex AI, SageMaker) with explicit promotion workflows: candidate → staging → production. The API would pull from the registry at startup rather than reading local files.

**Feature store**: features are recomputed from raw data at training time using `build_features()`. In production, time-sensitive features (behavioral counts, recency deltas) must be precomputed and served with sub-millisecond latency to meet scoring SLAs. A feature store (Feast, Tecton, or a custom Redis-backed service) would decouple feature computation from scoring.

**Shadow deployment and A/B testing**: promoting a new model to 100% traffic without evaluation is high risk in fraud detection. A production workflow would run the new model in shadow mode (scoring all traffic but not acting on it) to validate performance before cutover, then gradually shift traffic via a canary or A/B split.

**Async inference**: the current API is synchronous. For high-throughput fraud scoring (thousands of transactions per second), async processing with a message queue (Kafka, SQS) would decouple transaction receipt from scoring, enabling horizontal scaling.

**Circuit breakers**: if the model or upstream feature pipeline fails, the API should fail open (approve all) or fail closed (block all) based on business policy, rather than returning errors. This prevents a model failure from causing transaction failures.

**Monitoring scope**: the current implementation monitors `TransactionAmt` distribution only. A production monitoring system would track PSI for the top-N model features by importance, monitor the score distribution itself (score drift is often the earliest signal of model degradation), and track business KPIs (fraud rate, chargeback rate) in addition to statistical metrics.

**Retraining governance**: the current trigger is fully automated. Production systems typically require human approval for model promotions, especially in regulated industries. The trigger here should produce an alert and a candidate model rather than deploying automatically.
