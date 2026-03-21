# Demo Runbook — Fraud Detection ML System

This document is a structured walkthrough for presenting the project in a
technical interview setting. Each section includes the commands to run,
the key numbers to cite, and likely follow-up questions with answer frameworks.

---

## Prerequisites

```bash
# Clone and set up
git clone https://github.com/brunoramosmartins/fraud-detection-ml.git
cd fraud-detection-ml
make setup

# Place dataset files in data/raw/ (IEEE-CIS from Kaggle)
# Then verify everything works:
make demo
```

---

## 5-Minute Walkthrough

### 1. The Problem (30 seconds)

> "This project simulates a fraud detection system for card-not-present transactions. The goal is not to maximize accuracy — it's to minimize expected monetary loss, which means explicitly modeling the cost asymmetry between missing fraud and generating false alarms."

**Command to show:**
```bash
cat docs/10_executive_summary.md | head -30
```

**Key number to cite:** The baseline loss (approve all) is **$609,934**. The model reduces it to **$251,945** — a **58.7% reduction**.

---

### 2. Training the Model (45 seconds)

> "The full pipeline is one command. It loads the data, validates the schema, does a temporal split — no random splits, ever — builds features, trains a Gradient Boosting model, and saves the artifact with its metadata."

**Command to show:**
```bash
make train
# Output: Training finished. Model saved to artifacts/models/gb_v1_<ts>.pkl
```

**Key decisions to mention:**
- Temporal split at the 80th quantile of `TransactionDT`
- 380 numeric features, zero-filled imputation
- Threshold selected by minimizing EML over a sweep from 0.01 to 0.99
- Optimal threshold: **0.02** (much lower than 0.5 — explain why if asked)

---

### 3. Running the Tests (30 seconds)

> "The test suite covers the API scoring endpoint and the PSI drift utility. Tests use lightweight stubs — no real model artifact needed in CI."

```bash
make test
# 14 passed in ~3s
```

**Key point:** the TestClient fixture monkeypatches `_load_deployed_model` before the lifespan runs — this was a deliberate fix to avoid a common trap of creating the client at module scope.

---

### 4. Starting the API (30 seconds)

> "The inference service is a FastAPI app. It enforces the exact feature contract from training — if a required feature is missing from the payload it returns 422, it doesn't silently impute or guess."

```bash
# Terminal 1
make api

# Terminal 2 — verify it's up
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}

# Manual prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transactions":[{"TransactionID":1,"TransactionAmt":250.0}]}' | python -m json.tool
```

**Key design points:** `app.state` for runtime objects, `lifespan` context manager (not deprecated `on_event`), `HTTPException` with typed status codes.

---

### 5. Simulating Transactions (45 seconds)

> "This script loads the dataset, shuffles it, and sends batches to the API — simulating a production transaction stream. JSON sanitization handles NaN and inf before serialization. Failed batches are logged and skipped, not thrown."

```bash
# Terminal 2 (API still running in Terminal 1)
make simulate
# Batch 1: scored 100 transactions.
# ...
# Saved 500 prediction records to artifacts/monitoring/predictions/predictions_<ts>.csv
```

---

### 6. Monitoring Drift (30 seconds)

> "PSI is computed between the reference distribution and the production predictions log. The implementation uses quantile-based binning and epsilon for zero bins — which prevents silently underestimating drift when the current window falls outside reference bins."

```bash
make monitor
# PSI[TransactionAmt]=0.XXXX  [OK / WARNING]
# Saved drift report to artifacts/monitoring/drift/drift_report_<ts>.json
```

---

### 7. The Bigger Picture (30 seconds)

> "The monitoring can trigger retraining. If max PSI exceeds 0.2, the training pipeline re-runs automatically. In production you'd add human approval before promoting the new model — this simulation demonstrates the trigger logic, not the governance."

```bash
# Forced retraining simulation
echo '{"TransactionAmt": 0.35}' > /tmp/drift_fake.json
python scripts/retrain_model.py \
  --config configs/model_gb_v1.yml \
  --drift-report /tmp/drift_fake.json
```

---

## Anticipated Interview Questions

### System Design

**"How would this work at 10,000 transactions per second?"**

The synchronous batch API cannot meet this SLA. The production architecture would be: transactions arrive via Kafka → scoring service consumes from the queue asynchronously → results published to an output topic → downstream systems read from the output topic. The scoring service scales horizontally. See `docs/12_trade_off_analysis.md`, Trade-off 6.

**"How do you deploy a new model without downtime?"**

Shadow deployment: run the new model in parallel on all traffic, log scores without acting on them, validate EML and distribution against the current model for 24–48 hours, then cut over. Requires a model registry and a traffic routing layer that this simulation doesn't implement. See `docs/11_technical_report.md`, Section 7.

**"What happens if the feature pipeline breaks at inference time?"**

The API returns `HTTP 503` if the model isn't loaded and `HTTP 422` if required features are missing. What's missing is a circuit breaker: if the API fails, transactions should fail-open (approve) or fail-closed (block) based on business policy, not propagate an error to the caller. See `docs/13_model_limitations.md`, Limitation 7.

---

### Model and Evaluation

**"Why is the threshold 0.02 and not 0.5?"**

Three reasons: (1) the fraud rate is ~3.5%, so the model's calibrated probabilities are low even for fraud cases; (2) the cost of missing fraud (the transaction amount) is much larger than the cost of a false alarm (fixed fee of 5 units); (3) threshold is selected by minimizing EML, not by convention. The threshold where total expected cost is lowest is 0.02. At this point the model captures ~80% of fraud at a 26% false positive rate.

**"Why Gradient Boosting and not a neural network?"**

At ~590k rows with 380 numeric features, gradient boosting routinely matches or outperforms neural networks. The sequential error-correction in boosting concentrates capacity on hard cases — precisely what fraud detection needs, since rare and unusual fraud patterns are the most costly to miss. A neural network would require substantially more hyperparameter tuning and infrastructure with no guaranteed gain. See `docs/11_technical_report.md`, Section 2.

**"What does PR-AUC of 0.409 mean in practice?"**

A random classifier would achieve PR-AUC equal to the fraud rate (~0.035). The model achieves 0.409 — about 12× better than random on the precision-recall trade-off. In highly imbalanced datasets, PR-AUC is more informative than ROC-AUC because it directly measures performance on the rare positive class without being dominated by the large number of true negatives.

**"How did you prevent data leakage?"**

All models are evaluated on the most recent 20% of transactions by `TransactionDT`. The training set never sees any observation from the validation period. Random splits are explicitly prohibited in the codebase. Additionally, the feature contract (`feature_list` from `_meta.json`) enforces that only features computed at training time are used at inference — no post-transaction information can leak.

---

### Production Readiness

**"Is this production-ready?"**

No, and I'm explicit about that in `docs/13_model_limitations.md`. The gaps are: no model registry with promotion workflows, no feature store for sub-millisecond retrieval, no shadow deployment or A/B testing, monitoring covers one feature instead of the full feature set, the retraining trigger is automatic (production requires human approval), and the API has no authentication or circuit breakers. The system demonstrates correct patterns and contracts; it doesn't replace production engineering.

**"How would you explain a fraud decision to a customer?"**

The current model doesn't support explanations — it returns a probability score, not a rationale. Adding SHAP TreeExplainer would provide per-transaction feature attributions. This is legally significant under LGPD (Brazil) and similar regulations that require automated financial decisions to be explainable upon request. See `docs/13_model_limitations.md`, Limitation 6.

---

### Engineering Decisions

**"Why did you use `app.state` instead of global variables?"**

Two reasons: (1) test isolation — with global variables, you can't cleanly inject a stub model without module-level side effects; `app.state` is scoped to the application instance and can be replaced per-test. (2) The FastAPI `lifespan` pattern requires passing `app` explicitly to the loader function, which naturally leads to `app.state`. See `docs/decisions/ADR-005-app-state-for-model-storage.md`.

**"Why does the PSI use epsilon instead of skipping zero bins?"**

Skipping zero bins silently underestimates drift. If the current window falls entirely outside the reference bin range (extreme drift), skipping returns PSI=0 — the wrong answer. Epsilon (1e-4) ensures the formula remains defined for all bins, including those with no observations, and produces a large finite PSI for extreme drift. The fix was discovered by a failing test: `test_compute_psi_nonoverlapping_distributions_is_finite`.

---

## Files to Navigate During Demo

| Topic | File |
|---|---|
| Business impact | `docs/10_executive_summary.md` |
| Architecture | `docs/diagrams/` |
| Model decisions | `docs/11_technical_report.md` |
| Trade-offs | `docs/12_trade_off_analysis.md` |
| Limitations | `docs/13_model_limitations.md` |
| Hyperparameters | `docs/14_hyperparameter_guide.md` |
| ADRs | `docs/decisions/` |
| Extensions | `docs/15_extensions_roadmap.md` |
| API source | `app/main.py` |
| Training pipeline | `src/pipelines/training_pipeline.py` |
| PSI implementation | `src/utils/drift.py` |
| Results dashboard | `notebooks/06_results_dashboard.ipynb` |
