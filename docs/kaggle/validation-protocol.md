# Validation Protocol — Phase 8 Experiments

Every Phase 8 experiment is evaluated under this pinned protocol. Experiments
reference this document; they never redefine it. Changes to this protocol
require a new version section here and invalidate cross-version comparisons.

## Protocol v1

### Data

- Source: `/kaggle/input/ieee-fraud-detection/` (Kaggle) or `data/raw/` (local).
- `train_transaction.csv` LEFT JOIN `train_identity.csv` on `TransactionID`.
- Test-set identity columns use dashes (`id-01`); they are renamed to
  underscores (`id_01`) to match the training schema before any feature code
  runs.

### Scheme A — Temporal holdout (production-identical)

- Cutoff: `TransactionDT` quantile **0.8** computed on the full training set;
  train = rows strictly before the cutoff, holdout = rows at/after.
- Identical to `src/data/split.py::temporal_train_val_split` — this keeps every
  experiment comparable with the production baseline (AUC 0.861).
- The holdout is the sample for all DeLong tests.

### Scheme B — Month-wise GroupKFold

- Month index: `DT_M = floor(TransactionDT / (86400 * 30.44))` (≈ calendar
  months), leave-one-month-out.
- **Amendment (2026-07-10, EXP-001):** the bucketing yields **7** month groups
  in practice (DT_M 0–6), not the 6 estimated when this protocol was written.
  Fold count follows the data: 7 folds. Recorded before any cross-experiment
  Scheme B comparison existed; applies uniformly to all experiments.
- Group = `DT_M`; no shuffling; every fold trains on the remaining months.
- Report per-fold AUC and mean ± std. Early stopping, when used, is fit
  per-fold on the fold's own validation month.

Both schemes are computed on every experiment **from EXP-001 onward** — H4
compares them over EXP-001..004. EXP-000 is exempt from Scheme B: it has no
predecessor to compare against, sits outside H4's scope, and its single-threaded
sklearn GB makes 6 extra fits prohibitive (the cost ADR-007 removes).

### Seeds

- Model seed: 42 everywhere (LightGBM `seed=42`, sklearn `random_state=42`).
- EXP-005 seed averaging: seeds {42, 1337, 2019}, documented in the registry.

### Metrics recorded per experiment

| Metric | Where |
|---|---|
| Holdout ROC-AUC (Scheme A) | registry + notebook output |
| GroupKFold per-fold AUC, mean ± std (Scheme B) | registry + notebook output |
| DeLong ΔAUC vs predecessor, 95% CI, p-value (on Scheme A holdout) | registry |
| Public LB / Private LB | submission log |
| Validation-LB gaps: abs(A − private), abs(B mean − private) | registry (feeds H4) |

### DeLong procedure

1. Each experiment's notebook saves its Scheme-A holdout predictions as
   `holdout_pred_expNNN.csv` (`TransactionID, y_true, score`).
2. DeLong is computed **off-notebook** (locally) from the two holdout
   artifacts, aligned by `TransactionID`, using the tested module
   `src/models/delong.py::delong_roc_test(y_true, scores_new, scores_old)`
   → ΔAUC, 95% CI, z, two-sided p-value.
3. Significance level α = 0.05, two-sided. No correction within H1–H3 (each
   hypothesis is a single pre-registered comparison); any post-hoc exploratory
   comparison must be labeled as such.

**Why off-notebook (amended 2026-07-10, EXP-003):** an earlier design ran the
DeLong step *inside* the submission notebook, reading the predecessor's holdout
artifact via an attached kernel source. The kernel-attach step failed
intermittently and, worse, aborted the run *before* `submission.csv` was
written — wasting the full training time. Submission notebooks are now
modeling-only; the DeLong comparison is a separate local step over the
downloaded holdout artifacts, using the same tested module.

### Submission discipline

- One feature block per submission. The feature diff vs the predecessor is
  stated in the registry entry before the run.
- Registry entry BEFORE running; submission-log entry BEFORE upload.
- The final model (EXP-005) is a single LightGBM (seed averaging allowed);
  stacking/ensembling is out of scope by design.
