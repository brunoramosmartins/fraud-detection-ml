# ADR-003 — Temporal Split as the Only Valid Validation Strategy

## Status

Accepted

## Context

The IEEE-CIS dataset contains transactions ordered by time (`TransactionDT`).
Two validation strategies were considered:

1. **Random split:** shuffle the dataset and reserve a random fraction for
   validation. Standard practice for i.i.d. datasets.
2. **Temporal split:** reserve the most recent fraction of transactions for
   validation, using earlier data for training exclusively.

## Decision

Temporal split only. Random splits are explicitly prohibited.

Implementation: `split_quantile=0.8` in `configs/model_gb_v1.yml` splits
at the 80th quantile of `TransactionDT`. Training uses all rows before the
cutoff; validation uses all rows after.

## Consequences

**Positive:**
- Evaluation reflects realistic deployment conditions: the model is trained
  on past data and evaluated on future data, exactly as it will operate
  in production.
- Feature leakage from temporal correlations is prevented. In a random
  split, a transaction from day 100 could be in the training set while
  a transaction from day 50 is in the validation set. Temporal patterns
  (seasonality, fraud campaign timing) would leak across the split.
- Business metrics (expected monetary loss, fraud rate over time) are
  computed on a realistic future window, not an artificially mixed sample.

**Negative:**
- The training set is always larger than what a proportional random split
  would give. The 80/20 split means 80% of data is used for training —
  but all of it is from the past, which may underrepresent recent fraud
  patterns.
- Hyperparameter tuning is harder: cross-validation with temporal splits
  requires multiple non-overlapping future windows, which increases
  implementation complexity. This project uses a single split and manual
  configuration instead.
- Validation metrics are computed on one time window only. A single window
  may not be representative if there are seasonal effects. Multiple
  temporal folds would give more robust metric estimates.

**Why this matters:** models trained on historical data with a random split
routinely show 5–15% inflated ROC-AUC compared to temporal splits on
time-series datasets. Reporting metrics from a random split as if they
reflect production performance is a form of evaluation leakage. This ADR
establishes that this project does not do that.
