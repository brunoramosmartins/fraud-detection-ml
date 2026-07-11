# ADR-007 — LightGBM Replaces sklearn GradientBoosting for Phase 8+

## Status

Accepted

## Context

Phases 1–7 used sklearn's `GradientBoostingClassifier` (80 trees, depth 5).
It was chosen for zero additional dependencies and API uniformity with the
LR/RF alternatives in `src/models/factory.py`. Phase 8 targets a
Kaggle-competitive score on the IEEE-CIS competition, where the deployed
model's holdout AUC (0.861) sits ~0.08 below the frozen leaderboard's
competitive zone (~0.94). Part of that gap is attributable to the model class
itself — this is pre-registered as hypothesis H1 in `docs/kaggle/research.md`.

Three candidates were considered to replace it:

1. **LightGBM** — histogram-based GBDT, native NaN handling, native
   categorical support, leaf-wise growth, multithreaded; the dominant model
   family in the 2019 competition's top solutions.
2. **XGBoost** — equivalent accuracy class; the 2019 winner used it. Slightly
   slower on CPU for this data shape at comparable settings.
3. **CatBoost** — ordered target statistics for categoricals; strongest when
   target encoding is the main signal, but slower to iterate and less used in
   this competition's public solutions, making results harder to benchmark.

## Decision

LightGBM becomes the model for all Phase 8 experiments (EXP-001 onward) and,
after Phase 9, for the served model (`factory.py` gains an `"lgbm"` entry —
the existing `"lr"`, `"rf"`, `"gb"` entries remain for comparability).
XGBoost is kept as an optional cross-check in EXP-005 only. CatBoost is not
adopted.

Two properties are load-bearing for the experiment design, not just speed:

- **Native NaN handling** removes the `fillna(0)` imputation, which conflates
  "missing" with a legitimate zero value. H1 deliberately bundles the model
  swap with this imputation change, since `fillna(0)` is an artifact of the
  sklearn pipeline rather than an independent decision.
- **Training speed** (histogram binning + multithreading) makes the
  one-submission-per-feature-block protocol affordable on Kaggle CPU
  notebooks; sklearn GB at 590k × 380+ is single-threaded and takes hours per
  fit, which would make 6-fold GroupKFold experiments impractical.

## Consequences

**Positive:**
- H1 measures the model-class effect explicitly instead of leaving it as an
  assumed improvement.
- 6-fold GroupKFold per experiment becomes feasible (minutes per fold).
- Native categorical support gives ex02 a benchmark to compare frequency
  encoding against.

**Negative:**
- New production dependency (`lightgbm`) in `requirements.txt`; the Docker
  image grows and the artifact contract in `src/models/artifacts.py` must be
  verified against LightGBM's sklearn wrapper for joblib round-tripping.
- Results from Phases 1–7 (sklearn GB) and Phase 8+ (LightGBM) are not
  directly comparable at the hyperparameter level; comparisons are made at
  the score level under the pinned validation protocol instead.
- Leaf-wise growth overfits more readily than depth-wise at equal tree
  counts; `num_leaves`/`min_data_in_leaf` need explicit control (EXP-005
  tuning pass), whereas sklearn's depth cap was self-limiting.
