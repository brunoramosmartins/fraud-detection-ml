# Phase 9 — Production Reintegration

**Objective:** decide which Phase 8 Kaggle feature blocks are feasible in the
served system, retrain the production model on that subset, and refresh every
number the repository publishes. The Kaggle model and the production model
intentionally diverge — documenting that divergence is the point (ADR-006).

**Dates:** 2026-07-22 (single working session; roadmap budgeted ~14h).

**Ends with:** served model v2 (LightGBM), ADR-006/007, calibration + SHAP
reports, a repo-wide metric sweep, and this note.

---

## Issue #23 — ADR-006: Kaggle-vs-serving feature boundary

Wrote [ADR-006](../decisions/ADR-006-kaggle-features-vs-serving.md). The
boundary criterion: a block crosses only if computable from (a) the single
incoming row plus (b) state frozen at training time and shipped in the
artifact. Decision:

- **IN:** LightGBM + native NaN, categorical label/frequency encodings (frozen
  lookup tables), email split, time/amount features, D-column normalization.
- **OUT (deferred):** UID key + per-UID aggregates (need a client feature
  store / expanding window) and the transductive aggregation engine
  (structurally impossible at scoring time; its gain concentrated on
  already-seen entities per the EXP-007 seen/unseen diagnostic).

`v2` is, by construction, the EXP-003 configuration — so the ADR pre-registered
the expected retrain result (internal holdout AUC ≈ 0.93).

## Issue #24 — v2 feature builder, lgbm factory, config

- `src/features/engineering.py`: split each encoder into `fit_*` / `apply_*`
  halves (frozen dict + row-local application) so serving can apply frozen
  state without the training data; the original `frequency_encode` /
  `label_encode` became compositions (behavior unchanged).
- `src/features/feature_registry.py`: explicit `v2` block constants
  (`V2_EMAIL_SPLITS`, `V2_FREQ_NUMERIC_CATS`, `V2_D_NORM_COLS`); `get_feature_list("v2")`
  raises (v2 carries fitted state, is not a static list).
- `src/features/pipeline.py`: `FeatureBuilderV2` (fit/transform), serializable
  by joblib, fail-fast on missing columns (ADR-004), NaN preserved.
- `src/models/factory.py`: `"lgbm"` entry (EXP-003 hyperparameters), lazy import
  so v1 models stay usable without lightgbm.
- `src/pipelines/training_pipeline.py`: v2 branch packages `Pipeline([builder,
  clf])` so `predict_proba` takes RAW request columns; metadata records the raw
  input contract, the engineered feature list, and `imputation: "native"`.

## Issue #25 — Local retrain + operating point

Retrained locally (~4 min, LightGBM histogram + `n_jobs=-1`). Holdout ROC-AUC
**0.929588** — reproduces the EXP-003 Kaggle notebook to the 4th decimal,
confirming the `src/` builder constructs the same features. See the
private-LB decomposition in [research.md](research.md).

## Issue #26 — Calibration check

[`scripts/calibration_check.py`](../../scripts/calibration_check.py). Two
findings:

1. The training threshold grid started at 0.01 and **clipped the EML optimum**.
   The true optimum is **0.003** (EML $174,832 vs $192,943 at the clipped
   edge, −9.4%). Fixed the `threshold_sweep` default grid to extend to 0.001.
2. **Isotonic calibration tested leak-free** (fit on the earlier validation
   half, evaluated on the later half): it improves ECE substantially
   (0.0142 → 0.0035; the model is under-confident) but the EML gain is +0.71%
   ≈ noise. Reason: isotonic is monotone, so it cannot change the ranking and
   cannot improve EML at the optimal threshold — it only re-labels the
   threshold axis. **Not adopted.** (Would matter only under a per-transaction
   decision-theoretic rule; noted as future work.)

## Issue #27 — SHAP

[`scripts/shap_analysis.py`](../../scripts/shap_analysis.py) using LightGBM's
native TreeSHAP (`pred_contrib=True`) — exact, no external dependency. The
block-level attribution quantifies the ADR-006 boundary: the engineered blocks
are 19% of features but ~39% of attribution, and D-normalization has the
highest signal density of any block (~7× the numeric base per feature). Closes
the per-prediction-explanation half of limitation #6 in docs/13.

## Issue #28 — Repo-wide metric sweep + v2 promotion

Served-model metrics changed, so per the consistency rule the sweep touched
README (Key Results before/after table, badges, phases, skills), docs/10–13,
`notebooks/06_results_dashboard.ipynb` (parametrized on `MODEL_GLOB`, handles
the v2 Pipeline, fine grid, cleared stale outputs), docs/09 (commands, rollback
env var), and the Makefile. v2 promoted to the API default
(`DEFAULT_MODEL_GLOB="lgbm_v2_*.pkl"`, v1 one env var away). The interview
runbook was un-tracked (kept as personal local notes, gitignored) and its
references removed from the README and Makefile; the public-facing interview
material moves to Phase 10's README Kaggle Results section (#31) and the public
notebook (#30) instead of a versioned runbook.

## Issue #29 — API contract test

[`tests/test_api_v2_contract.py`](../../tests/test_api_v2_contract.py): raw
transaction scored end-to-end through the Pipeline, NaN reaches the model
un-imputed, unseen categories score via sentinels, missing raw column → 422,
and the v1 fillna(0) path still holds. Plus `tests/test_pipeline_v2.py` (builder
contract) and `tests/test_calibration.py`.

## Headline result

| Metric | v1 (sklearn GB) | v2 (LightGBM, served) |
|---|---|---|
| ROC-AUC | 0.861 | **0.930** |
| PR-AUC | 0.409 | **0.629** |
| Expected loss reduction | 58.7% ($357,989) | **71.3% ($435,102)** |
| Operating threshold | 0.02 | **0.003** |
| Precision / FPR at threshold | 10.1% / 25.9% | **18.0% / 13.8%** |

## Unplanned work — build-system centralization

Not in the roadmap. The mid-phase Python 3.10 → 3.14 upgrade (fresh venv)
surfaced that the pinned `requirements*.txt` had no wheels for 3.14 and that
`pyproject.toml` declared no dependencies at all (the editable install pulled
nothing — the reported "all tests fail" was an empty venv). Centralized all
dependencies into `pyproject.toml` with version ranges + optional `[dev]` /
`[notebooks]` extras; dropped both requirements files; CI now runs a 3.10/3.14
matrix. Capped `pandas<3.0` (pandas 3 changes string-dtype defaults the
`dtype=="O"` categorical detection relies on).

---

## Lessons Learned

<!-- Author to complete in first person — this is interview/writeup material and
     must be in your own words. Prompts below are only scaffolding; delete them.

  - The ADR-006 boundary as a reusable idea: what makes a Kaggle feature
    "die at the serving boundary"? (transductive vs row-local vs frozen-state)
  - What did the exact 0.929588 reproduction prove about porting notebook code
    into a src/ package — and why does that matter in an interview?
  - The isotonic result: why "monotone calibration can't improve optimal-
    threshold EML" is a sharper thing to say than "we tried calibration".
  - The threshold-grid clipping: a $18k lesson about default sweep ranges.
  - SHAP block-attribution as evidence FOR a design decision, not just an
    explainability checkbox.
-->

## Failed Attempts

<!-- Author to complete in first person — honest nulls and dead ends are the
     most valuable interview material. Prompts below are only scaffolding.

  - Isotonic calibration: adopted? No — record why it was the right call to
    reject despite the ECE improvement looking attractive.
  - The empty-venv false alarm (Python 3.14): what actually broke vs what it
    looked like it broke, and how the diagnosis went.
  - The sklearn 1.9 test stub (Pipeline check_is_fitted rejecting a plain
    stub) — a small one, include only if useful.
  - The reliability-table degenerate-bin bug (constant scores → zero bins).
-->
