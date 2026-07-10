# Experiment Registry — Phase 8

Rule: every experiment is registered here **before** it runs. The entry states
the expected result; the actual result is filled in afterwards. Protocol:
`validation-protocol.md` v1. Hypotheses: `research.md`.

| Field | Meaning |
|---|---|
| ID | EXP-xxx, immutable |
| Hypothesis | H1–H4, `exploratory`, or `none` |
| Feature diff | exactly what changed vs the predecessor experiment |
| Expected | predicted ΔAUC (registered before running) |
| A / B | Scheme A holdout AUC / Scheme B GroupKFold mean ± std |
| DeLong | ΔAUC [95% CI], p vs predecessor (Scheme A) |
| LB | public / private (from submission log) |

---

## EXP-000 — Baseline: current production model, external anchor

| Field | Value |
|---|---|
| Registered | 2026-07-10 |
| Hypothesis | none (anchor) |
| Notebook | `notebooks/kaggle/k00_baseline_submission/` |
| Predecessor | — |
| Feature diff | — faithful reproduction of the production pipeline: 380 numeric-only features, `fillna(0)`, sklearn GB (80 trees, depth 5, lr 0.1, min_samples_leaf 100, subsample 0.8, seed 42), trained on Scheme-A train partition only |
| Config | `configs/model_gb_v1.yml` equivalents, hardcoded in notebook for reproducibility |
| Expected | Scheme-A holdout AUC ≈ 0.861 (reproduction check, tolerance ±0.003); private LB expected *below* holdout (temporal drift + one-month gap in test period), rough guess 0.83–0.86 |
| A (holdout AUC) | **0.8614** — reproduction check PASSED (expected 0.861 ± 0.003) |
| B (GroupKFold) | exempt (see protocol) |
| DeLong | — (no predecessor) |
| LB public / private | **0.8896 / 0.8749** (SUB-001, 2026-07-10) |
| Verdict / notes | Anchor established. Fit time 16.2 min on Kaggle CPU. Kaggle env inferred 400 numeric features vs 380 documented locally (pandas dtype-inference difference across versions); holdout AUC matched regardless, so the pipelines are functionally equivalent. Data mounts at `/kaggle/input/competitions/ieee-fraud-detection` (new layout) — notebook locates files by search. Holdout predictions saved as `holdout_pred_exp000.csv` for the EXP-001 DeLong test. **Expectation miss (recorded):** registered guess was private LB 0.83–0.86 (below holdout); actual came in above the holdout (0.8749 private, 0.8896 public). Interpretation: AUC on different time windows/populations is not directly comparable; the registered expectation over-weighted temporal degradation. Public > private is consistent with drift increasing within the test period. Both scores remain far below the competitive zone (~0.93+), as intended for the anchor. |

---

## EXP-001 — H1: LightGBM on identical numeric features

| Field | Value |
|---|---|
| Registered | 2026-07-10 |
| Hypothesis | **H1** |
| Notebook | `notebooks/kaggle/k01_lgbm_numeric/` |
| Predecessor | EXP-000 |
| Feature diff | **none** — same runtime-inferred numeric feature list. Only two changes, bundled by design (see H1 statement): model class (sklearn GB → LightGBM) and imputation (`fillna(0)` removed; native NaN handling) |
| Config | LightGBM, fixed and registered before running: `learning_rate=0.05`, `num_leaves=192`, `min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `seed=42`. `n_estimators` via early stopping (patience 200) on an ES split *inside* the Scheme-A train partition (last month of the train window — the holdout is never used for model selection); refit on the full train partition at `int(best_iter * 1.1)` |
| Expected | H1 threshold: private LB ΔAUC ≥ +0.015 vs EXP-000 (i.e. private ≥ 0.890); DeLong on Scheme-A holdout significant at α=0.05 in the same direction. Scheme B (GroupKFold) reported for the first time — feeds H4 |
| A (holdout AUC) | *(pending)* |
| B (GroupKFold) | *(pending)* |
| DeLong vs EXP-000 | *(pending)* |
| LB public / private | *(pending — see SUB-002)* |
| Verdict / notes | *(pending)* |
