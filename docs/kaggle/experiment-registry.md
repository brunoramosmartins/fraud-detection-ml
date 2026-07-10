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
| A (holdout AUC) | **0.9124** |
| B (GroupKFold) | **0.9296 ± 0.0127** — per-fold: [0.9031, 0.9285, 0.9383, 0.9297, 0.9408, 0.9234, 0.9433] (7 folds — see protocol amendment) |
| DeLong vs EXP-000 | ΔAUC **+0.0510** [95% CI +0.0461, +0.0558], z = 20.57, p = 4.7e-94 — significant, predicted direction |
| LB public / private | **0.9134 / 0.8877** (SUB-002, 2026-07-10) |
| Verdict / notes | **H1: INCONCLUSIVE (leaning supported)** per the pre-registered rule: internal DeLong strongly significant (+0.0510), but private LB Δ = +0.0128 < +0.015 threshold. Public LB Δ = +0.0238 *would* have met the bar. Key finding: the model-class gain decays with temporal distance (holdout +0.051 → public +0.024 → private +0.013); the registered threshold ignored drift decay. H4 gap data: \|B mean − private\| = 0.0419, \|A − private\| = 0.0247 (temporal split closer, this experiment). ES: best_iter 213 → refit 234 rounds; total runtime ~18 min (vs 16 min for a single sklearn GB fit — 8 LightGBM fits in the same budget, confirming ADR-007's speed claim). |

---

## EXP-002 — H2: categorical features restored

| Field | Value |
|---|---|
| Registered | 2026-07-10 (before running) |
| Hypothesis | **H2** |
| Notebook | `notebooks/kaggle/k02_categoricals/` |
| Predecessor | EXP-001 |
| Feature diff | + **label encoding** of every object-dtype column (~31: ProductCD, card4/card6, P_/R_emaildomain, M1–M9, object-typed id_12–id_38, DeviceType, DeviceInfo), missing as its own category, unseen → −1; + **frequency encoding** (train-fit, normalized) of the same object columns plus the numeric-coded high-cardinality categoricals card1, card2, card3, card5, addr1, addr2; + **email provider/suffix split** of P_/R_emaildomain (4 derived columns, label- and frequency-encoded). Canonical implementations: `src/features/engineering.py`. Model and params unchanged from EXP-001. |
| Config | LightGBM identical to EXP-001 (registered params, seed 42). Encoders fit on each fold's training months (Scheme B) and on the train partition (Scheme A model + submission) — never on scored rows. |
| Expected | H2 threshold: private LB ΔAUC ≥ +0.020 vs EXP-001 (private ≥ 0.9077); DeLong on holdout significant. Given H1's drift-decay finding, internal Δ is expected to exceed LB Δ: registered guess holdout Δ +0.025 to +0.040. |
| A (holdout AUC) | *(pending — paste summary block from notebook output)* |
| B (GroupKFold) | *(pending — paste summary block from notebook output)* |
| DeLong vs EXP-001 | *(pending — paste summary block from notebook output)* |
| LB public / private | **0.9251 / 0.8968** (SUB-003, 2026-07-10) |
| Verdict / notes | *(preliminary, internals pending)* External evidence: private Δ = **+0.0091** < +0.020 threshold (criterion b failed) AND smaller than H1's realized model-swap gain (+0.0128 private, +0.0238 public vs +0.0116 public here) — the comparative core of H2 ("categoricals outgain the model swap") fails externally in both test periods. Final verdict (inconclusive vs rejected) awaits the internal DeLong result. Note: 0.8968 private is still the series best — the block helps, just less than pre-registered. |

---

## EXP-003 — Exploratory: time and amount features

| Field | Value |
|---|---|
| Registered | 2026-07-10 (before running) |
| Hypothesis | exploratory (no hypothesis — one feature block per submission; feeds H4 gap tracking and sets the predecessor for H3) |
| Notebook | `notebooks/kaggle/k03_time_amount/` |
| Predecessor | EXP-002 |
| Feature diff | + `tx_hour`, `tx_dow` (periodic only — no absolute time index, to avoid trend extrapolation); + `amt_log1p`, `amt_cents` (foreign-currency cent signal); + `D{1..15}_norm` except D9 (D − day, converting timedelta counters to time-invariant reference dates; originals kept). All row-local, leak-free by construction. Canonical implementations: `src/features/engineering.py`. Model and params unchanged. |
| Config | LightGBM identical to EXP-001/002 (registered params, seed 42). Row-local block needs no fit/transform boundary. |
| Expected | Holdout Δ +0.005 to +0.015 vs EXP-002; the D-normalization should specifically shrink the CV−LB gap (anti-drift transform) — watch the H4 gap metrics. |
| A (holdout AUC) | *(pending)* |
| B (GroupKFold) | *(pending)* |
| DeLong vs EXP-002 | *(pending)* |
| LB public / private | *(pending — see SUB-004)* |
| Verdict / notes | *(pending)* |
