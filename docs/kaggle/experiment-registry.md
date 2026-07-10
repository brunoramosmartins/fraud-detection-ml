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
| A (holdout AUC) | *(pending)* |
| B (GroupKFold) | *(pending)* |
| DeLong | — (no predecessor) |
| LB public / private | *(pending — see SUB-001)* |
| Verdict / notes | *(pending)* |

---

*(EXP-001 — H1: LightGBM on identical numeric features — registered here
before its notebook runs)*
