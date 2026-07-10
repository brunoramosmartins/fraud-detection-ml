# Research: Closing the Gap Between the Production Baseline and a Kaggle-Competitive Model

## Status

Hypotheses **LOCKED** (pre-registration). The commit that introduces this file
predates every improvement experiment (EXP-001 onward); the git history is the
proof of pre-registration. EXP-000 (baseline submission of the current
production model) is the only experiment allowed to run before this lock.

## Research Question

> **Where exactly does the gap between a well-engineered production baseline
> (temporal-holdout ROC-AUC 0.861) and a Kaggle-competitive score (~0.93+
> private LB) come from — and which parts of the gap-closing feature set
> survive contact with real-time serving constraints?**

The question is answered by decomposing the gap into attributable blocks —
model class (H1), categorical signal (H2), entity aggregation (H3) — under a
pinned validation protocol (`validation-protocol.md`), with the validation
scheme itself treated as a testable design choice (H4). Phase 9 answers the
serving-constraints half via ADR-006.

## Baseline (fixed reference)

| Item | Value |
|---|---|
| Model | sklearn `GradientBoostingClassifier` (80 trees, depth 5, lr 0.1, min_samples_leaf 100, subsample 0.8, seed 42) |
| Features | 380 numeric-only, categoricals dropped, `fillna(0)` |
| Validation | temporal 80/20 split on `TransactionDT` (quantile 0.8) |
| Internal holdout ROC-AUC | 0.861 |
| External score | EXP-000 public/private LB — see `submission-log.md` |

## Pre-Registered Hypotheses

Decision rule for every hypothesis: **supported** requires (a) DeLong test on
the internal temporal holdout significant at α = 0.05 in the predicted
direction AND (b) private LB delta in the same direction meeting the stated
threshold. If (a) and (b) disagree, the verdict is **inconclusive**. If both
agree against the prediction, **rejected**. A null finding is a finding.

| ID | Statement | Test | Threshold | Verdict |
|---|---|---|---|---|
| **H1** | Swapping sklearn GB → LightGBM on the *same* numeric-only feature set, with native NaN handling replacing `fillna(0)`, improves private LB AUC. Isolates model class + imputation; no new signal. | EXP-001 vs EXP-000 | ΔAUC ≥ +0.015 (private LB) | **INCONCLUSIVE** (leaning supported) |
| **H2** | Restoring the dropped categorical features (frequency + label encoding) yields a larger gain than the H1 model swap did. | EXP-002 vs EXP-001 | ΔAUC ≥ +0.020 (private LB) | **INCONCLUSIVE** (comparative claim refuted) |
| **H3** | UID entity aggregation (pseudo-client key `card1 + addr1 + floor(TransactionDT/86400 − D1)` plus per-UID aggregates) is the single largest block, lifting private LB to ≥ 0.93. | EXP-004 vs EXP-003 | ΔAUC ≥ +0.015 and private LB ≥ 0.93 | *(pending)* |
| **H4** | Month-wise GroupKFold CV predicts the private LB better than the single temporal 80/20 split: its mean absolute (CV − private LB) gap is strictly smaller across EXP-001..004. | Gap comparison over EXP-001..004 (both schemes recorded on every run) | mean abs gap (GroupKFold) < mean abs gap (temporal split) | *(pending)* |

Rationale for thresholds: H1/H3 thresholds (+0.015) sit well above the minimum
detectable ΔAUC on a ~118k-row holdout with 3.5% positives (derived in
`exercises/ex01_auc_inference.md`); H2's higher bar (+0.020) encodes the claim
that discarded signal outweighs model class. H3's absolute bar (0.93) is the
frozen-LB medal-zone equivalence from the 2019 competition write-ups.

## Experiment → Hypothesis Map

| Experiment | Feature diff vs predecessor | Hypothesis |
|---|---|---|
| EXP-000 | — (current production pipeline, external anchor) | none |
| EXP-001 | model swap + native NaN, features unchanged | H1 |
| EXP-002 | + categorical encodings | H2 |
| EXP-003 | + time/amount features (hour, dow, log/decimal Amt, D-normalization) | exploratory |
| EXP-004 | + UID key and per-UID aggregates | H3 |
| EXP-005 | consolidation + light tuning + seed averaging | none (headline) |
| (all) | both CV schemes recorded per run | H4 |

## Verdicts

### H1 — INCONCLUSIVE (leaning supported) — 2026-07-10

Evidence (EXP-001 vs EXP-000; SUB-002 vs SUB-001):

- Internal (Scheme A holdout, n = 118,108): ΔAUC **+0.0510** [95% CI +0.0461,
  +0.0558], DeLong z = 20.57, **p = 4.7e-94** — criterion (a) met decisively.
- External: private LB Δ = **+0.0128** (0.8877 vs 0.8749) — positive direction
  but **below** the pre-registered +0.015 threshold; criterion (b) not met.
  Public LB Δ = +0.0238 would have met it.
- Rule applied: (a) and (b) disagree → inconclusive.

Substantive finding: the model-class improvement **decays with temporal
distance** from the training window — holdout (+0.051) → public LB, early
test period (+0.024) → private LB, late test period (+0.013). The registered
threshold implicitly assumed internal gains transfer 1:1 to the private LB;
they do not under drift. Later hypotheses keep their registered thresholds
(changing thresholds after seeing data would defeat pre-registration), but
this decay pattern is itself a pre-specified quantity tracked for H4.

### H2 — INCONCLUSIVE (comparative claim refuted) — 2026-07-10

Evidence (EXP-002 vs EXP-001; SUB-003 vs SUB-002):

- Internal (Scheme A holdout): ΔAUC **+0.0133** [95% CI +0.0109, +0.0156],
  DeLong z = 10.90, **p = 1.1e-27** — criterion (a) met (significant, positive).
- External: private LB Δ = **+0.0091** (0.8968 vs 0.8877) — positive but
  **below** the +0.020 threshold; criterion (b) not met.
- Rule applied: (a) and (b) disagree → inconclusive.

Substantive reading: the categorical block adds genuine, internally-significant
signal (and gave the series-best private LB at the time), but it is **not the
bigger lever** H2 predicted. Categoricals gained *less* than the H1 model swap
on both scales — internal +0.0133 vs +0.0510, private +0.0091 vs +0.0128. The
comparative core of H2 is therefore refuted, even though the strict verdict is
inconclusive. The same drift-decay pattern from H1 recurs: internal +0.0133
shrinks to private +0.0091.

### H4 — running gap data (final verdict after EXP-004)

Per-experiment absolute gap between each CV scheme's estimate and the private
LB (smaller = better predictor of the external score):

| Exp | Scheme A holdout | Scheme B GroupKFold | Private LB | \|A−LB\| | \|B−LB\| |
|---|---|---|---|---|---|
| EXP-001 | 0.9124 | 0.9296 | 0.8877 | **0.0247** | 0.0419 |
| EXP-002 | 0.9257 | 0.9398 | 0.8968 | **0.0289** | 0.0430 |
| EXP-003 | 0.9296 | 0.9428 | 0.8998 | **0.0298** | 0.0430 |
| mean | | | | **0.0278** | 0.0426 |

Trend so far (3/3): the single **temporal split is the closer predictor of the
private LB**, not GroupKFold — the *opposite* of H4's prediction. Mechanism:
the temporal holdout is the most-recent train slice, structurally nearest the
(future) test set; GroupKFold averages over all months including easier early
ones, inflating its estimate. H4 is trending toward **rejected**; final verdict
recorded after EXP-004 completes the EXP-001..004 scope.

## Answer to the Research Question

*(written after H1–H4 verdicts and ADR-006)*
