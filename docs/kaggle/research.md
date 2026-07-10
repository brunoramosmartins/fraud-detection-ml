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
| **H1** | Swapping sklearn GB → LightGBM on the *same* numeric-only feature set, with native NaN handling replacing `fillna(0)`, improves private LB AUC. Isolates model class + imputation; no new signal. | EXP-001 vs EXP-000 | ΔAUC ≥ +0.015 (private LB) | *(pending)* |
| **H2** | Restoring the dropped categorical features (frequency + label encoding) yields a larger gain than the H1 model swap did. | EXP-002 vs EXP-001 | ΔAUC ≥ +0.020 (private LB) | *(pending)* |
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

*(filled as experiments complete; each verdict cites the DeLong p-value, the
ΔAUC with CI, and the submission-log entries involved)*

## Answer to the Research Question

*(written after H1–H4 verdicts and ADR-006)*
