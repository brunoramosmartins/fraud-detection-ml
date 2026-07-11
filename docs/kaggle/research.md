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
| **H3** | UID entity aggregation (pseudo-client key `card1 + addr1 + floor(TransactionDT/86400 − D1)` plus per-UID aggregates) is the single largest block, lifting private LB to ≥ 0.93. | EXP-004 vs EXP-003 | ΔAUC ≥ +0.015 and private LB ≥ 0.93 | **REJECTED** |
| **H4** | Month-wise GroupKFold CV predicts the private LB better than the single temporal 80/20 split: its mean absolute (CV − private LB) gap is strictly smaller across EXP-001..004. | Gap comparison over EXP-001..004 (both schemes recorded on every run) | mean abs gap (GroupKFold) < mean abs gap (temporal split) | **REJECTED** (opposite held, 4/4) |

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

### H3 — REJECTED — 2026-07-10

Evidence (EXP-004 vs EXP-003; SUB-005 vs SUB-004):

- Internal (Scheme A holdout): ΔAUC **+0.0004** [95% CI −0.0012, +0.0020],
  DeLong z = 0.45, **p = 0.65 — not significant** (indistinguishable from zero).
- External: private LB Δ = **+0.0034** (0.9032 vs 0.8998) — far below the
  +0.015 threshold; private 0.9032 **< 0.93**. Both external conditions fail.
- Rule: both criteria against the prediction → **rejected**.

Substantive reading: the pre-registered UID block (a 4-aggregate key: count,
amount mean/std/ratio + frequency-encoded UID) is **not the dominant lever**
H3 claimed. The 2019 winning solutions' "magic feature" jump came from *rich*
per-UID aggregation (nunique of many categoricals, per-UID D-column statistics,
dozens of engineered columns) — not the entity key alone. The minimal block
here adds series-best LB numbers (private 0.9032, GroupKFold +0.0047) but its
effect on the recent-slice holdout is statistically zero. A richer-aggregation
retry is the natural EXP-005 direction (no new hypothesis — H3 as stated is
settled).

### H4 — REJECTED — 2026-07-10

Per-experiment absolute gap between each CV scheme's estimate and the private
LB (smaller = better predictor of the external score):

| Exp | Scheme A holdout | Scheme B GroupKFold | Private LB | \|A−LB\| | \|B−LB\| |
|---|---|---|---|---|---|
| EXP-001 | 0.9124 | 0.9296 | 0.8877 | **0.0247** | 0.0419 |
| EXP-002 | 0.9257 | 0.9398 | 0.8968 | **0.0289** | 0.0430 |
| EXP-003 | 0.9296 | 0.9428 | 0.8998 | **0.0298** | 0.0430 |
| EXP-004 | 0.9299 | 0.9475 | 0.9032 | **0.0267** | 0.0443 |
| mean | | | | **0.0275** | 0.0431 |

H4 predicted GroupKFold would be the *closer* predictor of the private LB. The
**opposite held in all 4 experiments**: the single temporal split is uniformly
closer (mean gap 0.0275 vs 0.0431). H4 is **rejected** on its pre-registered
metric. Caveat: with 4 paired points a formal sign test (4/4 same direction)
gives p = 0.125, not significant at α=0.05 — but the direction is unanimous and
the magnitude substantial (~0.016 mean gap difference). Mechanism: the temporal
holdout is the most-recent train slice, structurally nearest the future test
set; GroupKFold averages over all months including easier early ones, inflating
its estimate. **Practical takeaway:** for a temporally-drifting deployment, the
"naive" most-recent-slice holdout is the more honest model-selection signal than
the more elaborate month-wise GroupKFold.

## Answer to the Research Question (Phase 8 closed — 2026-07-11)

> *Where does the gap between the production baseline (0.861 internal) and a
> Kaggle-competitive score come from, and which parts survive serving?*

### Where the gap comes from — the private-LB decomposition

Every block was added one at a time and scored on the frozen private leaderboard:

| Step | Block | Private LB | Δ private |
|---|---|---|---|
| EXP-000 | production baseline (sklearn GB, numeric-only) | 0.8749 | — |
| EXP-001 | LightGBM + native NaN (H1) | 0.8877 | +0.0128 |
| EXP-002 | categorical encodings (H2) | 0.8968 | +0.0091 |
| EXP-003 | time / amount / D-normalization | 0.8998 | +0.0030 |
| EXP-004 | minimal UID (H3) | 0.9032 | +0.0034 |
| EXP-006 | full aggregation engine (64 feats) | 0.9078 | +0.0046 |
| EXP-007 | temporal-stability selection | 0.9077 | −0.0001 |

Total realised: **0.8749 → 0.9078 (+0.033)** with a single explainable LightGBM.
No model class won by itself, no single feature block was the "magic"; the gain
is broad and incremental.

### The two findings that outweigh the number

1. **Internal validation systematically overstates private-LB gains under
   temporal drift.** Every block's internal ΔAUC exceeded its private ΔAUC
   (H1 +0.051 internal → +0.013 private; EXP-006 +0.012 → +0.005). Four
   hypotheses, none "supported" — not because the work failed, but because the
   pre-registered thresholds assumed internal gains transfer to the future, and
   they do not. H4 sharpened this: the *simple* temporal holdout predicts the
   private LB better than month-wise GroupKFold (4/4).

2. **The mechanism is entity memorisation, and it is the true ceiling.** The
   seen/unseen-UID diagnostic (EXP-007): holdout AUC **0.9897 on clients seen in
   training vs 0.8990 on new clients**. The private test set is mostly new
   clients, so the private LB ≈ the new-client ceiling (~0.90). This one split
   explains the entire drift-decay pattern. It also redraws the remaining gap to
   the winners' single model (0.9324): it is **not** more aggregation (EXP-006:
   internal-only gain) and **not** feature selection (EXP-007: no private gain) —
   it is **entity resolution / linkability** (more, more-precise UIDs to link
   test transactions to known clients) plus client-level label propagation. That
   is where the winners spent their nights (deanonymising D-columns, building
   transaction chains).

### Which parts survive serving — deferred to Phase 9 / ADR-006

The largest levers (UID aggregations, client-level features) are transductive:
they need the train+test union or a client feature store, and would be
recomputed at scoring time from a client's history. The model class swap
(LightGBM), native-NaN handling, frequency encoding, and row-local
time/amount/D-normalization features are serving-feasible. ADR-006 draws this
boundary and Phase 9 retrains the served model on the feasible subset.

### Verdict summary

| H | Verdict | One-line reason |
|---|---|---|
| H1 | Inconclusive | internal significant, private below +0.015 threshold |
| H2 | Inconclusive | categoricals gained *less* than the model swap (comparative claim refuted) |
| H3 | Rejected | minimal UID: internal Δ not significant, private below thresholds |
| H4 | Rejected | temporal holdout beats GroupKFold as an LB predictor (4/4) |

The investigation's value is the disciplined arc — pre-registration, honest
nulls, a claim (selection) tested and **refuted**, and a diagnostic that
explains the whole pattern — not the leaderboard digit.
