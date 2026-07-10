# Gap Analysis — Our Approach vs the Winning Solutions

## Purpose

Phase 8 adjudicated four pre-registered hypotheses and reached a single-model
private LB of **0.9032** (EXP-004). This document is the Phase 8.6 capstone: a
structured decomposition of what separates our result from the public winning
solutions, written **after** all hypotheses were locked and settled — so it
retro-alters no verdict. It exists to answer Bruno's framing question:

> "Reflect on what the roadmap lacked to reach a performance similar to the
> winning team — recognising that some of that gap may be a lot of energy spent
> refining parameters and testing, i.e. low learning-value per unit effort."

The goal is a **mature development method**, not a rank. So each missing
technique is scored not only by its likely AUC contribution but by its
**insight-per-effort** — and the roadmap only spends effort where the learning
is real. Every "estimated contribution" below is a hypothesis to be *measured*
by a confirmatory experiment (EXP-006+), not an assertion.

## Sourcing & honesty note

The technique inventory is drawn from two public artifacts named by the author:

- Deotte, *XGB Fraud with Magic* (public notebook) — single-model XGBoost.
- FraudSquad, *1st Place Solution* (competition writeup).

Only one number here is directly verified from the live page: the Deotte
notebook's **private score 0.934084** (the "0.9600" in its title is the inflated
public/CV figure). Technique rows are marked **[verify]** where the detail comes
from prior knowledge of these canonical solutions and should be confirmed
against the source — the confirmatory experiments are what turn each into a
measured delta, so no row is treated as fact until an experiment moves it.

## Benchmark ladder (private LB, single-model unless noted)

| Reference | Private LB | Gap to us | What it isolates |
|---|---|---|---|
| **Ours — EXP-000** (sklearn GB, numeric-only) | 0.8749 | — | production baseline |
| **Ours — EXP-004** (single LightGBM, minimal FE) | **0.9032** | — | our best, single model |
| Deotte single XGB "with magic" | **0.9341** (verified) | **+0.031** | *rich* feature engineering, single model |
| FraudSquad 1st place | ~0.9459 | +0.043 | ensemble + client-level post-processing |

Two distinct gaps, and they teach different things:

- **0.9032 → 0.9341 (+0.031): the feature-engineering-richness gap.** Same class
  of model (single GBDT). This is the *learnable, high-value* gap — it is closed
  by engineering, not by parameter search. This is where the roadmap
  under-invested (H3 shipped a 4-aggregate UID; the winners shipped hundreds).
- **0.9341 → 0.9459 (+0.012): the ensemble + post-processing gap.** Diminishing
  returns, higher energy, lower learning-value per unit effort — exactly the
  "refining parameters and stacking" work Bruno flagged as low-value.

**Bruno's instinct is confirmed by the ladder:** the first gap is worth closing
(it is knowledge); the last gap is mostly energy. The roadmap should chase the
+0.031, sample the post-processing idea cheaply, and stop there.

## Technique decomposition

Legend — Have: ✅ full · 🟡 minimal · ❌ absent. Effort: notebook-hours.

| # | Technique (winners) | Have | Our depth | Est. contribution | Insight/effort | Confirmatory exp |
|---|---|---|---|---|---|---|
| 1 | UID client key `card1_addr1_(day−D1)` | ✅ | key built | (in H3) | — | done (EXP-004) |
| 2 | **Rich per-UID aggregations** — mean/std of Amt, C1–C14, D1–D15, and many V; **nunique** of card/addr/email per UID [verify] | 🟡 | 4 aggregates only | **large (+0.01–0.02)** | **high** | **EXP-006** |
| 3 | **Client-level prediction averaging** — group final probs by UID and assign the UID mean (fraud is client-consistent) [verify] | ❌ | none | **large (+0.005–0.015)** | **very high** (cheap, no retrain) | **EXP-007** |
| 4 | Multiple UIDs (card1_addr1, card1_addr1_P_email, …) + aggregations on each [verify] | ❌ | single UID | medium | medium | EXP-006 (bundled) |
| 5 | D-column normalization `D_n − day` | ✅ | done | (in EXP-003) | — | done |
| 6 | Frequency encoding of high-card categoricals | ✅ | ~10 cols | (in EXP-002) | — | done |
| 7 | V-column reduction (drop/PCA correlated V groups) [verify] | ❌ | all V kept as-is | small (noise/speed) | low | optional |
| 8 | Model: XGBoost GPU, depth 12, subsample/colsample 0.4 [verify] | 🟡 | LightGBM, num_leaves 192 | small (model class ≈ settled in H1) | low | not planned |
| 9 | Ensemble XGB + LGB + CatBoost | ❌ | single model (by design) | +0.005–0.01 | low (energy-heavy) | **out of scope** (ADR: single-model stance) |
| 10 | GroupKFold-by-month CV | ✅ | computed | (H4: worse LB predictor than temporal split) | — | done |

## What the roadmap lacked (the honest reflection)

1. **Under-specified the UID block.** H3 pre-registered a *minimal* 4-aggregate
   UID and then, when it failed, we correctly diagnosed "richness, not the key,
   is the magic" — but as an assertion. The roadmap should have scoped the UID
   block as "aggregate a *broad column set* by UID," because the published
   magic was always the breadth of aggregation. Row 2 is the confirmatory fix.
2. **Missed the post-processing lever entirely.** Neither the roadmap nor any
   hypothesis considered client-level probability averaging (row 3). It is the
   highest insight-per-effort item on the board — no retraining, a few lines —
   and it is conceptually the deepest point of the whole competition (fraud is a
   property of the *client*, not the isolated transaction). Its absence is the
   biggest single miss.
3. **Correctly avoided the low-value work.** The roadmap's single-model stance
   (no stacking) and its refusal to grid-search params look *right* in
   hindsight: rows 7–9 are where energy goes to die. The 1st place's last
   +0.012 over a single strong model is not where the learning is.

## Proposed confirmatory experiments (measured, not asserted)

Ordered by insight-per-effort, not by expected AUC. Each is a *replication of a
published technique*, explicitly **not** a new pre-registered hypothesis (H1–H4
are closed). Each gets a registry entry and a logged submission like any other.

- **EXP-006 — Rich UID aggregations.** Aggregate a broad set (C1–C14, D1–D15,
  TransactionAmt, dist1) by UID with mean/std, plus nunique of {card1..6, addr1,
  P_emaildomain, DeviceInfo} per UID; add a second UID (card1_addr1_P_email).
  Measures rows 2 & 4. Predicts: closes much of the +0.031. DeLong vs EXP-004.
- **EXP-007 — Client-level probability post-processing.** Take EXP-006's (or the
  best model's) test predictions, group by UID, assign the per-UID mean. **No
  retraining** — operates on `submission.csv`. Measures row 3. This one is the
  cheap, deep insight; if it moves the private LB substantially, it is the
  headline of the whole gap analysis.
- **(Optional) EXP-008 — single-model consolidation** with light tuning + seed
  averaging on the EXP-006 feature set, as the final Phase 8 number.

The stopping rule stays honest: we chase the feature-engineering gap (EXP-006/
007) because it is *understanding*; we do **not** build the ensemble (row 9)
because that gap is *energy*. If EXP-007's post-processing reproduces the
client-level insight, the investigation has achieved its real goal — a mature,
reproducible method and a precise account of the last mile's cost — regardless
of the final digit.

## Serving-boundary note (feeds ADR-006, Phase 9)

Rows 2–4 are transductive/aggregate features. Their real-time-serving feasibility
(a UID's aggregates need a client feature store; client-level averaging needs the
client's other transactions) is exactly the Kaggle-vs-production divergence
ADR-006 will document. The gap analysis therefore also seeds Phase 9: the
techniques that close the Kaggle gap are precisely the ones that stress the
serving design — a genuine, defensible interview discussion.
