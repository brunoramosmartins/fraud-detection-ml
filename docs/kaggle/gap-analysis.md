# Gap Analysis — Our Approach vs the Winning Solution (verified)

## Purpose

Phase 8 adjudicated four pre-registered hypotheses and reached a single-model
private LB of **0.9032** (EXP-004). This Phase 8.6 capstone decomposes what
separates that from the public winning solution — written **after** all
hypotheses were locked, so it retro-alters no verdict. It answers the author's
framing question: *what did the roadmap lack, and where is the gap real learning
vs. mere energy?*

The technique inventory below is now **verified against the downloaded notebook**
`xgb-fraud-with-magic-0-9600.ipynb` (Chris Deotte), not memory. One earlier
estimate was wrong and is corrected here (post-processing is small, not large) —
which is exactly why verifying against source mattered.

## Benchmark ladder (private LB, single model unless noted)

| Reference | Private LB | Gap to us | What it isolates |
|---|---|---|---|
| Ours — EXP-000 (sklearn GB, numeric-only) | 0.8749 | — | production baseline |
| **Ours — EXP-004** (LightGBM, UID + 4 aggregates) | **0.9032** | — | our best, single model |
| Deotte XGB_96 (UID + **47** aggregates, pre-postprocess) | **0.9324** | +0.029 | *rich aggregation*, single model |
| Deotte XGB_96_PP (+ client-level post-process) | **0.9341** | +0.031 | post-process, single model |
| FraudSquad 1st place (XGB+LGB+CatBoost ensemble + PP) | ~0.9459 | +0.043 | ensemble |

**Corrected decomposition of the gap (verified from the notebook):**

- **0.9032 → 0.9324 (+0.029): rich group-aggregation.** Same model class. Deotte
  builds **47** aggregation features; we built **4**. This is the whole ballgame
  and it is *engineering knowledge*, not parameters.
- **0.9324 → 0.9341 (+0.0016): client-level post-processing.** Verified from the
  notebook: *"increases its Private LB to 0.9341 from 0.9324 … improvement of
  0.0016."* I previously over-estimated this at +0.005–0.015 — **it is small**.
  Still conceptually deep, but not the lever.
- **0.9341 → 0.9459 (+0.012): ensemble.** Energy, not insight.

**Confirmed conclusion (sharper than before):** the entire learnable gap is the
**breadth of UID aggregation**. Post-processing and ensembling — the things that
*look* like "the secret" — together add only ~+0.014 and cost the most energy.
The author's instinct to not grind params/stacking is right; the one thing worth
replicating in full is the aggregation engine.

## Exactly what they did (verified) vs what we did

### The UID (identical to ours ✅)

```python
uid = card1_addr1 + '_' + floor(day - D1)      # day = TransactionDT/86400
```

Our `make_uid` matches this. The key was never the gap.

### Group aggregations — 47 features (this IS the gap 🟡→❌)

| Winner's aggregation | Function | We have? |
|---|---|---|
| `TransactionAmt, D4, D9, D10, D15` by uid | mean, std | 🟡 only Amt mean/std |
| `C1..C14` (except C3) by uid | mean | ❌ |
| `M1..M9` by uid | mean | ❌ |
| `P_emaildomain, dist1, DT_M, id_02, cents` by uid | **nunique** | ❌ (no nunique at all) |
| `C14` by uid | std | ❌ |
| `C13, V314` by uid | nunique | ❌ |
| `V127,V136,V309,V307,V320` by uid | nunique | ❌ |
| **also at coarser UIDs** `card1`, `card1_addr1`, `card1_addr1_P_emaildomain`: `TransactionAmt, D9, D11` | mean, std | ❌ (single UID only) |
| `outsider15 = (|D1-D15|>3)` | interaction | ❌ |

Two things we entirely missed: **nunique aggregation** (`encode_AG2` — how many
distinct emails/devices/etc. a client touches, a powerful fraud signal) and
**multi-granularity UIDs** (aggregating at card1, card1_addr1, and
card1_addr1_email as well as the D1-based key).

### Time-Consistency feature selection (we don't do this — and it's the answer to "how to do FE on masked data")

Deotte trains a model on the **first month**, validates on the **last month**,
**one feature at a time**, and drops any feature whose AUC isn't consistently
> 0.5 across time (he removed `C3, M5, id_08, id_33, card4, id_07, id_14,
id_21, id_30, id_32, id_34, id_22..27`). This is a *purely statistical* feature
filter — no domain meaning needed. See `docs/kaggle/fe-playbook.md` for why this
is the master technique for masked data.

### Post-process (verified small: +0.0016)

Group final predictions by a *precise* UID (Konstantin's cleaned UIDs) and
replace every transaction's prediction with the client's **mean prediction,
including train `isFraud` labels**. Rationale: all transactions of one client
share the same label. Cheap, conceptually deep, but only +0.0016 private here.

## What the roadmap lacked (honest reflection)

1. **Under-scoped the UID block to 4 aggregates.** H3 tested "does a UID help"
   with a minimal block; the published magic was always *47 aggregations across
   multiple UID granularities, including nunique*. The diagnosis after H3's
   rejection ("richness, not the key") is now **verified**: the key is identical;
   the breadth is the difference. EXP-006 replicates the full engine.
2. **No statistical feature-selection step.** We kept all columns; Deotte prunes
   with a time-consistency test. This is the single most transferable technique
   for masked data and we skipped it. EXP-006 adds it.
3. **No nunique aggregations, single UID granularity.** Two concrete, cheap
   additions with real signal.
4. **Correctly avoided the low-value work.** Ensembling (+0.012, high energy) and
   post-processing (+0.0016) are *not* where the learning is — the roadmap's
   single-model stance holds up.

## Confirmatory experiments (replications of verified technique — not new hypotheses)

- **EXP-006 — Full aggregation engine + time-consistency selection.** Replicate
  the verified feature set: multi-UID (card1, card1_addr1, card1_addr1_email,
  D1-based), mean/std of Amt+D-cols+C-cols+M-cols, nunique of email/device/id/V
  cols, `outsider15`, then time-consistency pruning. DeLong vs EXP-004. Predicts:
  closes most of the +0.029. **This is the one experiment that matters.**
- **EXP-007 — Client-level post-processing.** Group EXP-006's test predictions by
  UID, assign the per-UID mean (incl. train labels). No retraining. Now scoped as
  a **small** expected gain (~+0.002), run for completeness and the conceptual
  point, not for the number.
- **Not planned: ensemble.** ADR single-model stance; the +0.012 is energy.

## Serving-boundary note (feeds ADR-006, Phase 9)

The aggregation engine and post-processing are transductive/client-level: in
real-time serving a UID's aggregates need a client feature store and the
post-process needs the client's history. This is precisely the Kaggle-vs-
production divergence ADR-006 documents — the features that close the Kaggle gap
are the ones that stress the serving design.
