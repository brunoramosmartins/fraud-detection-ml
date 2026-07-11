# Validation & Feature-Selection Playbook (transferable strategies)

## Why this document exists

The feature-*construction* playbook (`fe-playbook.md`) answers "how do I build
features on masked data." This companion answers the question the 2nd-place
solution and our own EXP-006 both forced: **once you have many features, how do
you validate honestly and select the ones that generalize?** These are the
higher-leverage, more transferable skills — they apply to *every* tabular
project, not just fraud, and they are what separates a model that scores well
internally from one that scores well on unseen future data.

The strategies below are distilled from the IEEE-CIS 2nd-place write-up
(sggpls / CPMP / Giba team) and Deotte's solution, then filtered to what is
**general and high-value**, discarding the competition-specific grind
(deanonymization by hand, LB-probed post-processing, 5-person blending).

## The finding that motivates all of this (our EXP-006)

We faithfully replicated the winners' aggregation engine (64 features). Result:

- Internal holdout: **+0.0122** over the minimal model (DeLong p = 5.6e-25) —
  the features carry real signal.
- Private LB: **+0.0046** — only a third of the internal gain.
- **The CV−LB gap widened** (0.0267 → 0.0343): *more features made the
  overfitting-to-the-training-period worse.*

We matched their feature *construction* and still sit 0.025 below their single
model. The missing piece is not more features — it is **feature selection for
temporal stability**. Every strategy below addresses that.

## Strategy 1 — Train/test distribution screening (the highest-value habit)

**What** (this is the plot the 2nd place showed): for each feature, plot its
distribution in **train (red) vs test (blue)** — both the raw value and its
frequency encoding — plus the target mean. Keep the representation that is
*stable across train and test*; discard the one that drifts.

**Why:** high-cardinality ids like `card1` have many values that appear only in
test. Used raw, the model splits on values it will never generalize on → private
LB collapses. The *frequency encoding* of the same column is balanced across
train/test → stable. (In the winners' plot, raw `card1` is wildly train/test-
imbalanced; `log(frequency of card1)` is balanced.)

**How, mechanically:** `train[col].plot.hist` vs `test[col].plot.hist`; or
compare `value_counts` coverage: `len(set(test[col]) - set(train[col])) /
n_unique`. If a feature's values are mostly test-only, drop the raw column and
keep only its frequency/target encoding.

**Applies to us:** directly. We still feed **raw** `card1, card2, ..., addr1`
into the model *and* their frequency encodings. Per this screen, the raw
high-cardinality ids likely hurt our private LB — a concrete, testable fix.

## Strategy 2 — Adversarial validation (the automated version of Strategy 1)

**What:** label train rows 0 and test rows 1, train a classifier to tell them
apart. If it succeeds (AUC ≫ 0.5), train and test differ — and the top features
by importance are the *drifting* ones. Drop or frequency-encode them; re-check.

**Why:** it turns "eyeball every feature" into one model that ranks drift for
you. The features it uses to distinguish train from test are exactly the ones
that will not generalize.

**Applies to us:** yes — a single cheap model that would flag our raw ids
automatically. A natural component of EXP-007.

## Strategy 3 — Forward-in-time CV with a gap (validates our H4)

**What** (CPMP's exact scheme, months numbered 0–6, `|` = train/val split):

```
0     | 2 3 4 5 6
0 1   | 3 4 5 6
0 1 2 | 4 5 6
0 1 2 3 | 5 6
```

Train only on the past, **skip a month** (the gap), validate on the future. Four
folds, each testing genuine long-range forecast.

**Why:** it mimics the real deployment gap between train and the future test set.
Note what it is *not*: plain GroupKFold, which trains on future months to predict
past ones. **This is precisely our H4 result** — we found GroupKFold is a *worse*
private-LB predictor than a simple temporal holdout, because it leaks time. The
winners independently reached the same conclusion: "almost all our HoldOut scores
were highly correlated with public and private scores." Time-respecting
validation is the honest signal.

**Applies to us:** high value. Adopting forward-CV-with-gap would give a CV
number that tracks the private LB — and would have stopped us over-trusting the
GroupKFold estimate (which sat ~0.045 above the private LB every time).

## Strategy 4 — Model-based feature selection (permutation importance, iterative)

**What** (CPMP): frequency-encode all features; with the forward CV, use
**permutation importance** — keep a feature only if permuting it does *not*
improve any fold's predictions; drop the rest; repeat until the list stops
shrinking. This alone got him to 0.942 public with no UID yet.

**Why:** it removes features that add variance without signal — the ones that
inflate internal scores and hurt generalization. It is model-aware (unlike a
univariate filter) and self-terminating.

**Applies to us:** yes — the counterpart to Deotte's time-consistency test, and
the direct fix for EXP-006's widened gap. Core of EXP-007.

## Strategy 5 — Time-consistency feature selection (Deotte's version)

**What:** train on the first period, validate on the last period, one feature at
a time; drop any feature that scores AUC > 0.5 in one period but not the other.
A purely statistical stability filter.

**Why:** a feature useful early but not late is fitting a time window, not the
target. Simpler and cheaper than permutation importance; a good first pass.

**Applies to us:** yes — cheapest selection method to try first in EXP-007.

## Strategy 6 — Segment validation by seen-vs-unseen entity

**What:** split the validation score by whether the entity (card/UID) was
**seen in train** or is **new in the validation/test period**. The winners found
CatBoost scored ~0.999 on *old* cardholders (memorising the leak) but overfit and
was hard to stabilise on *new* users.

**Why:** one global AUC hides the failure mode. If your model is 0.99 on seen
entities and 0.85 on unseen, you are memorising, not generalising — and the test
set's unseen fraction decides your real score. It tells you *where* to invest.

**Applies to us:** high diagnostic value. Splitting our holdout AUC by
seen/unseen UID would likely explain the entire drift-decay pattern in one chart —
strong material for the write-up and interviews.

## Strategy 7 — Distrust anything tuned on the leaderboard

**What** (stated outright by the 2nd place): "more than half of the submissions
were made to calibrate a post-processing procedure … the idea of post-processing
is bad and leads to strong overfit." CPMP's temporal-stacking experiment: "CV
skyrocketed but LB dropped 0.01."

**Why:** techniques whose thresholds are tuned by probing the public LB overfit
to it and can collapse on private. A robust internal CV that *correlates with the
LB* (Strategy 3) is the antidote.

**Applies to us:** it validates our entire methodology — pre-registered
hypotheses, internal DeLong significance required alongside LB direction, no
LB-probing. We were already doing the disciplined thing; this is the winners
confirming why it matters.

## Strategy 8 — Sequence / inter-event aggregations (a feature-construction add-on)

**What** (CPMP's `add_gr`): beyond static per-UID mean/std, compute the **time to
the next transaction** (`groupby(uid).TransactionDT.shift(-1) − TransactionDT`)
and its mean/std/median per UID, plus next/mean/std/median of amount. Velocity
and rhythm of a client's activity.

**Why:** fraud has temporal rhythm (bursts of activity); static aggregates miss
it. Generalises to any entity-event log (sessions, clicks, payments).

**Applies to us:** a natural EXP-006 extension if we want more construction; but
per the EXP-006 finding, **selection (7 strategies above) is the higher-value
next move than more construction.**

## The convergent conclusion

Our EXP-006 result and the 2nd-place write-up point to the same place from two
directions: **we have enough feature construction; what we lack is
validation-and-selection discipline for temporal generalization.** The single
most valuable next experiment is therefore not more features — it is EXP-007:
train/test screening + adversarial validation + model-based selection, evaluated
under forward-CV-with-gap, with the holdout segmented by seen/unseen UID. That is
the mature method the whole project has been building toward, and it is fully
transferable to any future tabular problem with temporal drift.

## Interview one-liner

> "On IEEE-CIS I learned the hard way that beyond a point, adding features widened
> my CV-to-leaderboard gap — the model was overfitting the training period. The
> fix the top teams used, and that I replicated, wasn't more features: it was
> validation that respects time (forward folds with a gap) and feature selection
> for temporal stability (train/test distribution screening, adversarial
> validation, permutation importance). I now treat 'does this feature generalize
> across time?' as a first-class question, not an afterthought."
