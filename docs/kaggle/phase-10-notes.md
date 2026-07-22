# Phase 10 — Public Kaggle Presence & Interview Narrative

**Objective:** convert the Phase 8–9 trail into public, citable artifacts. In
2026 the competition medals are closed, but notebook medals (community upvotes)
are not — the public notebook is the "classifiable" piece.

**Dates:** opened 2026-07-22 (well ahead of the roadmap's self-imposed
31 Aug – 13 Sep window; no external deadline exists).

**Ends with:** a public Kaggle notebook, a `## Kaggle Results` section in the
README with honest frozen-LB percentile phrasing, and every published Kaggle
claim traceable to `submission-log.md`.

---

## Issue #30 — Public Kaggle notebook: "Fraud Detection in Dollars, Not AUC"

The differentiated angle: the cost-sensitive / EML framing (threshold as a
business decision, monetary waterfall) that almost no notebook in this
competition covers, with the 0.8749 → 0.9078 improvement trail as supporting
material.

**Status: PUBLISHED (2026-07-22).**
https://www.kaggle.com/code/brunoramosmartins/fraud-detection-in-dollars-not-auc

`notebooks/kaggle/k10_public_dollars_not_auc/k10-fraud-in-dollars.ipynb` (14
cells, pure ASCII, self-contained — inline feature functions, `rglob` data
locate). It is a narrative notebook, not a submission notebook: it produces no
`submission.csv`; it publishes for notebook-medal (community upvotes).
Structure: (1) reframe AUC → dollars, (2) the EML cost model, (3) a deliberately
simple LightGBM on the serving-feasible feature set, (4) threshold sweep with
the fine-grid lesson, (5) monetary waterfall, (6) the one-block-at-a-time
improvement trail + the two honest findings (drift decay, entity-memorization
ceiling). Honest framing throughout: private LB ~median of 6,381 teams, single
model, method over rank.

Kaggle run (v337192876) reproduced cleanly end to end: holdout ROC-AUC 0.9301,
cost-optimal threshold 0.0045, EML reduction 71.6% ($436,804), recall 86.0%,
precision 16.4% vs 3.5% random — consistent with the served v2 model.

Two build bugs fixed along the way (see failed attempts): notebook cell `source`
must be a single string, not a list of newline-less lines (papermill joined them
into one broken line); and cells need an `id` field to avoid nbformat warnings.

## Issue #31 — README `## Kaggle Results` section

**Status: done.** Reworked the README "Kaggle Results (Phases 8–10)" section:
full EXP-000..007 ladder (public + private LB, all traced to `submission-log.md`),
a Kaggle-notebook badge, the public-notebook link, and honest placement phrasing
— private 0.9078 sits **around the median** of the 6,381 teams, NOT "top X%"
(the roadmap's original phrasing assumed a stronger score; honesty required the
correction). Method-over-rank framing kept throughout.

## Issue #32 — Interview narrative *(reframed — see decision below)*

The roadmap targeted a committed `DEMO.md` interview narrative. As of Phase 9,
`DEMO.md` was un-tracked and kept as personal, gitignored study notes (author's
preference — the portfolio repo should not carry interview-rehearsal material).
The public-facing narrative therefore folds into #31 (README Kaggle Results) and
#30 (the public notebook); `DEMO.md` remains a private local file the author
maintains for their own prep. Confirm this reframing before closing.

<!-- fill as work happens -->

## Issue #33 — LinkedIn post / article *(optional, discretionary)*

<!-- fill as work happens -->

---

## Lessons Learned

<!-- Author to complete in first person at phase close — interview/writeup
     material, in your own words. -->

## Failed Attempts

<!-- Author to complete in first person at phase close. -->
