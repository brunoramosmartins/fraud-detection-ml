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

**Status: drafted, ready to run + publish.**
`notebooks/kaggle/k10_public_dollars_not_auc/k10-fraud-in-dollars.ipynb` (14
cells, pure ASCII, self-contained — inline feature functions, `rglob` data
locate). It is a narrative notebook, not a submission notebook: it produces no
`submission.csv` (nothing to log in `submission-log.md`); it publishes for
notebook-medal (community upvotes). Structure: (1) reframe AUC → dollars, (2)
the EML cost model, (3) a deliberately simple LightGBM on the serving-feasible
feature set, (4) threshold sweep with the fine-grid lesson, (5) monetary
waterfall, (6) the one-block-at-a-time improvement trail + the two honest
findings (drift decay, entity-memorization ceiling). Honest framing throughout:
private LB ~median of 6,381 teams, single model, method over rank.

## Issue #31 — README `## Kaggle Results` section

EXP-000 vs final scores; frozen-LB percentile equivalence phrased honestly
("late submission; would have placed ~top X% of 6,381 teams"); links to the
public notebook and `submission-log.md`. Every number traces to a logged entry.

<!-- fill as work happens -->

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
