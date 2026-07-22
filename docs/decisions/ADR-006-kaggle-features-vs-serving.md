# ADR-006 — Which Kaggle Feature Blocks Cross the Serving Boundary

## Status

Accepted

## Context

Phase 8 decomposed the gap between the served model (temporal-holdout
ROC-AUC 0.861) and a Kaggle-competitive score into feature blocks, each
scored by a logged late submission (`docs/kaggle/submission-log.md`):

| Block | Experiment | Private LB | Δ private | Internal Δ (holdout) |
|---|---|---|---|---|
| Baseline (sklearn GB, numeric-only, fillna(0)) | EXP-000 | 0.8749 | — | — |
| LightGBM + native NaN | EXP-001 | 0.8877 | +0.0128 | +0.0510 |
| Categorical encodings (freq + label + email split) | EXP-002 | 0.8968 | +0.0091 | +0.0133 |
| Time / amount / D-normalization | EXP-003 | 0.8998 | +0.0030 | +0.0039 |
| Minimal UID + per-UID aggregates | EXP-004 | 0.9032 | +0.0034 | +0.0004 (n.s.) |
| Full aggregation engine (64 feats, multi-UID) | EXP-006 | 0.9078 | +0.0046 | +0.0122 |
| Temporal-stability selection | EXP-007 | 0.9077 | −0.0001 | −0.0038 |

The Kaggle notebooks are **transductive**: encodings and aggregations are
computed over the train+test union, with the full test cohort available
in memory. The served system is the opposite regime: FastAPI scores **one
transaction per request**, sees no cohort, holds no client history, and
must answer within a request timeout. A feature block crosses the serving
boundary only if it is computable from (a) the single incoming row plus
(b) state frozen at training time and shipped with the model artifact.

Two Phase 8 findings constrain the decision beyond mere feasibility:

1. **Drift decay** — every block's internal ΔAUC overstated its private
   ΔAUC (research.md, finding 1). Serving-side expectations must be set
   from the private-LB column, not the holdout column.
2. **Entity memorisation** — the seen/unseen-UID diagnostic (EXP-007)
   showed holdout AUC 0.9897 on clients seen in training vs 0.8990 on new
   clients. UID-aggregate gains come largely from *recognising known
   clients*; a production scorer meets new clients continuously, which is
   exactly the segment where these features are weakest.

## Decision

Feature set **`v2`** (registered in `src/features/feature_registry.py`,
served after the Phase 9 retrain) contains the blocks that are row-local
or computable from frozen training-time state:

**Crosses the boundary (IN):**

- **Model class + imputation** — LightGBM with native NaN handling
  replaces sklearn GB + `fillna(0)` (ADR-007). Largest single private-LB
  block (+0.0128) and zero serving cost.
- **Categorical encodings** — frequency and label encodings for the
  previously-dropped categoricals, plus the email-domain split. The
  encoding maps are **fit on train only** (already the contract of the
  pure functions in `src/features/engineering.py`), frozen as lookup
  tables inside the model artifact, and applied at scoring time as O(1)
  dictionary lookups. Unseen categories map to the sentinel defined at
  fit time — the same path the Kaggle test set exercised.
- **Time / amount features** — hour, day-of-week, log-amount, decimal
  cents. Pure row-local arithmetic on `TransactionDT` / `TransactionAmt`.
- **D-column normalization** — `D_n − TransactionDT/86400` is row-local
  arithmetic; the leakage argument in `engineering.py` shows it uses no
  future information.

**Does not cross (OUT):**

- **UID key + per-UID aggregates** (EXP-004/EXP-006) — the aggregates
  (per-UID count, mean/std of amount, nunique of categoricals) require
  either the full cohort (transductive, structurally impossible at
  scoring time) or a **client feature store** with expanding-window
  state updated on ingest. Deferred, not rejected — design sketch below.
- **Transductive aggregation engine** (EXP-006) — computed over the
  train+test union by construction; no serving equivalent exists. Its
  gain was also mostly internal (+0.0122 internal vs +0.0046 private),
  i.e. concentrated on seen entities.
- **Temporal-stability feature selection** (EXP-007) — refuted on its
  own terms (no private-LB gain); nothing to port.

### Deferred design sketch — UID feature store (future work, not Phase 9)

If per-client features are later wanted in serving: key a feature store
on the UID (`card1 + addr1 + floor(TransactionDT/86400 − D1)`, computable
row-locally); maintain expanding-window aggregates (count, amount
mean/std, nunique via approximate sketches) updated at ingest, read at
scoring time; cold-start clients yield NaN, which LightGBM handles
natively — mirroring the unseen-client regime measured in EXP-007. The
seen/unseen diagnostic sets the honest expectation: such a store helps
returning clients (~0.99 regime) and does nothing for new ones (~0.90
ceiling).

### Expected served-model performance

Feature set `v2` is, by construction, the EXP-003 configuration. The
honest expectation for the Phase 9 retrain is therefore the EXP-003 row:
internal temporal-holdout AUC ≈ 0.93 (vs 0.861 for v1) and, if the
production traffic drifts like the private test period, a deployed
effective gain nearer +0.025 than the internal delta suggests. Both
numbers enter the README only after the retrain measures them.

## Consequences

**Positive:**
- The served model gains the three cheapest, most robust blocks
  (+0.0249 private-LB equivalent over the baseline) with no new runtime
  infrastructure — every v2 feature is O(1) per request.
- The Kaggle model and the served model now diverge **by documented
  decision** rather than by drift; this ADR is the boundary referenced
  by the project rule "never silently port a Kaggle-only feature".
- Expectations for the retrain are pre-registered here, continuing the
  Phase 8 discipline into Phase 9.

**Negative:**
- The served model deliberately leaves ~+0.008 private-LB equivalent
  (UID blocks) on the table until a feature store exists.
- Encoding lookup tables become part of the model artifact; the
  artifact contract (`src/models/artifacts.py`, ADR-002) must serialize
  and reload them, and the API contract test must cover the
  unseen-category path.
- Two models with different feature sets must be kept distinguishable
  in docs; every published metric states which model it belongs to
  (Phase 9 consistency sweep).
