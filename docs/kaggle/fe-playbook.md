# Feature Engineering on Masked Data — A Transferable Playbook

## Why this document exists

The hardest interview question for many data scientists is *"how would you
engineer features here?"* — and it is hardest precisely when the columns are
**anonymized** (`card1`, `V127`, `D15`), because the usual advice ("use domain
knowledge") does not apply. IEEE-CIS is the canonical masked-feature problem, and
the winning solution is the canonical answer. This playbook extracts the
*transferable method* so it can be reused on any masked tabular dataset — and
recited under interview pressure.

**The one-sentence thesis:** when you cannot reason about what a feature *means*,
you reason about what it *is* (its structure) and how it *behaves* (its
statistics). Masked FE is structural and statistical, never semantic.

## The mindset shift

| Semantic FE (needs domain) | Structural/statistical FE (masked-safe) |
|---|---|
| "Amount is high → risky" | "This value repeats across rows → it identifies an entity" |
| "Weekend transactions differ" | "This column drifts with time → normalize it away" |
| "This merchant category is risky" | "This category is rare → frequency-encode it" |
| "New customers are riskier" | "These rows share a group → aggregate within the group" |

You never needed to know that `card1` is a card or `D1` is a tenure counter.
You needed to notice that `card1` is *constant per entity* and `D1` *grows
linearly with time* — both readable from the data alone.

## The playbook — 7 moves, in priority order

Ranked by the value they delivered in IEEE-CIS (top = biggest lever).

### 1. Entity resolution (the highest-value move) — build a UID

**The reasoning, with zero domain knowledge:** a fraud dataset must contain
repeat actors (a card is used many times). If I can *group rows by actor*, I can
compute per-actor behavior. So I hunt for columns that are **constant within an
actor**: an id-like column (`card1`) plus a stable attribute (`addr1`). Then I
need to cancel time: `D1` looks like "days since something" (it increases ~1 per
day), so `day - D1` is **constant per actor** across all their transactions.

```python
uid = card1_addr1 + '_' + floor(TransactionDT/86400 - D1)
```

**How you'd spot this in an interview without the answer key:** plot a candidate
id column's value_counts (is it high-cardinality but repeating?); check which
columns increase monotonically with the time column (those are timedelta
counters); a counter minus the day is a constant per-entity anchor. This is
*detective work on distributions*, not domain knowledge.

### 2. Group aggregations (where the AUC actually comes from)

Once you have a UID, aggregate **many** columns within it. This is what we
under-did (4 features) and the winner did fully (**47** features):

- `mean`, `std` of continuous columns (amount, the D counters, the C counters)
- **`nunique`** of categorical-ish columns per entity — *how many distinct
  emails / devices / values a client touches* is a strong fraud signal and needs
  no domain meaning
- do it at **multiple UID granularities** (card1, card1+addr1, card1+addr1+email)

Interview soundbite: *"a raw feature describes a transaction; the same feature
averaged over the client describes the client — and fraud is a property of the
client."*

### 3. Frequency / count encoding

Replace a category by how often it occurs. Rare values are often risky, and this
needs no meaning — just `value_counts(normalize=True)`. Works for any
high-cardinality masked category.

### 4. Time-invariance transforms (anti-drift)

If a column drifts with absolute time, the model can't extrapolate to the future
test period. Detect drift by correlation with the time column; fix it by
subtracting the time index: `D_norm = D - TransactionDT/86400`. We *did* do this
(EXP-003) — it's the one advanced move the roadmap got right.

### 5. Interaction / combination features

Concatenate two columns into a new category (`card1_addr1`), then encode it. Lets
the model split on combinations a single column can't express. Purely mechanical.

### 6. Statistical feature SELECTION — the master technique for masked data

You can't judge a masked feature's usefulness by reading its name, so judge it by
**behavior over time**. Deotte's *time-consistency test*: train on the first
month, validate on the last month, **one feature at a time**; keep only features
that score AUC > 0.5 on *both*. A feature that helps early but not late is
overfitting to a period, not learning fraud — drop it. (He dropped `C3, M5, card4,
id_07/08/14/21/22-27/30/32/33/34`.)

Related tool: **adversarial validation** — train a classifier to distinguish
train from test; if it succeeds, the features it relies on are drifting and
should be pruned or normalized. Both techniques replace domain-based feature
selection with a *statistical* one. This is the single most reusable idea in the
whole competition.

### 7. Decimal / representation features

Small structural tells: `cents = amount - floor(amount)` (foreign-currency
conversions leave non-round cents). We did this (EXP-003). Generalizes to "extract
the structural sub-parts of a value" (decimals, string suffixes, digit counts).

## How to practice this before an interview

1. **Take any masked dataset** (IEEE-CIS, or the Santander / Porto Seguro Kaggle
   sets) and, *before reading any solution*, run the 7 moves top-to-bottom.
2. **Write the reasoning, not the code first**: for each column, ask "is this an
   id, a counter, a category, or a measurement?" — answerable from dtype +
   cardinality + correlation-with-time alone.
3. **Rehearse the entity-resolution move out loud.** In interviews it is the
   differentiator: most candidates jump to one-hot encoding; few say "first I'd
   look for a grouping key that identifies the entity behind the rows, because
   aggregations within it are usually the strongest signal."
4. **Memorize the selection step.** Saying "I'd validate feature stability across
   time with a time-consistency or adversarial-validation check" signals maturity
   that pure FE listing does not.

## The 60-second interview answer (masked-data FE)

> "With anonymized columns I can't use domain intuition, so I reason
> structurally. First I profile each column by dtype, cardinality, and
> correlation with time to classify it as an id, counter, category, or
> measurement. Then, in order: I try to build an entity key — an id column plus a
> time-invariant transform of a counter — because grouping rows by the underlying
> entity and aggregating within it (mean, std, and especially nunique) is usually
> the biggest lever. I frequency-encode high-cardinality categoricals,
> normalize any time-drifting columns by subtracting the time index, and add a
> few combination features. Finally I select features statistically, not
> semantically: a time-consistency test or adversarial validation to drop
> features that don't generalize to the future period. On IEEE-CIS that exact
> recipe — a `card1+addr1+(day−D1)` client key plus ~47 group aggregations — is
> what separated a 0.90 model from a 0.93 one."

## Trace to this project

Every move above maps to an experiment: entity key (EXP-004 `make_uid`),
aggregations (EXP-004 minimal → EXP-006 full), frequency encoding (EXP-002),
anti-drift (EXP-003), and the missing selection step (added in EXP-006). The
honest through-line — *we built the key but under-built the aggregations, and
skipped statistical selection* — is itself the most credible interview story,
because it shows a diagnosed gap, not a claimed mastery. See
`docs/kaggle/gap-analysis.md`.
