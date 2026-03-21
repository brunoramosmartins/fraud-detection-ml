# Hyperparameter Guide — Gradient Boosting Configuration

This document explains every parameter in `configs/model_gb_v1.yml`, the
reasoning behind each chosen value, and the bias-variance implications of
changing it.

The model is `sklearn.ensemble.GradientBoostingClassifier`. The configuration
applies to the `gb` key under `models:`.

---

## Configuration File

```yaml
version: "v1"
feature_set: "v1"
split_quantile: 0.8

models:
  gb:
    n_estimators: 80
    max_depth: 5
    learning_rate: 0.1
    min_samples_leaf: 100
    subsample: 0.8
    random_state: 42
```

---

## Pipeline Parameters

### `split_quantile: 0.8`

Controls the temporal train/validation split point. The 80th quantile of
`TransactionDT` becomes the cutoff: everything before trains the model,
everything after is held out for evaluation.

**Why 0.8:** this reserves approximately 20% of the dataset — the most
recent 20% — for validation. This reflects realistic deployment conditions
where the model is evaluated on future data. A larger quantile (e.g., 0.9)
would give more training data but a smaller validation window, reducing
confidence in the metric estimates. A smaller quantile (e.g., 0.7) gives
a larger validation window but less training data and risks underestimating
performance if the model has not seen enough examples.

**What changes if modified:** moving to 0.7 would likely improve validation
metric stability but could degrade training performance for models that
benefit from more data (like GB). Moving to 0.9 risks overfitting to the
training window with fewer validation examples to catch it.

---

## Model Hyperparameters

### `n_estimators: 80`

The number of boosting rounds — how many individual decision trees are
chained sequentially. Each tree corrects the residual errors of all
previous trees.

**Bias-variance implication:**
- Too few estimators → high bias (underfitting). The model hasn't iterated
  enough to approximate the true decision boundary.
- Too many estimators → overfitting risk if `learning_rate` is high or
  regularization is weak. However, GB is relatively robust to over-estimating
  because later trees have smaller residuals to correct.

**Why 80:** a balance between training speed and performance. The sweet spot
for this dataset size (~470k training rows) is in the range 50–200. At 80,
training completes in under 15 minutes on a standard laptop. Increasing to
200 would improve performance marginally at ~2.5× the training time.

**Reasonable range to explore:** 50–300. Use early stopping (not implemented
here) to find the optimal n_estimators without manual tuning.

---

### `max_depth: 5`

The maximum depth of each individual decision tree. Controls the complexity
of each base learner: a tree of depth 5 can capture up to 5-way feature
interactions.

**Bias-variance implication:**
- Low depth (1–2, "stumps") → each tree is a weak learner; many estimators
  are needed to compensate; training is fast but may underfit complex patterns.
- High depth (7–10) → each tree captures complex interactions; fewer
  estimators needed; higher risk of overfitting on training data.

**Why 5:** depth 5 is a standard starting point for tabular fraud data with
mixed feature types (count-based, time-delta, anonymized). It allows the
model to learn moderately complex interaction patterns (e.g., "card5 is
unusual AND V-feature suggests bot behavior AND transaction amount is high")
without memorizing individual training examples.

**What to watch:** if validation EML significantly exceeds training EML,
`max_depth` is likely too high and should be reduced to 3 or 4.

---

### `learning_rate: 0.1`

The step size applied to each tree's contribution. Each tree's prediction
is multiplied by `learning_rate` before adding to the ensemble.

**Bias-variance implication:**
- High learning rate (0.5–1.0) → the ensemble converges quickly but may
  overshoot the optimal boundary. Requires fewer `n_estimators` to converge.
- Low learning rate (0.01–0.05) → the ensemble moves slowly; requires many
  more `n_estimators` to converge, but typically achieves better generalization.

**The learning_rate × n_estimators trade-off:** these two parameters are
coupled. Halving `learning_rate` to 0.05 and doubling `n_estimators` to 160
often improves performance with no change to training cost per epoch. This
is a well-known tuning heuristic.

**Why 0.1:** a reasonable default that converges reliably at 80 estimators
without requiring a grid search. For a competition or production-critical
model, the correct approach is to reduce `learning_rate` to 0.05 or 0.01
and increase `n_estimators` accordingly with early stopping.

---

### `min_samples_leaf: 100`

The minimum number of training samples required to be at a leaf node. A
split is only accepted if both resulting leaves contain at least 100 samples.

**Bias-variance implication:**
- Small value (1–5) → trees can create leaf nodes from very few examples;
  high variance; the model memorizes training noise.
- Large value (100–500) → each leaf represents a reliable pattern seen in
  many examples; lower variance; the model generalizes better.

**Why 100:** with ~470k training rows and ~3.5% fraud rate (~16k fraud
examples), a leaf minimum of 100 means each decision node must be supported
by at least 100 transactions. This prevents the model from learning rules
that apply to 3 or 4 transactions (likely noise) and encourages generalizable
patterns. This is the primary regularization mechanism in this configuration
— more important than `max_depth` alone.

**What to watch:** if the fraud base rate shifts significantly (e.g., a new
fraud campaign), reducing `min_samples_leaf` may be appropriate to allow
the model to learn from fewer examples of the new pattern.

---

### `subsample: 0.8`

The fraction of training samples used to fit each individual tree. Each
boosting round draws a random 80% of the training data (without replacement).

**Bias-variance implication:**
- `subsample=1.0` → each tree sees all training data; no stochastic variance
  reduction; can lead to overfitting on highly correlated trees.
- `subsample < 1.0` → introduces randomness (Stochastic Gradient Boosting);
  each tree is trained on a different subset; reduces correlation between
  trees; improves generalization and acts as an implicit regularizer.

**Why 0.8:** the 20% excluded per round introduces enough variance to
improve generalization without discarding too much data per tree. A value
below 0.5 would introduce too much noise and destabilize convergence.

**Connection to Random Forests:** subsampling in GB plays a similar role
to the random feature subsampling in Random Forests — it decorrelates the
base learners and reduces ensemble variance.

---

### `random_state: 42`

Seeds the random number generator for subsample selection and tree building.

**Why 42:** reproducibility. Every training run with the same data and
config produces the identical model artifact. This is a requirement, not
a preference — it ensures that `artifacts/runs/` metadata can be trusted
to describe the exact model in `artifacts/models/`.

---

## What Was Not Tuned

Several parameters were left at `sklearn` defaults:

| Parameter | Default | Why not tuned |
|---|---|---|
| `max_features` | 1.0 (all features) | No indication that feature subsampling helps on this dataset; adds search cost |
| `loss` | `"log_loss"` | Correct for binary classification with probability output |
| `criterion` | `"friedman_mse"` | Standard for boosting; no reason to deviate |
| `min_samples_split` | 2 | Superseded by `min_samples_leaf=100` as the primary regularizer |
| `min_weight_fraction_leaf` | 0.0 | Irrelevant when samples are not weighted |

No cross-validation hyperparameter search was performed. The rationale:
temporal validation makes grid search non-trivial — a proper temporal
CV would require multiple non-overlapping folds in time, each with a
different training window. For a portfolio-scale project, a single
temporal split with a manually chosen configuration is a reasonable
trade-off. The configuration is informed by standard GB tuning heuristics
rather than guessed.

---

## Sensitivity Summary

Ranked by expected impact on Expected Monetary Loss (qualitative):

| Parameter | Impact on EML | Direction |
|---|---|---|
| `learning_rate` × `n_estimators` | High | Lower LR + more estimators → lower EML |
| `min_samples_leaf` | High | Higher value → lower variance → better EML on val |
| `max_depth` | Medium | Reduce to 3–4 if overfitting; increase to 6 if underfitting |
| `subsample` | Medium | Values 0.7–0.9 are stable; below 0.5 degrades convergence |
| `n_estimators` alone | Low | Diminishing returns above 150 at current LR |
