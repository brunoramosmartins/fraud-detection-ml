# Fraud Detection System – Statistical Diagnostics

## 1. Purpose

This document describes the **statistical diagnostics and baselines** used to support rigorous model comparison and interpretation in the fraud detection system.

It connects:

- The data understanding in `02_data_understanding.md`
- The modeling strategy and baseline in `05_modeling_strategy.md`
- The cost structure and primary metric in `03_metrics_and_cost_modeling.md`

The implementation and numerical results are in:

`notebooks/statistical_diagnostics_v1.ipynb`

Phase 1 validated that statistical modeling reduces Expected Monetary Loss. Before advancing to more complex models, we establish:

- The **base rate** of fraud and its uncertainty
- The **stability** of fraud patterns over time
- A **statistical baseline** and testing strategy for comparing models

This prevents overinterpreting improvements that may fall within statistical noise.

---

## 2. Fraud Base Rate Estimation

### 2.1 Overall Rate

The fraud rate is estimated on the full training set (transaction table merged with identity via left join).

- **Point estimate:** proportion of transactions with `isFraud == 1`.
- **Uncertainty:** reported with a 95% confidence interval (Wilson score interval), suitable for binomial proportions and low rates.

Typical range for this dataset: on the order of 1–3% (see notebook for exact values).

### 2.2 By Time Window

Fraud rate is estimated within **time windows** defined by **quantile-based bins** (`pd.qcut(TransactionDT, q=10)`), so each window has comparable observation counts and avoids distortion from uneven transaction density.

- Enables checking whether the rate is stable over time or varies (temporal drift).
- Each window’s rate is reported with a 95% Wilson confidence interval and visualized with a shaded CI band.

### 2.3 By Transaction Amount Bucket

Fraud rate is estimated by **transaction amount buckets** (e.g. quintiles of `TransactionAmt`).

- Reveals whether fraud is more prevalent in certain amount ranges.
- Supports cost-sensitive interpretation (e.g. high-value fraud).

Exact definitions and results are in the diagnostics notebook.

---

## 3. Confidence Intervals

### 3.1 Method

For the fraud rate (binomial proportion), we use the **Wilson score interval**:

- Preferable to the normal approximation for low rates and moderate sample sizes.
- Produces bounds in [0, 1] and has good coverage properties.

A 95% interval is reported for:

- The overall fraud rate
- The fraud rate in each time window
- (Optionally) the fraud rate in each amount bucket

### 3.2 Interpretation

- **Narrow CI:** estimate is precise; differences between models or periods are easier to interpret.
- **Wide CI:** high uncertainty; avoid overclaiming small improvements.

Documenting CIs makes the statistical uncertainty of the base rate explicit for stakeholders and for model comparison.

---

## 4. Class Imbalance Analysis

### 4.1 Quantification

- **Imbalance ratio:** (number of non-fraud) / (number of fraud), i.e. how many legitimate transactions per fraudulent one.
- Typical values for this dataset: on the order of 30:1 to 100:1 (see notebook).

### 4.2 Why accuracy is misleading

With fraud rate on the order of 1–3%, a classifier that always predicts "non-fraud" would achieve high accuracy (~97–99%) but would miss all fraud. Metrics such as ROC-AUC or F1 do not encode the **economic cost** of errors (e.g. C_FN = TransactionAmt, C_FP = 5). The project therefore uses **Expected Monetary Loss** as the primary metric (see `03_metrics_and_cost_modeling.md` and `05_modeling_strategy.md`).

### 4.3 Documentation

The diagnostics notebook reports:

- Proportion and counts per class, imbalance ratio
- A bar chart of class counts
- Fraud rate (with CI) by time window

---

## 5. Distribution of Transaction Amount (TransactionAmt)

Because the cost of fraud depends on **TransactionAmt** (C_FN = TransactionAmt), the diagnostics include:

- **Histogram** of transaction amount for legitimate vs fraudulent transactions separately.
- **Comparison of mean and median** by class.
- **Mann–Whitney U test** (non-parametric) to assess whether fraudulent and legitimate transaction amounts differ systematically.

This strengthens the justification for cost-sensitive evaluation. Exact figures and plots are in the notebook (Section 6).

---

## 6. Temporal Fraud Analysis

### 6.1 Evolution Over Time

- Fraud rate is plotted across time windows (same as in base rate by time).
- A horizontal reference line at the overall rate helps spot periods with higher or lower fraud.

### 6.2 Drift Check (target: fraud rate)

A simple drift check compares:

- **First half** and **second half** of the time span (by `TransactionDT`).

For each half we report:

- Fraud rate and 95% Wilson CI
- **Absolute** and **relative** difference (effect size), so that large samples do not lead to overinterpreting small practical differences.
- A **two-proportion z-test** (H0: equal rates) for statistical significance.

- **Significant p-value:** suggests temporal drift; validation and monitoring should account for it.
- **Non-significant:** rate is relatively stable between the two periods (within sampling variability).

### 6.3 Rolling fraud rate

A **rolling window** (e.g. 10k transactions in time order) is used to plot fraud rate over the timeline. This helps identify gradual trends or regime shifts in fraud prevalence.

This analysis **justifies temporal validation** in the project (see `05_modeling_strategy.md`).

---

## 7. Feature-level temporal drift

Statistics and **distributions** of key features are compared between the **first 30%** and **last 30%** of the time span (by `TransactionDT`), to provide stronger evidence of potential dataset shift.

- **Summary stats:** mean, median, std of `TransactionAmt` (and optionally other numeric features) for early vs late period.
- **Kolmogorov–Smirnov (KS) test** to compare the full distribution of selected features between the two periods.
- **Distribution plots** (e.g. histograms of TransactionAmt) for early vs late.
- **Feature stability ranking:** features are ranked by magnitude of drift (e.g. KS statistic or mean difference), to identify those that change most over time for monitoring and feature engineering.

Interpretation: if KS or summary stats differ, the problem shows initial signs of shift that could affect model performance over time and support temporal validation and monitoring.

---

## 8. Variability of fraud rate by segment

Segmentation is **explicit and reproducible:** **TransactionAmt quantile buckets (Q1–Q5)**. Fraud rate is computed per bucket and visualized (e.g. bar chart of rate by quintile).

- This shows that fraud is **not uniform** and highlights **risk concentration** (which amount bands have higher fraud rate).
- Implications: cost-sensitive and segment-aware evaluation may be needed in later phases; the primary metric (Expected Loss) naturally weights by transaction amount. Optional: fraud rate by **decile** of TransactionAmt to reveal nonlinear relationships.

---

## 9. Statistical Baseline for Model Comparison

### 9.1 Primary Metric

Model comparison is based on **Expected Monetary Loss** (and thus **Expected Loss Reduction** vs approve-all), as defined in `03_metrics_and_cost_modeling.md` and used in `05_modeling_strategy.md`.

- Each model is evaluated on the **same** temporal validation set.
- Thresholds are chosen by **cost minimization** per model (or by a common policy if applicable).
- We compare loss at the chosen threshold(s), not raw AUC or F1.

### 9.2 Uncertainty for Loss

- **Bootstrap is used to estimate the variability of Expected Monetary Loss** — model performance is subject to sampling variability.
- Bootstrap (e.g. 500 resamples with replacement on the validation set) yields:
  - 95% CI for Expected Monetary Loss at \(T^*\)
  - 95% CI for Expected Loss Reduction (vs approve-all)
- When bootstrap is run in modeling notebooks, outputs should include a **histogram of Expected Loss** and its confidence interval.
- If the 95% CI for loss reduction is entirely above 0, we have evidence that the model improves over the baseline.
- Implementation is in the baseline (or comparison) notebook; methodology is documented in the diagnostics notebook.

### 9.3 Secondary Metrics

- **ROC-AUC** and **PR-AUC** are diagnostic.
- For comparing two ROC curves, **DeLong test** or bootstrap can be used; the primary decision remains based on Expected Loss.

### 9.4 Decision Rule

- Prefer the model with **lower** Expected Monetary Loss on the validation set.
- Claim improvement only if the difference is:
  - Meaningful in business terms, and
  - (Optionally) supported by bootstrap CI (e.g. 95% CI for loss reduction excludes 0).

This strategy keeps model comparison aligned with financial impact and guards against overinterpreting small or unstable gains.

---

## 10. Conclusion: three questions

This phase is designed to answer three questions clearly:

1. **What is the fraud rate and its uncertainty?** — Point estimate and 95% Wilson CI; margin of error is documented so that the base rate is a solid reference for model comparison.
2. **How does the rate vary over time and across segments?** — Time windows and amount buckets show non-uniform fraud; temporal validation is justified; optional feature-level drift (first 30% vs last 30%, KS, ranking) indicates whether the problem shows dataset shift.
3. **What are the statistical implications for evaluating models fairly?** — Primary metric is Expected Monetary Loss; bootstrap for uncertainty; only claim improvement when supported by evidence (e.g. 95% CI for loss reduction).

**Explicit modeling implications** (detailed in the notebook and linked to `05_modeling_strategy.md`): justification for temporal validation; justification for cost-sensitive evaluation (EML); implications of imbalance ratio; potential temporal drift and retraining.

These conclusions form the analytical base for the next phase (Advanced Modeling & Calibration).

---

## 11. Outputs and Deliverables


| Output                      | Description                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Base fraud rate             | Point estimate and 95% Wilson CI; explicit print (rate %, total, fraud count); plot of transactions vs frauds; interpretation (rarity, margin of error) |
| Confidence intervals        | Wilson method; CI per time window; shaded band in temporal plot; interpretation                                                              |
| Class imbalance             | Proportion per class, fraud rate (%), imbalance ratio; why accuracy is misleading; why EML                                                 |
| TransactionAmt distribution | Histogram by class; mean/median comparison; Mann–Whitney U test                                                                            |
| Temporal analysis           | Rate by time window (quantile-based); line plot with CI; first vs second half drift with effect size (absolute, relative); z-test; rolling fraud rate (e.g. 10k window) |
| Feature-level drift         | First 30% vs last 30% of timeline; summary stats; KS test; distribution plots (early vs late); feature stability ranking (e.g. by KS)     |
| Rate by segment             | TransactionAmt quantile buckets (Q1–Q5); fraud rate per bucket; risk concentration plot; optional decile analysis                          |
| Statistical baseline        | Primary metric (Expected Loss); bootstrap to estimate variability of Expected Loss; histogram/CI when run in modeling notebooks            |
| Optional                    | Bootstrap distribution of fraud rate (B=500); fraud rate by TransactionAmt decile                                                         |

Numerical results and plots are in `notebooks/statistical_diagnostics_v1.ipynb`. The **summary table** at the end includes: fraud base rate, imbalance ratio, temporal variation (min/max rate), average TransactionAmt (legitimate vs fraud), first/second half rates, and z-test p-value. Expected loss baseline is computed in `model_baseline_v1.ipynb`.

---

## 12. References

- **Notebook:** `notebooks/statistical_diagnostics_v1.ipynb`
- **Data and target:** `02_data_understanding.md`
- **Modeling and validation:** `05_modeling_strategy.md`
- **Cost and metrics:** `03_metrics_and_cost_modeling.md`

