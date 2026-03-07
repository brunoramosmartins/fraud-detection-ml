# Fraud Detection System – Modeling Strategy

## 1. Purpose

This document describes the modeling strategy for the fraud detection system, focusing on **Phase 1: baseline modeling and cost-sensitive evaluation**.

It connects the modeling workflow with previously defined system components:

- The system objective and scope in `01_system_scope.md`
- The data understanding in `02_data_understanding.md`
- The cost structure and evaluation metrics in `03_metrics_and_cost_modeling.md`
- The architecture and decision engine in `04_architecture.md`

The implementation and numerical results are provided in:

`notebooks/model_baseline_v1.ipynb`

---

# 2. Phase 1 Objective

Phase 1 aims to:

- Establish a statistical baseline using **logistic regression**, trained and evaluated with **temporal validation**.
- Quantify **Expected Monetary Loss** and **Expected Loss Reduction** relative to an approve-all baseline.
- Select the decision threshold through **cost minimization**, not by accuracy or F1-score.
- Document modeling assumptions, limitations, and next steps for future model iterations.

The goal of this phase is **not model complexity**, but verifying that statistical modeling can produce measurable financial impact.

---

# 3. Validation Strategy

Validation follows a **time-based split** to avoid leakage and to approximate production conditions.

### Split Strategy

- Split variable: `TransactionDT`
- Training set: earliest portion of the timeline (e.g., first 80%)
- Validation set: most recent portion (e.g., last 20%)

### Key Constraints

- **No random shuffling**
- The validation set occurs strictly **after** the training period
- The model is therefore evaluated on **future transactions**

### Rationale

This strategy prevents information leakage and reflects the real operational scenario where models are trained on historical data and applied to new transactions.

This approach is consistent with the temporal risks identified in `02_data_understanding.md`.

---

# 4. Baseline Model Definition

## 4.1 Model and Training

Baseline algorithm:

- **Model:** Logistic Regression
- **Regularization:** L2
- **C:** 1.0
- **Solver:** lbfgs
- **max_iter:** 1000

### Class Imbalance

Fraud is a rare event in the dataset (approximately 1–3% of transactions).

To mitigate the effect of class imbalance during training, the logistic regression model uses:

`class_weight="balanced"`

This reweights the loss function so that errors on the minority (fraud) class receive higher importance during optimization.

This adjusts the optimization objective so that fraud observations receive higher importance during training.

### Reproducibility

The experiment uses:

- Fixed random seed (`RANDOM_STATE = 42`)
- Explicit dataset paths
- Documented constants and configuration

---

## 4.2 Features and Preprocessing

### Feature Set

The baseline uses **numeric features only**, derived from the merged transaction and identity tables.

Exclusions:

- `TransactionID`
- `TransactionDT`
- `isFraud` (target)

This simplification allows the baseline to focus on validating the evaluation framework.

### Missing Values

Missing values are filled with **0** in the baseline implementation.

This choice simplifies the first iteration but may introduce bias for certain variables.

Future versions will consider:

- Median imputation
- Mode imputation
- Missing-value indicator features

### Scaling

Numeric features are standardized using:

`StandardScaler`

The scaler is **fitted on the training set only** and then applied to the validation set.

---

## 4.3 Model Output

The model produces a **fraud probability** for each transaction.

Operational decisions are derived from this probability using a threshold \(T\):

- Approve if \(p < T\)
- Flag if \(p \ge T\)

The optimal threshold is determined through **Expected Monetary Loss minimization**.

---

# 5. Cost Structure and Primary Metric

The cost structure is defined in:

`03_metrics_and_cost_modeling.md`

For the baseline:

- **C_FN (False Negative cost)**  
  Approximated by `TransactionAmount`.

- **C_FP (False Positive cost)**  
  Fixed at **5 monetary units**.

### Primary Metric

Expected Monetary Loss on the validation set.

For a threshold \(T\):

$$
L(T) =
\sum_i
\left[
y_i \mathbf{1}(p_i < T) C_{FN,i}
+
(1 - y_i) \mathbf{1}(p_i \ge T) C_{FP}
\right]
$$

### Expected Loss Reduction

Baseline policy: approve all transactions.

Baseline loss:

Sum of `TransactionAmount` across all fraudulent transactions.

Model savings:

```
Savings = baseline_loss − L(T*)
```

Relative reduction:

```
(baseline_loss − L(T*)) / baseline_loss
```

Threshold \(T^*\) is selected by minimizing \(L(T)\).

---

# 6. Threshold Optimization

Threshold optimization follows these steps:

1. Generate a grid of thresholds (e.g., 0.01–0.99).
2. Compute Expected Monetary Loss for each threshold.
3. Identify the optimal threshold:

```
T* = argmin L(T)
```

A **Loss vs Threshold curve** is produced in the baseline notebook.

Operational metrics reported at \(T^*\):

- Fraud Detection Rate (FDR)
- False Positive Rate (FPR)

In this phase:

- No FPR constraint is enforced
- No ReviewRate constraint is enforced

Future phases will introduce operational constraints and re-optimize thresholds.

---

# 7. Results Summary

Numerical results are produced in:

`notebooks/model_baseline_v1.ipynb`

The notebook reports the following metrics:

| Metric | Description |
|------|------|
| baseline_loss | Expected loss under approve-all policy |
| best_loss | Model loss at optimal threshold |
| T* | Cost-minimizing threshold |
| Expected Loss Reduction (abs) | baseline_loss − best_loss |
| Expected Loss Reduction (rel) | relative reduction vs baseline |
| FDR @ T* | Fraud detection rate |
| FPR @ T* | False positive rate |

Diagnostic metrics also reported:

- ROC-AUC
- PR-AUC

These are **secondary metrics** used for model comparison.

---

# 8. Modeling Assumptions and Limitations

## 8.1 Assumptions

- Cost structure:  
  `C_FP = 5`, `C_FN = TransactionAmount`
- Temporal split:  
  80/20 based on `TransactionDT`
- Feature set:  
  numeric variables only
- Missing values:  
  filled with zero
- Model:  
  logistic regression with balanced class weights

---

## 8.2 Limitations

### Simplified Imputation

Using zero for missing values may introduce bias.

Future improvements:

- median/mode imputation
- missing indicators

### Limited Model Capacity

Logistic regression assumes a linear decision boundary and may underfit complex fraud patterns.

Future phases will evaluate:

- tree-based models
- boosting methods

### Binary Decision Policy

This phase implements a **two-way decision**:

Approve vs Flag.

Future phases will introduce:

Approve / Review / Block policies with review-rate constraints.

### Calibration

Calibration curves are inspected in the notebook.

If miscalibration is detected, methods such as:

- Platt scaling
- Isotonic regression

may be applied.

---

# 9. Next Steps

Future phases will focus on:

### Feature Engineering

- Categorical encoding
- Aggregation features
- Improved missing value handling

### Model Expansion

- Gradient boosting
- Model comparison using Expected Monetary Loss

### Operational Constraints

- ReviewRate limits
- FPR constraints

### Documentation

This document and the baseline notebook should remain synchronized when assumptions or results change.

---

# 10. References

Notebook implementation:

`notebooks/model_baseline_v1.ipynb`

Cost modeling and metrics:

`docs/03_metrics_and_cost_modeling.md`