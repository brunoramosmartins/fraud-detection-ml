## Fraud Detection System – Metrics and Cost Modeling

### 1. Purpose

This document formalizes the cost structure, evaluation metrics, and threshold optimization strategy of the fraud detection system.

The system is treated as a cost-sensitive decision engine, not a generic binary classifier.

This document connects:

- The business objective defined in `01_system_scope.md`
- The data constraints described in `02_data_understanding.md`
- The architecture and decision engine in `04_architecture.md`

The primary objective of the system is:

Minimize expected monetary loss under operational constraints.

---

### 2. Decision Outcomes

At transaction level, the system solves a binary classification problem.

- True Positive (TP) – Fraud correctly flagged.
- False Positive (FP) – Legitimate transaction incorrectly flagged.
- True Negative (TN) – Legitimate transaction correctly approved.
- False Negative (FN) – Fraudulent transaction incorrectly approved.

Only FP and FN generate direct economic or operational cost.

---

### 3. Cost Structure

Fraud detection is inherently asymmetric. Missing fraud is typically far more expensive than incorrectly blocking a legitimate transaction.

#### 3.1 Cost of False Negative (C_FN)

When fraud is approved:

$$
C_{FN,i} = \text{TransactionAmount}_i + \text{RecoveryCost}_i
$$

In practice this may include:

- Chargeback amount  
- Operational investigation  
- Reputational damage  

For modeling purposes, we adopt a conservative approximation:

$$
C_{FN,i} \approx \text{TransactionAmount}_i
$$

This keeps the model simple while preserving transaction-level cost sensitivity.

---

#### 3.2 Cost of False Positive (C_FP)

When a legitimate transaction is blocked or sent to review:

$$
C_{FP} = \text{SupportCost} + \text{FrictionCost} + \text{ChurnRiskCost}
$$

For the baseline system, we assume a fixed operational cost:

$$
C_{FP} = k
$$

Working assumption:

- $C_{FP} = 5$ monetary units  
- $C_{FN,i} = \text{TransactionAmount}_i$

Thus:

$$
C_{FN,i} \gg C_{FP}
$$

This asymmetry must drive model evaluation and threshold selection.

---

### 4. Expected Monetary Loss

Let:

- $y_i \in \{0,1\}$ – true label  
- $\hat{y}_i \in \{0,1\}$ – model decision  
- $p_i$ – predicted fraud probability  

The total loss over a dataset is:

$$
L = \sum_i \left[
y_i (1 - \hat{y}_i) C_{FN,i}
+
(1 - y_i) \hat{y}_i C_{FP}
\right]
$$

Interpretation:

- Fraud approved → incurs $C_{FN,i}$
- Legitimate flagged → incurs $C_{FP}$

This is the primary system objective.

Accuracy, F1, or AUC do not encode economic impact.

---

### 5. Threshold-Based Loss

Given a threshold $T$ applied to predicted probability:

- Approve if $p_i < T$
- Flag if $p_i \ge T$

Loss as a function of threshold:

$$
L(T) = \sum_i \left[
y_i \mathbf{1}(p_i < T) C_{FN,i}
+
(1 - y_i) \mathbf{1}(p_i \ge T) C_{FP}
\right]
$$

Optimal threshold:

$$
T^* = \arg\min_T L(T)
$$

Thresholds are selected to minimize Expected Monetary Loss, not maximize F1.

---

### 6. Secondary Technical Metrics

These metrics are diagnostic and used for model comparison.

#### 6.1 Discrimination Metrics

- ROC-AUC  
- PR-AUC (preferred under class imbalance)  
- Recall  
- Precision  
- F1-score  

Due to low fraud rate (~1–3%), PR-AUC and Recall at operational precision levels are more informative than ROC-AUC.

---

#### 6.2 Operating-Point Metrics

For selected thresholds:

- Recall @ Fixed Precision  
- Precision @ Fixed Recall  
- False Positive Rate (FPR)  
- Fraud Detection Rate (FDR)

Where:

$$
FDR = \frac{TP}{TP + FN}
$$

$$
FPR = \frac{FP}{FP + TN}
$$

These are reported alongside Expected Loss.

---

### 7. Operational Metrics

Fraud detection systems must satisfy production constraints.

#### 7.1 Latency

Real-time scoring must satisfy:

- P95 latency < 200 ms

Even in offline simulation, this constraint is documented to preserve architectural realism.

---

#### 7.2 Review Rate

If a three-way decision is used (approve / review / block):

$$
\text{ReviewRate} =
\frac{\text{Review Decisions}}{\text{Total Transactions}}
$$

Operational assumption:

- ReviewRate < 5%

Thresholds must satisfy this constraint.

---

### 8. Business KPIs

#### 8.1 Baseline Loss

Baseline policy: approve all transactions.

$$
L_{baseline} =
\sum_{\text{fraud } i} \text{TransactionAmount}_i
$$

#### 8.2 Model Loss

$$
L_{model} = L(T)
$$

#### 8.3 Savings

$$
\text{Savings} =
L_{baseline} - L_{model}
$$

This directly quantifies business value.

---

### 9. Evaluation Strategy

Given temporal drift:

- No random splits.
- Use temporal validation.

Procedure:

1. Train on past window.
2. Validate on future window.
3. Compute predictions.
4. For a grid of thresholds:
   - Compute $L(T)$
   - Compute FDR, FPR
   - Compute PR-AUC
   - Compute ReviewRate

This simulates real deployment.

---

### 10. Three-Way Decision Extension

For approve / review / block decisions:

Two thresholds are defined:

- $T_1 < T_2$

Decision policy:

- $p < T_1$ → Approve  
- $T_1 \le p < T_2$ → Review  
- $p \ge T_2$ → Block  

Different effective costs can be assigned to review vs block, allowing more granular optimization.

---

### 11. Assumptions

- Fraud rate: 1–3%
- $C_{FN,i} \approx \text{TransactionAmount}_i$
- $C_{FP} = 5$
- ReviewRate < 5%
- P95 latency < 200 ms

All parameters should be configurable.

---

### 12. Strategic Positioning

This framework reframes the project from:

Binary classification on an imbalanced dataset

to:

Cost-sensitive fraud decision system with explicit economic optimization and operational constraints

This is consistent with production-grade ML engineering practice.