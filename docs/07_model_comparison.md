# Fraud Detection System – Model Comparison

## Reproducibility

To reproduce the experiments in this phase:

- **Random seeds:** `RANDOM_STATE = 42` for all models and splits.
- **Dataset:** IEEE-CIS Fraud Detection; `train_transaction.csv` and `train_identity.csv` in `data/raw/`. Same merged dataset as in Phase 1 and 2.
- **Preprocessing:** Numeric features only; missing values filled with 0; StandardScaler (fit on train) for logistic regression; same pipeline as `notebooks/model_baseline_v1.ipynb`.
- **Environment:** Python 3.x; scikit-learn, numpy, pandas (versions recorded in project requirements or environment file).

---

## 1. Purpose

This document summarizes the **modeling experiments conducted in Phase 3** (Advanced Modeling & Calibration).

The objective is to compare candidate models under the same **cost-sensitive evaluation framework** and determine which model **minimizes Expected Monetary Loss** on the validation set, while remaining statistically defensible and operationally feasible.

**This phase focuses on comparing model classes rather than performing exhaustive hyperparameter tuning.** The goal is to identify which family of models (logistic regression, Random Forest, gradient boosting) performs best under the defined cost structure; fine-grained tuning can be addressed in later phases.

It connects:

- The cost structure and primary metric in `03_metrics_and_cost_modeling.md`
- The baseline and validation strategy in `05_modeling_strategy.md`
- The statistical diagnostics in `06_statistical_diagnostics.md`

The implementation and numerical results are in:

`notebooks/model_comparison_v1.ipynb`

**Results summary (from notebook run):** Section 9 contains the comparison table. In the executed run, the **Gradient Boosting** model achieved the **lowest Expected Monetary Loss** (251,945) and the highest Expected Loss Reduction (357,989 vs approve-all baseline 609,934). Logistic Regression was second (EML 264,155; T* = 0.42), and Random Forest third (EML 268,935; T* = 0.45). GB’s optimal threshold is low (0.02), leading to higher FPR (0.26) but lower EML. Section 7 reports EML after Platt calibration for RF and GB; in this run calibration did not improve EML (calibrator fit on the same validation set; for production a separate calibration set is recommended). **Selected model for later phases: Gradient Boosting**, with optimal threshold T* = 0.02.

---

## 2. Evaluation Framework

The evaluation methodology is aligned with Phase 1 and the cost document.

- **Temporal validation:** Train on the earliest 80% of transactions by `TransactionDT`, validate on the most recent 20%. No random shuffling; validation is strictly future.
- **Primary metric: Expected Monetary Loss (EML).** For a given threshold \(T\), with binary predictions \(\hat{y}_i(T) = \mathbf{1}(p_i \geq T)\):

  **EML(T) = Σ [ y_i · (1 − ŷ_i(T)) · C_FN,i + (1 − y_i) · ŷ_i(T) · C_FP ]**

  where \(C_{FN,i} \approx \text{TransactionAmt}_i\) and \(C_{FP} = 5\). This is the sum of (i) cost of false negatives (fraud approved) and (ii) cost of false positives (legitimate flagged).
- **Expected Loss Reduction:** Difference between the approve-all baseline loss and the loss at the chosen threshold. Measures financial benefit of the model.
- **Calibration analysis:** Reliability curves (fraction of positives vs mean predicted probability in bins). If needed, Platt scaling or isotonic regression is applied; EML is re-evaluated after calibration.
- **Threshold optimization:** The decision threshold is chosen to **minimize Expected Monetary Loss** on the validation set (cost-sensitive evaluation), not to maximize accuracy or F1.

**Why EML over ROC-AUC or PR-AUC:** Discriminative metrics such as ROC-AUC and PR-AUC measure ranking quality but do not directly encode business cost. A model with higher AUC can still yield higher monetary loss if its probability scale or optimal operating point does not align with the cost asymmetry (\(C_{FN} \gg C_{FP}\)). Financial cost is the primary metric because the business objective is to minimize monetary loss; ROC-AUC and PR-AUC are secondary and used for discrimination and comparison.

---

## 3. Candidate Models

The following models were evaluated:

- **Logistic Regression (LR)** — Baseline from Phase 1; linear model with L2 regularization, `class_weight="balanced"`. Interpretable and stable.
- **Random Forest (RF)** — Ensemble of trees; can capture non-linearities and interactions; moderate complexity with tuned depth and leaf size.
- **Gradient Boosting (GB)** — Sequential ensemble (e.g. sklearn `GradientBoostingClassifier`); strong discrimination potential with conservative hyperparameters to limit overfitting.

These were selected to cover a range of bias–variance trade-offs and to test whether more expressive models yield lower EML under the same preprocessing and validation setup.

---

## 4. Model Configuration

**All models use the same preprocessing pipeline defined in previous phases** (`notebooks/model_baseline_v1.ipynb`) to ensure fair comparison. Feature handling is identical across LR, RF, and GB.

### 4.1 Preprocessing (common)

- **Features:** Numeric columns only from the merged transaction and identity tables; `TransactionID`, `TransactionDT`, and `isFraud` excluded. **Categorical variables are excluded in this phase to maintain consistency with earlier modeling stages.** Future phases may introduce dedicated categorical encoding strategies.
- **Missing values:** Filled with 0.
- **Scaling:** Applied for logistic regression (StandardScaler fit on train). Tree-based models use the same feature matrix without scaling (trees are scale-invariant).

### 4.2 Logistic Regression

- L2 penalty, \(C = 1.0\), `class_weight="balanced"`, solver=lbfgs, max_iter=1000, random_state=42.
- Pipeline: StandardScaler → LogisticRegression.

### 4.3 Random Forest

- n_estimators=100, max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=42.
- Trained on the same numeric feature matrix (no scaling).

### 4.4 Gradient Boosting

- n_estimators=80, max_depth=5, learning_rate=0.1, min_samples_leaf=100, subsample=0.8, random_state=42.
- Conservative settings to reduce overfitting and training time.

Configuration choices aim for reproducibility (fixed seed), handling of class imbalance (balanced weights where supported), and limited overfitting (depth and leaf constraints for trees).

---

## 5. Calibration Analysis

- **Reliability curves** are plotted for each model on the validation set (fraction of positives in bins of predicted probability vs mean predicted probability). A well-calibrated model lies close to the diagonal.
- **Calibration methods:** Where tree-based models show systematic miscalibration, **Platt scaling** (sigmoid calibration) is applied via `CalibratedClassifierCV(method="sigmoid", cv="prefit")`. **In this exploratory analysis the calibrator is fitted on the validation set used for evaluation.** This creates optimistic bias (information leakage) because the same data are used to fit the calibrator and to evaluate EML. **In production systems a separate calibration dataset should be used** (e.g. a held-out time window or a dedicated calibration split).
- **Impact on EML:** Expected Monetary Loss is recomputed after calibration. In the executed run, Platt calibration on the validation set did not improve EML for RF or GB (calibrated losses were slightly higher), so the main comparison and selection use uncalibrated probabilities. Threshold optimization is performed on the probabilities used at decision time (post-calibration only if calibration is adopted).

---

## 6. Threshold Optimization

- For each model, a **grid of thresholds** (e.g. 0.01 to 0.99) is evaluated; Expected Monetary Loss is computed at each threshold.
- The **optimal threshold \(T^*\)** is the one that **minimizes EML** on the validation set.
- **Because \(C_{FN} \gg C_{FP}\)** (missing fraud is much costlier than false alarms), the optimal threshold tends to be **lower** than typical classification thresholds (e.g. 0.5). Lower thresholds flag more transactions, reducing missed fraud at the cost of more false positives; under the defined cost structure this trade-off minimizes total loss.
- A **loss vs threshold** plot is produced for all models; the approve-all baseline loss is shown as a horizontal reference.
- **Operational metrics** at \(T^*\) (e.g. Precision, FPR, confusion counts) are reported in the comparison table. Threshold stability can be checked by repeating the evaluation on different temporal windows or bootstrap samples (see Phase 2 diagnostics).

---

## 7. Model Comparison

The comparison table in `notebooks/model_comparison_v1.ipynb` (Section 9) reports, for each model. **Example from one notebook run:**

| Model | ROC-AUC | PR-AUC | Expected Monetary Loss | Expected Loss Reduction | Optimal Threshold | Precision | FPR |
|-------|---------|--------|------------------------|--------------------------|-------------------|-----------|-----|
| Logistic Regression | 0.825 | 0.176 | 264,155 | 345,779 | 0.42 | 0.102 | 0.235 |
| Random Forest | 0.870 | 0.449 | 268,935 | 340,999 | 0.45 | 0.147 | 0.150 |
| Gradient Boosting | 0.861 | 0.409 | **251,945** | **357,989** | 0.02 | 0.101 | 0.259 |

Approve-all baseline loss (validation): 609,934. GB has the lowest EML and is selected; LR is close second. **Although Random Forest shows stronger discriminative performance (highest ROC-AUC and PR-AUC), its probability estimates produce less favorable decisions under the defined cost structure**, so RF has higher EML at its optimal threshold than both LR and GB. This illustrates that ranking performance and economic performance are not the same when costs are asymmetric.

Column definitions:

| Column | Description |
|--------|-------------|
| Model | Logistic Regression, Random Forest, Gradient Boosting |
| ROC-AUC | Area under ROC curve (discrimination) |
| PR-AUC | Area under precision-recall curve (discrimination under imbalance) |
| Expected Monetary Loss | Loss at optimal threshold on validation set |
| Expected Loss Reduction | Baseline (approve-all) loss minus model loss at \(T^*\) |
| Optimal Threshold | \(T^*\) minimizing EML |
| Precision | Fraction of flagged transactions that are fraud: \(TP/(TP+FP)\) |
| FPR | False positive rate: \(FP/(FP+TN)\) |

The **model with the lowest Expected Monetary Loss** is the primary candidate for deployment. Differences in ROC-AUC or PR-AUC do not directly imply better economic outcome; EML and Expected Loss Reduction do.

---

## 8. Bias–Variance Analysis

- **Logistic regression:** High bias (linear boundary), low variance; easily interpretable (coefficients); may underfit complex fraud patterns.
- **Tree-based models (RF, GB):** Higher flexibility, can capture non-linearities and interactions; risk of overfitting if not regularized. **Tree depth and leaf constraints (e.g. max_depth, min_samples_leaf) were intentionally set to conservative values to avoid overfitting on the training set** and to keep training time and model size manageable.
- **Interpretability vs complexity:** LR supports direct coefficient interpretation; tree ensembles offer feature importance but less direct interpretability. For deployment, the chosen model should balance EML, stability across temporal windows, and operational constraints (latency, model size, maintainability).

---

## 9. Final Model Selection

In the executed notebook run, **Gradient Boosting** achieves the lowest Expected Monetary Loss (251,945) and is selected as the **final model for use in future phases**. Threshold T* = 0.02. Subject to:

- **Stability:** Results should be consistent with the temporal validation strategy; large swings in \(T^*\) or EML across time windows would warrant further checks.
- **Operational feasibility:** Model artifact size, inference latency, and integration with the decision engine (see `04_architecture.md`) must be acceptable.

In this run: **model = Gradient Boosting**, \(T^* = 0.02\). Calibration (Section 7) did not improve EML, so the deployed artifact uses the uncalibrated GB model and T* = 0.02. If in another run calibration were to improve EML, the artifact would include the calibrator.

**Future validation should verify that the optimal threshold remains stable across different temporal windows** (e.g. by re-running the threshold sweep on multiple hold-out periods or by monitoring realized loss over time). Large shifts in \(T^*\) would signal distribution shift or cost-model mismatch and would warrant re-evaluation.

---

## 10. Implications for Deployment

- **Model artifact:** The selected model (and calibrator, if used) is **serialized (e.g. via joblib or pickle) and loaded by the inference service**. Preprocessing (feature list, fillna, scaler for LR) must be identical in production. The trained model will be persisted as a single artifact (or model + calibrator) and deployed to the scoring service as defined in `04_architecture.md`.
- **Threshold configuration:** The decision engine uses the **cost-optimal threshold \(T^*\)** from this phase. Any change in cost assumptions (\(C_{FP}\), \(C_{FN}\)) requires re-running the threshold sweep.
- **Integration with decision engine:** The architecture in `04_architecture.md` describes how the scoring model and threshold feed into the decision logic. Phase 3 outputs (model choice, \(T^*\), EML, Expected Loss Reduction) are the inputs for that integration and for monitoring (e.g. tracking realized loss vs expected).

### Operational Implications

- **Inference latency:** Tree-based models (RF, GB) typically have sub-100 ms inference per transaction for hundreds of features; latency should be validated against the target (e.g. P95 &lt; 200 ms as in `03_metrics_and_cost_modeling.md`).
- **Model size:** Serialized GB or RF artifacts are on the order of megabytes depending on tree count and depth; storage and load time should be acceptable for the deployment environment.
- **Update frequency:** Models and thresholds should be re-trained or re-tuned when cost assumptions change, when significant distribution shift is detected, or on a scheduled basis (e.g. quarterly) using the same evaluation pipeline.

Results from this phase are documented in the notebook; this document provides the rationale and structure for model comparison and deployment decisions.

---

## Summary: What This Modeling Notebook Delivers

- **Reproducible comparison:** Same data, same temporal split (80/20 by `TransactionDT`), same preprocessing and cost (\(C_{FP}=5\), \(C_{FN}=\text{TransactionAmt}\)). All models are evaluated on the same validation set.
- **Primary decision rule:** Choose the model that **minimizes Expected Monetary Loss** at its optimal threshold; ROC-AUC and PR-AUC support interpretation but do not override EML.
- **Calibration:** Reliability curves and (for RF/GB) Platt scaling show whether probabilities are usable as risks; post-calibration EML is computed for reference. If calibration improves EML and is applied in production, the calibrator and \(T^*\) must be stored with the model.
- **Threshold:** Each model has its own cost-optimal \(T^*\); the notebook plots loss vs threshold and reports Precision/FPR at \(T^*\) for operational awareness.
- **Next steps:** The selected model and \(T^*\) feed into the decision engine (see `04_architecture.md`); monitoring should track realized loss vs the expected loss from this phase.
