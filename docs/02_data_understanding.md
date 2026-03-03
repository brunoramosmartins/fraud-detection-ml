# Fraud Detection System – Data Understanding

## 1. Objective

The purpose of this document is to evaluate whether the IEEE-CIS fraud dataset supports building a production-like antifraud system for card-not-present (CNP) e-commerce transactions.

This analysis focuses on:

- Data structure and semantics
- Data quality and limitations
- Fraud distribution
- Leakage risks
- Implications for system architecture

---

## 2. Dataset Overview

The dataset comes from Vesta Corporation and represents real-world card-not-present e-commerce transactions.

It consists of two main tables:

- **Transaction table** – transaction-level features
- **Identity table** – device and identity-related features

Both tables are joined by `TransactionID`.

The target variable is:

- `isFraud` – binary label indicating fraudulent transaction.

The dataset contains hundreds of anonymized features, including device, payment, and behavioral information.

---

## 3. Data Model

### 3.1 Tables and Relationships

The dataset is composed of two relational tables joined by `TransactionID`.

Not all transactions have identity information.

This implies partial identity coverage.

### 3.2 Identity Coverage

Identity data is missing for a significant portion of transactions.

Implications:

- Models relying heavily on identity features may be biased.
- The system must work when identity data is unavailable.
- Feature engineering must consider sparse identity signals.

---

## 4. Feature Taxonomy

Features were grouped into the following categories:

### Payment Metadata
Examples: `card1–card6`, `ProductCD`, transaction amount.

These represent card and purchase characteristics.

### Device Information
Examples: `DeviceType`, `DeviceInfo`.

Used for device fingerprinting.

### Identity Features
Examples: `id_12–id_38`.

Represent anonymized identity signals.

### Temporal Features
Example: `TransactionDT`.

Represents a time delta, not an absolute timestamp.

### Engineered Anonymized Features
Examples: `V*`, `D*`.

These likely represent aggregated behavioral features.

Implications:

- High-cardinality categorical features require careful encoding.
- Missing values are expected.
- Some anonymized features may introduce leakage risk.

---

## 5. Data Quality Assessment

Key observations:

- Many features have high missing rates.
- Identity table has incomplete coverage.
- Several categorical variables have rare categories.
- Outliers exist in transaction amounts.

Implications:

- Models must handle missing values explicitly.
- Rare categories require robust encoding.
- Feature normalization may be required.

No major duplicate transactions were detected.

---

## 6. Target Variable Analysis

The fraud rate is low (~1–3%), indicating strong class imbalance.

Fraud distribution varies across product categories and transaction segments.

Implications:

- Accuracy is not a meaningful metric.
- Cost-sensitive modeling is required.
- Threshold optimization must be used.

---

## 7. Temporal Structure

`TransactionDT` represents a time delta from a reference point.

Fraud patterns change over time.

Implications:

- Random train/test split may introduce leakage.
- Time-based cross-validation is required.
- Model drift is likely in production.

---

## 8. Identity Coverage Impact

Transactions with identity information show different fraud patterns compared to those without identity data.

Implications:

- Identity features may introduce bias.
- Models must perform well without identity.
- Separate evaluation with and without identity is recommended.

---

## 9. Potential Data Leakage

Possible leakage sources include:

- Engineered anonymized features encoding future behavior.
- Improper random splits mixing time periods.
- Identity features derived from post-transaction data.

Mitigation strategies:

- Use temporal cross-validation.
- Perform feature ablation tests.
- Evaluate models on holdout future periods.

---

## 10. Dataset Limitations

The dataset has several limitations:

- Features are anonymized.
- No merchant information.
- No customer network data.
- Fraud labels may be delayed.
- Dataset may not represent full production diversity.

Implications:

- Some fraud patterns cannot be modeled.
- Graph-based fraud detection is not possible.
- Model performance may differ in real deployment.

---

## 11. Implications for System Architecture

The data analysis suggests the following requirements:

- Real-time feature availability is feasible.
- Models must handle missing identity data.
- Continuous monitoring is required due to drift.
- Logging must store predictions and features.
- Threshold optimization is required to minimize expected loss.

These requirements align with the architecture defined in `04_architecture.md`.

---

## 12. Conclusion

The IEEE-CIS dataset is suitable for building a realistic fraud detection system prototype.

However, careful handling is required for:

- Class imbalance
- Temporal validation
- Missing identity information
- Potential leakage

With these considerations, the dataset supports development of a production-like antifraud pipeline.

---

## 13. Next Steps

Next phases in the project:

1. Statistical baseline analysis
2. Modeling strategy definition
3. Feature engineering pipeline
4. Training and evaluation
5. Deployment simulation
6. Monitoring and retraining