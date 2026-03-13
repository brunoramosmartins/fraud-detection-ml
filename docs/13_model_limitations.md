# Model Limitations and Failure Mode Analysis

This document describes the known limitations of the fraud detection system as built, the consequences of each limitation in a real deployment, and what would be required to address them. Limitations are stated as current constraints, not future enhancements.

---

## Limitation 1 — Static Training Data: Concept Drift

**Constraint:** The model was trained on a single historical snapshot of the IEEE-CIS dataset (transactions from a fixed time window). Once deployed, the model does not update its parameters as new data arrives. It becomes progressively less accurate as fraud patterns evolve.

**Consequence in production:** Fraud tactics change continuously. New attack vectors (synthetic identities, account takeover via new channels, bot-driven carding attacks) produce transaction signatures that were absent at training time. The model will score these transactions as low-risk, causing a silent increase in fraud losses that may not be immediately visible in review queues.

**What would address this:** A continuous learning strategy with periodic retraining on recent labeled data, a champion/challenger framework to evaluate candidate models before promotion, and a monitoring system that tracks business KPIs (chargeback rate, fraud rate) in addition to statistical drift metrics. Labeling latency must be accounted for: chargebacks from fraud may not be confirmed for 30–90 days after the transaction, creating a feedback delay that complicates timely retraining.

---

## Limitation 2 — Class Imbalance: Rare New Fraud Patterns

**Constraint:** The model was trained without explicit class balancing techniques (no SMOTE, no class weights). The imbalance (~3.5% fraud) was addressed through threshold optimization against Expected Monetary Loss, not through resampling.

**Consequence in production:** The model has low sensitivity to fraud subtypes that are rare in the training data. A new fraud ring that constitutes 0.01% of training transactions will have minimal influence on the learned decision boundary. The model may perform well in aggregate while failing to detect emerging, high-value fraud vectors.

**What would address this:** Ensemble approaches that combine a general fraud detector with anomaly detection models trained specifically to flag statistically unusual transactions, regardless of whether they match historical fraud patterns. Additionally, active learning — surfacing uncertain or unusual scores for human review — can generate labeled data for rare patterns faster than waiting for chargeback labels.

---

## Limitation 3 — Potential Feature Leakage

**Constraint:** Several features in the IEEE-CIS dataset are behavioral aggregates or counts that could, in a real-time system, be computed differently than how they appear in the historical data. The `C` series features (e.g., "count of transactions matching by card") aggregate information that might not be fully available at the moment of the transaction decision.

**Consequence in production:** If any feature is computed using information that would not be available at inference time (e.g., a count that includes the current transaction itself, or a feature that implicitly uses post-transaction data), the model would overestimate its real-world performance. This is a common source of inflated metrics in fraud ML projects.

**What would address this:** Rigorous point-in-time feature computation in a feature store, where each feature is computed using only the data available strictly before the transaction timestamp. In a production system, this requires careful definition of each feature's lookback window and the exact timestamp anchor used for computation. Temporal cross-validation across multiple time windows (not just one split) provides additional validation that leakage is not present.

---

## Limitation 4 — Identity Data Dependency

**Constraint:** The model uses 14 identity features (`id_01` through `id_32`, non-contiguous) that are available for approximately 40% of transactions in the dataset. The remaining 60% of transactions have no associated identity record. For those transactions, `fillna(0.0)` is applied, which may not be an appropriate imputation.

**Consequence in production:** If the identity enrichment service is unavailable at inference time (API timeout, downstream failure, or device data not captured), all identity features default to zero. The model will score these transactions using only transaction and card features. Whether this degrades performance depends on how much predictive signal the identity features carry that is not captured by transaction features. This degradation is not currently quantified.

**What would address this:** Train a separate "identity-absent" model trained only on transactions without identity data. Route transactions to the appropriate model based on whether identity data is available. This is a form of model specialization that prevents the zero-imputed identity values from contaminating the score of transactions that genuinely have no identity data. The current system conflates "identity unavailable" with "identity = zero", which is an incorrect assumption.

---

## Limitation 5 — Adversarial Robustness

**Constraint:** The model is not adversarially robust. If a fraudster learns the feature set or can infer the decision boundary from observed accept/decline responses, they can systematically probe the model to find transaction configurations that score below the threshold.

**Consequence in production:** Sophisticated fraud operations can reverse-engineer the model's behavior through probing attacks: submit small test transactions, observe outcomes, and gradually learn which feature patterns avoid detection. Gradient boosting models, despite not being interpretable to a human, are not opaque to systematic probing. Published feature names (as in this project) accelerate this attack.

**What would address this:** Feature obfuscation (the Vesta `V` features are already anonymized, which is one mitigation), rate limiting on declined transactions to prevent systematic probing, periodic model refresh to invalidate learned adversarial patterns, and velocity checks that detect unusual patterns in declined-then-accepted sequences. Additionally, not publishing the exact feature list and threshold in a public repository is a basic operational security measure not observed in this portfolio project (intentionally, for transparency).

---

## Limitation 6 — Explainability and Regulatory Constraints

**Constraint:** The deployed model is a 100-estimator Gradient Boosting ensemble. It produces a probability score but no explanation of which features drove that score for a specific transaction. In Brazil, the LGPD (Lei Geral de Proteção de Dados) and BACEN regulations require that automated financial decisions affecting consumers be explainable upon request. The current system cannot satisfy this requirement.

**Consequence in production:** A customer who has a transaction blocked cannot be told why it was blocked beyond a generic message. An analyst reviewing a flagged transaction cannot see which features triggered the flag, making their review less efficient. A regulator auditing the system's fairness cannot evaluate whether the model systematically disadvantages any protected group.

**What would address this:** SHAP (SHapley Additive exPlanations) values computed post-hoc provide feature-level attributions for each prediction from tree-based models. SHAP can explain why a specific transaction received its score in terms of the top contributing features. This adds inference latency and must be stored per prediction for auditability. Alternatively, a simpler linear scoring model (logistic regression or scorecard) that is inherently interpretable may be required in highly regulated contexts, accepting the performance cost. The appropriate choice depends on the regulatory environment and the institution's risk appetite.

---

## Limitation 7 — Simulation vs. Production Gap

**Constraint:** Several components of this system are simulated rather than implemented to production standards. The gap between what is built and what a production deployment would require is substantial.

| Component | This Implementation | Production Requirement |
|---|---|---|
| Model storage | Local file directory | Model registry (MLflow, Vertex AI) with promotion workflows |
| Feature serving | Recomputed from raw data | Feature store with sub-millisecond retrieval |
| Model promotion | Automatic | Human-approved champion/challenger evaluation |
| Monitoring | PSI on one feature | Score distribution + top-N features + business KPIs |
| Retraining | Triggered automatically | Alert → candidate model → shadow evaluation → human approval |
| API resilience | Single instance, no fallback | Load-balanced, circuit breakers, fail-open/closed policy |
| Latency | Not tested | Sub-100ms P99 for online fraud scoring |
| Data pipeline | Batch CSV loading | Real-time feature computation at transaction time |
| Audit trail | Run JSON files | Immutable audit log for regulatory compliance |
| Security | No authentication on API | mTLS, API gateway, request signing |

**What would address this:** Deploying this system in production would require treating each row of the table above as a separate engineering project. The current implementation demonstrates the correct concepts and interfaces; it does not demonstrate production engineering. The distinction is intentional: this is a portfolio project demonstrating ML engineering competency, not a production deployment.
