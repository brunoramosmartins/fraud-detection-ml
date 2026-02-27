# Fraud Detection System – Functional Scope

## 1. Problem Statement

The objective of this system is to detect fraudulent e-commerce transactions in near real-time in order to support automated approval, manual review, or transaction blocking decisions, minimizing financial loss and customer friction.

This project is based on the IEEE Fraud Detection dataset and is designed to simulate a real-world fraud detection system in a banking or fintech environment.

---

## 2. System Objective

The system estimates the probability that a transaction is fraudulent and uses this probability to support operational decisions that reduce expected financial loss while maintaining acceptable customer experience.

Model output → fraud probability  
Decision engine → approve / review / block

---

## 3. System Inputs and Outputs

### Inputs
Each transaction contains information such as:

- Payment details
- Device and browser information
- Transaction metadata
- Behavioral features derived from historical activity

These features are assumed to be available at the time of transaction authorization.

### Outputs
For each transaction, the system returns:

- Fraud probability score (0–1)
- Decision label:
  - Approve
  - Send to manual review
  - Block transaction
- Metadata for logging and audit

---

## 4. Stakeholders

The system serves multiple simulated stakeholders:

- Risk Team → defines thresholds and evaluates fraud performance
- Customer Support → handles manual review and customer disputes
- Compliance/Audit → validates decision traceability
- Data Science Team → maintains models and monitoring
- Data Engineering Team → maintains data pipelines

---

## 5. Functional Scope

### Real-Time Scoring

The system evaluates transactions before authorization.

Steps:
1. Transaction arrives from payment gateway.
2. Features are generated or retrieved.
3. Fraud model predicts probability.
4. Decision engine applies thresholds.
5. Response returned to payment system.

Target latency constraint (assumed): < 200 ms.

---

### Batch Processing

Batch processes are required for model lifecycle management.

Includes:

- Periodic model training with new labeled data
- Performance evaluation on recent transactions
- Model recalibration
- Drift monitoring
- Threshold optimization

Batch jobs run daily or weekly.

---

## 6. Decision Policy

Decisions are based on probability thresholds defined by the risk team.

Example policy:

- Score < T1 → Approve
- T1 ≤ Score < T2 → Manual Review
- Score ≥ T2 → Block

Thresholds are optimized to minimize expected financial loss.

---

## 7. Business Constraints

The system must satisfy operational constraints:

- Low latency for real-time scoring
- High recall for fraudulent transactions
- Controlled false positive rate
- Decision explainability for audit
- Reproducible model training

---

## 8. Assumptions

Because real bank data is not available, the following assumptions are made:

- Fraud rate is approximately 1–3% of transactions
- Cost of false negative equals transaction value
- Cost of false positive equals customer support cost plus estimated churn impact
- Features available in dataset approximate real transaction signals
- Historical data is available for batch retraining

These assumptions will be revisited during later phases.

---

## 9. Out of Scope

The following topics are not included in the initial phase:

- Integration with real payment gateways
- Customer identity verification workflows
- Advanced graph-based fraud detection
- Legal/regulatory certification

These may be considered as future extensions.

---

## 10. Success Criteria

The system is considered successful if it:

- Reduces expected financial loss compared to baseline rules
- Maintains acceptable false positive rate
- Meets latency requirements
- Provides reproducible and auditable decisions
- Can be monitored and retrained automatically

---

## 11. Next Steps

After defining functional scope, the next phases are:

1. Business Cost Modeling
2. Statistical Baseline
3. Modeling Strategy
4. Engineering Pipeline
5. Deployment Simulation