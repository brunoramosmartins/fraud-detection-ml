# IEEE-CIS Fraud Detection – End-to-End ML System Simulation

## 1. Project Overview

This repository simulates a production-grade fraud detection system for card-not-present (CNP) e-commerce transactions.

The project is based on the IEEE-CIS Fraud Detection dataset and is structured as a senior-level Machine Learning Engineering project. The objective is not to maximize leaderboard performance, but to design and document a realistic, cost-sensitive antifraud system aligned with business and operational constraints.

The system is treated as a decision engine that:

- Estimates fraud probability for each transaction
- Applies threshold-based business logic
- Minimizes expected monetary loss
- Respects operational constraints (latency, review capacity)

This repository is organized using issues, milestones, and structured documentation to reflect real-world ML system development.

---

## 2. Business Objective

The formal objective of the system is:

Minimize expected monetary loss from fraudulent transactions while controlling customer friction and operational review costs.

This is implemented through:

- Explicit cost modeling (False Positives vs False Negatives)
- Expected Monetary Loss optimization
- Threshold selection under operational constraints
- Baseline comparison against "approve-all" strategy

The system is therefore framed as a cost-sensitive decision problem, not a generic binary classification task.

---

## 3. System Scope

The system simulates a real-world antifraud pipeline including:

### Real-Time Scoring

- Transaction event received from payment gateway
- Feature generation and retrieval
- Fraud probability prediction
- Decision engine (approve / review / block)
- Logging for monitoring and audit

Latency assumption:
- P95 < 200 ms

### Batch Training and Monitoring

- Periodic retraining on labeled historical data
- Temporal validation (no random splits)
- Threshold optimization
- Drift monitoring
- Model versioning

This separation reflects production ML architecture patterns.

---

## 4. Cost Modeling and Optimization

The system uses an explicit asymmetric cost structure:

- Cost of False Negative (missed fraud):
  Approximately equal to TransactionAmount

- Cost of False Positive (incorrectly flagged legitimate transaction):
  Fixed operational cost (e.g., 5 monetary units)

Primary optimization metric:

Expected Monetary Loss

Thresholds are selected via:

T* = argmin_T L(T)

subject to operational constraints such as review capacity.

This aligns model selection directly with business impact.

---

## 5. Evaluation Strategy

Due to temporal drift in transaction behavior:

- No random train/test split
- Temporal validation (train on past, validate on future)

For each model version:

- Compute Expected Monetary Loss
- Compute PR-AUC and ROC-AUC
- Compute Fraud Detection Rate (Recall)
- Compute False Positive Rate
- Evaluate Review Rate
- Compare against baseline loss

This simulates real deployment behavior.

---

## 6. High-Level Architecture

The system architecture separates:

- Prediction (Fraud Model API)
- Decision Logic (Threshold-based policy)
- Logging and Monitoring
- Batch Training Pipeline
- Model Registry

See:
docs/04_architecture.md

The architecture reflects production ML system design rather than notebook-only experimentation.

---

## 7. Repository Structure

docs/
- 01_system_scope.md
- 02_data_understanding.md
- 03_metrics_and_cost_modeling.md
- 04_architecture.md

notebooks/
- eda_v1.ipynb
- (future modeling and evaluation notebooks)

src/
- (future feature engineering, training, evaluation modules)

This structure mirrors professional ML project organization.

---

## 8. Development Roadmap

Phase 0 – System Framing and Architecture  
- Business framing  
- Cost modeling  
- Architecture definition  
- Metrics formalization  

Phase 1 – Statistical Baseline and Modeling Strategy  
- Baseline model  
- Temporal validation setup  
- Threshold optimization  

Phase 2 – Feature Engineering and Training Pipeline  
- Robust preprocessing  
- Model comparison  
- Cost-based evaluation  

Phase 3 – Monitoring and Deployment Simulation  
- Drift analysis  
- Threshold recalibration  
- Model version comparison  

Each phase is tracked through issues and milestones to simulate real engineering workflow.

---

## 9. Positioning

This repository is designed to demonstrate:

- Cost-sensitive modeling
- Business-aligned ML decision systems
- Production-oriented architecture
- Structured ML project development
- Reproducibility and documentation discipline

It is intended as a professional portfolio project reflecting senior-level Machine Learning Engineering practices rather than a competition-focused solution.

---

## 10. Future Work

- Multiple model comparison (Logistic Regression, Gradient Boosting, etc.)
- Calibration analysis
- Three-way decision optimization (approve / review / block)
- Drift simulation and monitoring dashboards
- Configuration-driven cost parameterization
- Packaging into deployable scoring service

---

## 11. Disclaimer

This project uses the IEEE-CIS dataset as a proxy for real banking data.  
Certain assumptions (cost structure, fraud rate, operational capacity) are simulated for educational and portfolio purposes.