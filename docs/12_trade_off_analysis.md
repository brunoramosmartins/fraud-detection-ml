# Trade-off Analysis and Design Decision Log

This document records the key design decisions made throughout the project, the alternatives that were considered, the reasoning behind each choice, and the conditions under which a different choice would have been correct.

---

## Trade-off 1 — Model Selection: Gradient Boosting vs. Alternatives

**Decision made:** Gradient Boosting (GB) selected as the deployed model.

**Alternatives considered:**

*Logistic Regression (LR)*: fast to train, fully interpretable, and well-understood by regulators. LR would have been the correct choice if: (a) the institution required a scorecard-style model that analysts can inspect and explain to auditors; (b) latency constraints were extremely tight and model simplicity was required; or (c) the feature set contained strong linear signals and had been carefully engineered. In this dataset, with 380 numeric features including 339 anonymized Vesta variables with non-linear distributions, a linear boundary is structurally insufficient.

*Random Forest (RF)*: strong ensemble method with implicit feature selection and robustness to overfitting. RF was not selected because, in cost-sensitive evaluation, GB achieved lower expected monetary loss. The sequential error-correction mechanism in boosting concentrates model capacity on hard-to-classify cases — which in fraud are exactly the unusual patterns that cause the most loss. RF builds trees independently, without this correction mechanism.

*Neural networks*: not evaluated in this project. Would require substantially more data engineering, hyperparameter tuning, and infrastructure. The dataset size (~590k rows) does not clearly favor deep learning over gradient boosting, and the added complexity would be difficult to justify for a portfolio-scale project without a demonstrated performance advantage.

**When to choose differently:** LR if regulatory explainability is required (PCI-DSS compliance, BACEN audit requirements). RF if training speed matters and performance gap is acceptable. Neural networks if the feature set includes unstructured or high-cardinality categorical data that would benefit from learned embeddings.

---

## Trade-off 2 — Threshold Strategy: Fixed vs. Dynamic

**Decision made:** A fixed threshold (0.02) selected by minimizing Expected Monetary Loss at training time, stored in model metadata, and applied at inference time.

**Alternatives considered:**

*Dynamic threshold calibrated per segment*: apply different thresholds for different transaction segments (e.g., high-value vs. low-value transactions, new vs. returning customers). This would reflect that the cost of missing a $5,000 fraud is not the same as missing a $10 fraud. A production system operating at scale would implement this, but it requires a more sophisticated decision policy layer that is outside the scope of this simulation.

*Threshold recalibration at monitoring time*: recalibrate the threshold based on the current score distribution, rather than fixing it at training time. This is appropriate when the score distribution shifts (which PSI monitoring would detect) but the underlying model still discriminates well. Not implemented here due to complexity.

*No threshold (return probability only)*: some systems return a score and let downstream systems apply business logic. This decouples the model from the decision, which is architecturally clean but requires a separate policy component.

**When to choose differently:** In production, a tiered threshold strategy based on transaction value and customer segment would almost certainly outperform a single threshold. The fixed threshold is correct for a simulation but should not be treated as final in a real deployment.

**Sensitivity note:** if `c_fp` increases (operational review becomes more expensive), the optimal threshold shifts upward, generating fewer alerts. If the fraud base rate increases, the threshold may need to recalibrate downward to maintain loss reduction. The current approach does not adapt to these changes between training runs.

---

## Trade-off 3 — Feature Imputation: fillna(0.0) vs. Alternatives

**Decision made:** `fillna(0.0)` applied uniformly to all features before training and inference.

**Alternatives considered:**

*Median imputation*: replace missing values with the median of the training set. More statistically principled than zero-filling for features with continuous distributions. The problem is that computing and storing medians per feature adds pipeline state (must be fit on training data and stored for inference). For a quick first-pass pipeline, zero-filling avoids this complexity.

*Model-based imputation*: train a secondary model to predict missing values. Far too complex for the marginal gain expected in this dataset, where missingness may itself be informative (a missing `id_02` might indicate a non-standard device, which is a fraud signal).

*Indicator variables*: add a binary `{feature}_is_missing` flag for each feature with meaningful missingness. This preserves the signal that a feature was absent, rather than conflating "absent" with "zero". Not implemented, but worth considering for the high-missingness identity features (`id_*`).

**The core assumption of zero-fill:** in count and behavioral features (the `C` series, e.g., "count of transactions with matching card"), zero is a plausible value meaning "no prior activity". For time-delta features (`D` series), zero is less natural. For anonymized features (`V` series), the meaning of zero depends on the Vesta encoding.

**When to choose differently:** if model performance must be maximized, indicator variables plus calibrated imputation per feature group would be the correct approach. Zero-fill is a reasonable baseline but should be validated experimentally, especially for the identity features where missingness is structured (many transactions simply have no associated identity record).

---

## Trade-off 4 — Monitoring Scope: Single Feature vs. Full Feature Set

**Decision made:** PSI is computed for `TransactionAmt` only in `scripts/monitor_model.py`.

**Alternatives considered:**

*PSI for all model features (380 features)*: the most complete signal of distributional shift, but computationally intensive and generates a high-dimensional report that requires aggregation logic (e.g., "flag if max PSI exceeds threshold" or "flag if median PSI exceeds threshold"). For a portfolio implementation, this would require looping through all features and handling the many edge cases in PSI computation (constant features, low-variance features, etc.).

*PSI for top-N features by importance*: a practical middle ground that monitors the features that most influence model predictions. Requires that `feature_importances_` be stored in model metadata (not currently done) and then selecting the top-N by importance. This is the correct production approach.

*Score distribution monitoring*: monitor the distribution of output probabilities directly. Score drift is often the earliest, most actionable signal: if the fraction of transactions scoring above 0.02 changes substantially, the model's effective operating point has shifted even if input features have not. Not implemented but easier to compute than feature-level PSI.

**When to choose differently:** in production, the monitoring scope should be: (1) score distribution (always), (2) top-10 features by importance, and (3) any feature known to be sensitive to business changes (e.g., a new card program launches and `card1` distribution shifts). The current single-feature implementation is a proof of concept, not a production monitoring strategy.

---

## Trade-off 5 — Docker Artifact Strategy: Baked vs. External

**Decision made:** model artifact is copied into the Docker image via `COPY artifacts/models ./artifacts/models`.

**Alternatives considered:**

*Volume mount at runtime*: pass the artifact directory as a Docker volume. This keeps the image stateless and model-agnostic: the same image can serve any model version. Updating the model does not require rebuilding the image. The downside is that deployment requires coordinating the image with the correct artifact version — a potential source of misconfiguration.

*Pull from object storage at startup*: the container startup script downloads the model from S3 or GCS before the API begins serving. This is the standard pattern in production: the image is model-agnostic, and model versioning is managed in the artifact store, not in Docker tags. Requires cloud credentials, a startup hook with retry logic, and adds startup latency.

*Model server (e.g., Triton, TorchServe)*: a dedicated model serving framework handles artifact loading, versioning, and hot-swapping. Significant infrastructure investment; justified at scale but overkill for this project.

**The baked artifact approach** is appropriate for a portfolio demonstration where reproducibility and simplicity matter more than operational flexibility. For production, the pattern would be: stateless image + artifact from object storage + health check that confirms the model loaded before accepting traffic.

---

## Trade-off 6 — API Design: Synchronous Batch vs. Async vs. Streaming

**Decision made:** synchronous batch endpoint (`POST /predict` accepting a list of transactions, returning predictions immediately).

**Alternatives considered:**

*Async endpoint with message queue*: transactions arrive via Kafka or SQS. The API consumes from the queue, scores, and publishes results. This architecture decouples ingestion from scoring, enables horizontal scaling independently for each component, and provides durability if the scoring service is temporarily unavailable. Required for high-throughput systems (>1k transactions/second). Not appropriate for this project given the simulation scope.

*Per-transaction synchronous endpoint*: accept one transaction per request. Simpler contract, easier to reason about. The batch design was chosen because the `simulate_transactions.py` script sends data in batches, and batch scoring is more efficient (single model loading overhead shared across N transactions).

*WebSocket or server-sent events for streaming*: appropriate for scenarios where transaction scores must be pushed to a client in real time as they arrive. Adds protocol complexity without benefit for the batch simulation use case.

**When to choose differently:** if the real-time requirement is sub-100ms per transaction at scale, the synchronous batch design cannot meet SLAs. The correct architecture would be an async queue-based system with the scoring service reading from the queue continuously. For a single-instance pilot processing <100 transactions/second, synchronous batch is appropriate and simpler to operate and debug.
