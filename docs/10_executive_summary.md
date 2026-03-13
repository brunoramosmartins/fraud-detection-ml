# Executive Summary: Fraud Detection System

## The Problem

Card-not-present fraud — transactions made online without a physical card — represents a significant and growing financial risk for payment networks and card issuers. In the dataset used for this project, approximately 1 in 28 transactions is fraudulent.

The straightforward response to fraud is to block suspicious transactions. But this creates a different problem: incorrectly blocking a legitimate customer causes friction, damages trust, and generates operational costs. Every alert must be reviewed by a human analyst or trigger an automatic block — both of which carry a cost.

The challenge is not simply detecting fraud. It is deciding how aggressively to intervene, given that intervening too little means absorbing fraud losses, and intervening too much means degrading the customer experience and overwhelming review teams.

An institution that approves every transaction regardless of risk would absorb, in the simulated dataset used here, approximately **$610,000 in fraud losses** across the validation period. This is the cost of doing nothing — the baseline against which the system is evaluated.

---

## The Approach

This project builds a scoring system that evaluates each transaction as it arrives and assigns a fraud probability between 0 and 1. A transaction is flagged for intervention when its probability exceeds a defined threshold.

The critical design choice is that **the threshold is not arbitrary**. It is selected by minimizing the total expected cost — the combined cost of fraud that slips through and the operational cost of incorrectly flagging legitimate transactions. A low threshold catches more fraud but generates more false alarms; a high threshold reduces false alarms but misses more fraud. The system finds the point where total cost is minimized.

The model is trained on historical transaction data and uses behavioral and transactional signals that distinguish fraudulent from legitimate activity. It does not rely on rules written by humans. It learns patterns from data.

---

## The Result

At its operating threshold, the system reduces the expected monetary loss from $610,000 to $252,000 — a **reduction of 58.7%**, or $358,000 in prevented losses over the validation period.

This improvement comes at a cost: approximately **26% of legitimate transactions** are flagged at this operating point, requiring review or triggering a friction step such as additional authentication. The precision of the system — the fraction of flagged transactions that are genuinely fraudulent — is approximately 10%.

This precision figure requires context. With a fraud rate of 3.5%, a random review queue would contain 3.5% fraud. The model raises that to 10% — meaning the model makes the review queue roughly three times more efficient than random sampling. In practice, financial institutions accept low precision at the alert stage because the cost of missing fraud exceeds the cost of reviewing a false alarm.

The model's discriminative ability — how well it separates fraudulent from legitimate transactions across all possible thresholds — achieves a score of **0.861 on a 0–1 scale**, where 1 represents perfect separation. This indicates strong signal in the data that the model is successfully capturing.

---

## The Trade-off

The threshold of 0.02 was chosen because, at that point, total cost is minimized given the assumed cost structure: missing a fraud costs the full transaction amount, while a false alarm costs a fixed operational fee.

If the institution is more concerned about customer experience and has a limited review team, a higher threshold would be appropriate — fewer alerts, but more fraud slipping through. If the institution is in a high-risk environment or faces regulatory pressure to minimize fraud losses, a lower threshold may be warranted.

This is not a technical decision. It is a business decision that should be revisited periodically as fraud patterns evolve, operational capacity changes, and cost assumptions are updated.

---

## The Limits

This system was built and evaluated on a single historical dataset. In a real deployment, fraud patterns change continuously — fraudsters adapt their behavior in response to detection systems. A model trained on historical data begins to degrade as soon as it is deployed, and must be retrained regularly.

The system includes a monitoring component that tracks whether the distribution of incoming transactions has shifted relative to the training data. When shift is detected above a defined threshold, a retraining workflow is triggered automatically. This simulation demonstrates the concept; a production system would require additional safeguards, human oversight, and a more sophisticated retraining strategy.

Additionally, this system produces a probability score but not an explanation. For a human analyst reviewing a flagged transaction, knowing the score is 0.85 provides limited guidance on why the transaction was flagged or what to look for. Explainability is a known gap in this type of model and an important consideration for regulatory compliance.

---

## Summary

The system transforms a fraud detection problem into a cost minimization problem, selects a model and operating threshold that minimize total expected loss, and delivers a **58.7% reduction in expected fraud-related costs** compared to an uncontrolled approval policy. It operates as a containerized service, can process transactions in real time, and includes mechanisms to detect when the model's environment has shifted and retraining is needed.
