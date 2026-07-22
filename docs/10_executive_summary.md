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

At its operating threshold, the system reduces the expected monetary loss from $610,000 to $175,000 — a **reduction of 71.3%**, or $435,000 in prevented losses over the validation period. The first-generation system (v1) achieved a 58.7% reduction; the current model (v2, introduced in Phase 9) recovers a further $77,000 by restoring the categorical signals v1 discarded and letting the model handle missing data natively.

This improvement comes at a cost: approximately **14% of legitimate transactions** are flagged at this operating point (down from 26% in v1), requiring review or triggering a friction step such as additional authentication. The precision of the system — the fraction of flagged transactions that are genuinely fraudulent — is approximately 18%.

This precision figure requires context. With a fraud rate of 3.5%, a random review queue would contain 3.5% fraud. The model raises that to 18% — meaning the model makes the review queue roughly five times more efficient than random sampling (v1 managed three times). In practice, financial institutions accept low precision at the alert stage because the cost of missing fraud exceeds the cost of reviewing a false alarm.

The model's discriminative ability — how well it separates fraudulent from legitimate transactions across all possible thresholds — achieves a score of **0.930 on a 0–1 scale** (v1: 0.861), where 1 represents perfect separation. This indicates strong signal in the data that the model is successfully capturing.

---

## The Trade-off

The threshold of 0.003 was chosen because, at that point, total cost is minimized given the assumed cost structure: missing a fraud costs the full transaction amount, while a false alarm costs a fixed operational fee. (v1 operated at 0.02; the v2 model separates classes more sharply, which pushes its cost-optimal threshold lower.)

If the institution is more concerned about customer experience and has a limited review team, a higher threshold would be appropriate — fewer alerts, but more fraud slipping through. If the institution is in a high-risk environment or faces regulatory pressure to minimize fraud losses, a lower threshold may be warranted.

This is not a technical decision. It is a business decision that should be revisited periodically as fraud patterns evolve, operational capacity changes, and cost assumptions are updated.

---

## The Limits

This system was built and evaluated on a single historical dataset. In a real deployment, fraud patterns change continuously — fraudsters adapt their behavior in response to detection systems. A model trained on historical data begins to degrade as soon as it is deployed, and must be retrained regularly.

The system includes a monitoring component that tracks whether the distribution of incoming transactions has shifted relative to the training data. When shift is detected above a defined threshold, a retraining workflow is triggered automatically. This simulation demonstrates the concept; a production system would require additional safeguards, human oversight, and a more sophisticated retraining strategy.

Additionally, the score alone provides limited guidance to a human analyst. Since Phase 9 the system computes per-prediction attributions (TreeSHAP): for any flagged transaction, the top features driving the score can be listed, supporting analyst review and regulatory explainability requirements. Serving these explanations in real time (rather than on demand) remains future work.

---

## Summary

The system transforms a fraud detection problem into a cost minimization problem, selects a model and operating threshold that minimize total expected loss, and delivers a **71.3% reduction in expected fraud-related costs** (58.7% in its first generation) compared to an uncontrolled approval policy. It operates as a containerized service, can process transactions in real time, explains individual decisions on demand, and includes mechanisms to detect when the model's environment has shifted and retraining is needed.
