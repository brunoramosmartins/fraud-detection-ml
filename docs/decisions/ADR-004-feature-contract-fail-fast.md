# ADR-004 — Feature Contract Enforcement: Fail-Fast on Missing Columns

## Status

Accepted

## Context

At inference time, the API receives a JSON payload from an external caller.
The payload may be missing features that the model was trained on — due to
upstream data pipeline failures, schema drift, or intentional payload
trimming.

Two strategies were considered:

1. **Silent imputation:** if a required feature is missing, fill it with
   a default value (0.0, median, or the training mean) and score anyway.
2. **Fail-fast:** if any required feature is missing, reject the request
   with `HTTP 422` and list the missing columns.

## Decision

Fail-fast. If any column in `feature_list` is absent from the payload,
the API returns `HTTP 422 Unprocessable Entity` immediately.

```python
missing = [c for c in feature_list if c not in df.columns]
if missing:
    raise HTTPException(
        status_code=422,
        detail=f"Missing required features for inference: {missing}",
    )
```

## Consequences

**Positive:**
- Schema drift becomes visible immediately at the API boundary. If an
  upstream pipeline stops sending `card5`, the API returns 422 within
  milliseconds. Without fail-fast, the model would score with imputed
  values and the issue would only surface later through metric degradation
  — which is much harder to diagnose and may take days to detect.
- The contract between the caller and the API is explicit and enforced.
  The caller knows exactly what the API requires because it will fail
  loudly if the contract is violated.
- Model predictions are only returned when the model is operating within
  its training distribution (feature-wise). There is no silent corner case
  where missing features produce nonsense scores.

**Negative:**
- The API is less tolerant of partial payloads. A caller that sends most
  features but not all will receive an error and must handle it. Silent
  imputation would allow graceful degradation.
- In production, some features may be legitimately unavailable for specific
  transaction types (e.g., identity features for guests). Fail-fast forces
  the caller to handle these cases explicitly, which increases integration
  complexity.
- The distinction between "feature missing due to pipeline failure" (should
  fail) and "feature genuinely absent for this transaction type" (could be
  imputed) is not made. A more sophisticated contract would separate required
  features from optional features with defined defaults.

**Why fail-fast is the right default:** in a fraud detection system, a
wrong score is worse than no score. A model that silently scores with
imputed values for missing identity features may produce miscalibrated
probabilities that cause the system to approve fraudulent transactions or
block legitimate ones at rates outside expected operational parameters.
Visibility of failures is a safety property.
