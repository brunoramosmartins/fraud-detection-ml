# Submission Log — IEEE-CIS Fraud Detection (late submissions)

Rule: every submission is logged here **before** upload. LB scores are filled
in after Kaggle returns them. Late submissions receive public + private LB
scores but no leaderboard rank; percentile equivalences reference the frozen
2019 leaderboard (6,381 teams) and are always phrased as "would have placed".

| # | Date | Experiment | Notebook / commit | Feature set | Holdout AUC (A) | GroupKFold (B) | Expected LB | Public LB | Private LB |
|---|---|---|---|---|---|---|---|---|---|
| SUB-001 | 2026-07-10 | EXP-000 | `k00_baseline_submission` @ *(commit)* | v1 numeric-only, fillna(0), sklearn GB | 0.8614 | exempt | 0.83–0.86 | *(pending)* | *(pending)* |

## Submission command (author executes)

```bash
kaggle competitions submit ieee-fraud-detection \
  -f submission.csv \
  -m "EXP-000 baseline: production pipeline reproduction (numeric-only, sklearn GB)"
```

## Frozen-LB reference points (private)

| Position | Private AUC |
|---|---|
| 1st | ~0.9459 |
| Gold zone (top ~13) | ~0.94+ |
| Silver zone (top ~5%) | ~0.934+ |
| Median | ~0.91 |

*(reference points from the public frozen leaderboard; used only for
"would-have-placed" phrasing, never as significance evidence)*
