# Submission Log — IEEE-CIS Fraud Detection (late submissions)

Rule: every submission is logged here **before** upload. LB scores are filled
in after Kaggle returns them. Late submissions receive public + private LB
scores but no leaderboard rank; percentile equivalences reference the frozen
2019 leaderboard (6,381 teams) and are always phrased as "would have placed".

| # | Date | Experiment | Notebook / commit | Feature set | Holdout AUC (A) | GroupKFold (B) | Expected LB | Public LB | Private LB |
|---|---|---|---|---|---|---|---|---|---|
| SUB-001 | 2026-07-10 | EXP-000 | `k00_baseline_submission` @ *(commit)* | v1 numeric-only, fillna(0), sklearn GB | 0.8614 | exempt | 0.83–0.86 | **0.8896** | **0.8749** |
| SUB-002 | 2026-07-10 | EXP-001 | `k01_lgbm_numeric` @ *(commit)* | v1 numeric-only, native NaN, LightGBM | 0.9124 | 0.9296 ± 0.0127 | private ≥ 0.890 (H1: +0.015 over SUB-001) | **0.9134** | **0.8877** |
| SUB-003 | 2026-07-10 | EXP-002 | `k02_categoricals` @ *(commit)* | v2a: + categorical encodings (label + freq + email split) | 0.9257 | 0.9398 ± 0.0109 | private ≥ 0.9077 (H2: +0.020 over SUB-002) | **0.9251** | **0.8968** |
| SUB-004 | 2026-07-10 | EXP-003 | `k03_time_amount` @ *(commit)* | v2b: + time/amount/D-norm (exploratory) | 0.9296 | 0.9428 ± 0.0106 | private ~0.90 (exploratory, +0.005 a +0.015) | **0.9284** | **0.8998** |
| SUB-005 | 2026-07-10 | EXP-004 | `k04_uid_aggregations` @ *(commit)* | v2c: + UID key + per-UID aggregates | 0.9299 | 0.9475 ± 0.0124 | private ≥ 0.9148 and ≥ 0.93 (H3: +0.015 over SUB-004) | **0.9314** | **0.9032** |

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
