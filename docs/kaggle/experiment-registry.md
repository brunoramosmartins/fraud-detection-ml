# Experiment Registry — Phase 8

Rule: every experiment is registered here **before** it runs. The entry states
the expected result; the actual result is filled in afterwards. Protocol:
`validation-protocol.md` v1. Hypotheses: `research.md`.

| Field | Meaning |
|---|---|
| ID | EXP-xxx, immutable |
| Hypothesis | H1–H4, `exploratory`, or `none` |
| Feature diff | exactly what changed vs the predecessor experiment |
| Expected | predicted ΔAUC (registered before running) |
| A / B | Scheme A holdout AUC / Scheme B GroupKFold mean ± std |
| DeLong | ΔAUC [95% CI], p vs predecessor (Scheme A) |
| LB | public / private (from submission log) |

---

## EXP-000 — Baseline: current production model, external anchor

| Field | Value |
|---|---|
| Registered | 2026-07-10 |
| Hypothesis | none (anchor) |
| Notebook | `notebooks/kaggle/k00_baseline_submission/` |
| Predecessor | — |
| Feature diff | — faithful reproduction of the production pipeline: 380 numeric-only features, `fillna(0)`, sklearn GB (80 trees, depth 5, lr 0.1, min_samples_leaf 100, subsample 0.8, seed 42), trained on Scheme-A train partition only |
| Config | `configs/model_gb_v1.yml` equivalents, hardcoded in notebook for reproducibility |
| Expected | Scheme-A holdout AUC ≈ 0.861 (reproduction check, tolerance ±0.003); private LB expected *below* holdout (temporal drift + one-month gap in test period), rough guess 0.83–0.86 |
| A (holdout AUC) | **0.8614** — reproduction check PASSED (expected 0.861 ± 0.003) |
| B (GroupKFold) | exempt (see protocol) |
| DeLong | — (no predecessor) |
| LB public / private | **0.8896 / 0.8749** (SUB-001, 2026-07-10) |
| Verdict / notes | Anchor established. Fit time 16.2 min on Kaggle CPU. Kaggle env inferred 400 numeric features vs 380 documented locally (pandas dtype-inference difference across versions); holdout AUC matched regardless, so the pipelines are functionally equivalent. Data mounts at `/kaggle/input/competitions/ieee-fraud-detection` (new layout) — notebook locates files by search. Holdout predictions saved as `holdout_pred_exp000.csv` for the EXP-001 DeLong test. **Expectation miss (recorded):** registered guess was private LB 0.83–0.86 (below holdout); actual came in above the holdout (0.8749 private, 0.8896 public). Interpretation: AUC on different time windows/populations is not directly comparable; the registered expectation over-weighted temporal degradation. Public > private is consistent with drift increasing within the test period. Both scores remain far below the competitive zone (~0.93+), as intended for the anchor. |

---

## EXP-001 — H1: LightGBM on identical numeric features

| Field | Value |
|---|---|
| Registered | 2026-07-10 |
| Hypothesis | **H1** |
| Notebook | `notebooks/kaggle/k01_lgbm_numeric/` |
| Predecessor | EXP-000 |
| Feature diff | **none** — same runtime-inferred numeric feature list. Only two changes, bundled by design (see H1 statement): model class (sklearn GB → LightGBM) and imputation (`fillna(0)` removed; native NaN handling) |
| Config | LightGBM, fixed and registered before running: `learning_rate=0.05`, `num_leaves=192`, `min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `seed=42`. `n_estimators` via early stopping (patience 200) on an ES split *inside* the Scheme-A train partition (last month of the train window — the holdout is never used for model selection); refit on the full train partition at `int(best_iter * 1.1)` |
| Expected | H1 threshold: private LB ΔAUC ≥ +0.015 vs EXP-000 (i.e. private ≥ 0.890); DeLong on Scheme-A holdout significant at α=0.05 in the same direction. Scheme B (GroupKFold) reported for the first time — feeds H4 |
| A (holdout AUC) | **0.9124** |
| B (GroupKFold) | **0.9296 ± 0.0127** — per-fold: [0.9031, 0.9285, 0.9383, 0.9297, 0.9408, 0.9234, 0.9433] (7 folds — see protocol amendment) |
| DeLong vs EXP-000 | ΔAUC **+0.0510** [95% CI +0.0461, +0.0558], z = 20.57, p = 4.7e-94 — significant, predicted direction |
| LB public / private | **0.9134 / 0.8877** (SUB-002, 2026-07-10) |
| Verdict / notes | **H1: INCONCLUSIVE (leaning supported)** per the pre-registered rule: internal DeLong strongly significant (+0.0510), but private LB Δ = +0.0128 < +0.015 threshold. Public LB Δ = +0.0238 *would* have met the bar. Key finding: the model-class gain decays with temporal distance (holdout +0.051 → public +0.024 → private +0.013); the registered threshold ignored drift decay. H4 gap data: \|B mean − private\| = 0.0419, \|A − private\| = 0.0247 (temporal split closer, this experiment). ES: best_iter 213 → refit 234 rounds; total runtime ~18 min (vs 16 min for a single sklearn GB fit — 8 LightGBM fits in the same budget, confirming ADR-007's speed claim). |

---

## EXP-002 — H2: categorical features restored

| Field | Value |
|---|---|
| Registered | 2026-07-10 (before running) |
| Hypothesis | **H2** |
| Notebook | `notebooks/kaggle/k02_categoricals/` |
| Predecessor | EXP-001 |
| Feature diff | + **label encoding** of every object-dtype column (~31: ProductCD, card4/card6, P_/R_emaildomain, M1–M9, object-typed id_12–id_38, DeviceType, DeviceInfo), missing as its own category, unseen → −1; + **frequency encoding** (train-fit, normalized) of the same object columns plus the numeric-coded high-cardinality categoricals card1, card2, card3, card5, addr1, addr2; + **email provider/suffix split** of P_/R_emaildomain (4 derived columns, label- and frequency-encoded). Canonical implementations: `src/features/engineering.py`. Model and params unchanged from EXP-001. |
| Config | LightGBM identical to EXP-001 (registered params, seed 42). Encoders fit on each fold's training months (Scheme B) and on the train partition (Scheme A model + submission) — never on scored rows. |
| Expected | H2 threshold: private LB ΔAUC ≥ +0.020 vs EXP-001 (private ≥ 0.9077); DeLong on holdout significant. Given H1's drift-decay finding, internal Δ is expected to exceed LB Δ: registered guess holdout Δ +0.025 to +0.040. |
| A (holdout AUC) | **0.9257** |
| B (GroupKFold) | **0.9398 ± 0.0109** — per-fold: [0.9181, 0.9400, 0.9456, 0.9401, 0.9515, 0.9321, 0.9514] |
| DeLong vs EXP-001 | ΔAUC **+0.0133** [95% CI +0.0109, +0.0156], z = 10.90, p = 1.1e-27 — significant, predicted direction |
| LB public / private | **0.9251 / 0.8968** (SUB-003, 2026-07-10) |
| Verdict / notes | **H2: INCONCLUSIVE** by the pre-registered rule (internal DeLong significant & positive, but private LB Δ = +0.0091 < +0.020 threshold — criteria disagree). **The comparative core of H2 is refuted:** categoricals gained *less* than the H1 model swap on both scales — internal holdout +0.0133 (vs H1 +0.0510) and private LB +0.0091 (vs H1 +0.0128). The block does add internally-significant signal and 0.8968 was the series best at the time, but it is not the bigger lever H2 predicted. H4 gaps: \|A−private\| = 0.0289, \|B−private\| = 0.0430 (temporal split closer again). |

---

## EXP-003 — Exploratory: time and amount features

| Field | Value |
|---|---|
| Registered | 2026-07-10 (before running) |
| Hypothesis | exploratory (no hypothesis — one feature block per submission; feeds H4 gap tracking and sets the predecessor for H3) |
| Notebook | `notebooks/kaggle/k03_time_amount/` |
| Predecessor | EXP-002 |
| Feature diff | + `tx_hour`, `tx_dow` (periodic only — no absolute time index, to avoid trend extrapolation); + `amt_log1p`, `amt_cents` (foreign-currency cent signal); + `D{1..15}_norm` except D9 (D − day, converting timedelta counters to time-invariant reference dates; originals kept). All row-local, leak-free by construction. Canonical implementations: `src/features/engineering.py`. Model and params unchanged. |
| Config | LightGBM identical to EXP-001/002 (registered params, seed 42). Row-local block needs no fit/transform boundary. |
| Expected | Holdout Δ +0.005 to +0.015 vs EXP-002; the D-normalization should specifically shrink the CV−LB gap (anti-drift transform) — watch the H4 gap metrics. |
| A (holdout AUC) | **0.9296** |
| B (GroupKFold) | **0.9428 ± 0.0106** — per-fold: [0.9215, 0.9485, 0.9498, 0.9436, 0.9551, 0.9339, 0.9469] |
| DeLong vs EXP-002 | ΔAUC **+0.0039** [95% CI +0.0019, +0.0059], z = 3.85, p = 1.2e-04 — significant, positive |
| LB public / private | **0.9284 / 0.8998** (SUB-004, 2026-07-10) |
| Verdict / notes | Exploratory (no hypothesis). Small but internally-significant gain (+0.0039 holdout, +0.0030 private over EXP-002); private 0.8998 is the series best. The time/amount/D-norm block helps modestly. H4 gaps: \|A−private\| = 0.0298, \|B−private\| = 0.0430 (temporal split closer, 3rd time). Notebook was modeling-only (the new pattern); DeLong computed off-notebook via `scripts/kaggle_delong.py`. Feature count 418 numeric+row-local + categorical block. |

---

## EXP-004 — H3: UID entity aggregation

| Field | Value |
|---|---|
| Registered | 2026-07-10 (before running) |
| Hypothesis | **H3** |
| Notebook | `notebooks/kaggle/k04_uid_aggregations/` |
| Predecessor | EXP-003 |
| Feature diff | + **UID key** `card1 "_" addr1 "_" round(TransactionDT/86400 − D1)` (pseudo-client id), frequency-encoded; + **per-UID aggregates** computed over the train+test union (label-free, transductive): `uid_count`, `uid_amt_mean`, `uid_amt_std`, `uid_amt_ratio`. Canonical implementations: `src/features/engineering.py` (`make_uid`, `add_uid_aggregates`). Model and params unchanged. |
| Config | LightGBM identical to EXP-001..003 (registered params, seed 42). UID aggregates fit over the full union once (no label); categorical encoders keep the per-fit discipline. |
| Expected | H3 threshold: private LB ΔAUC ≥ +0.015 vs EXP-003 (private ≥ 0.9148) **and** private ≥ 0.93. Given the H1/H2 drift-decay pattern, the internal holdout Δ is expected to be substantially larger than the private Δ. This is the block that decided the 2019 competition, so a large internal jump is expected regardless of the LB verdict. |
| A (holdout AUC) | **0.9299** |
| B (GroupKFold) | **0.9475 ± 0.0124** — per-fold: [0.9222, 0.9518, 0.9522, 0.9452, 0.9575, 0.9404, 0.9629] |
| DeLong vs EXP-003 | ΔAUC **+0.0004** [95% CI −0.0012, +0.0020], z = 0.45, **p = 0.65 — NOT significant** |
| LB public / private | **0.9314 / 0.9032** (SUB-005, 2026-07-10) |
| Verdict / notes | **H3: REJECTED.** Both criteria fail: internal DeLong is *not significant* (Δ +0.0004, p = 0.65 — indistinguishable from zero on the Scheme-A holdout) AND private LB Δ = +0.0034 << +0.015 threshold, with private 0.9032 < 0.93. Per the rule (both criteria against the prediction → rejected). **Interpretation:** the famous "magic feature" is not magic *in this minimal form* — a 4-aggregate UID (count, amt mean/std/ratio) + frequency-encoded key. The 2019 top solutions derived the jump from *rich* per-UID aggregation (nunique of many categoricals, per-UID D-stats, dozens of columns), not the key alone. Signal exists on the broader distribution (GroupKFold +0.0047, private +0.0034, both series-best) but is ~0 on the recent-slice holdout. 431,398 UIDs over 1,097,231 union rows. H4 gaps: \|A−private\| = 0.0267, \|B−private\| = 0.0443. |

---

## EXP-006 — Confirmatory: full aggregation engine (winning-solution replication)

| Field | Value |
|---|---|
| Registered | 2026-07-11 (before running) |
| Hypothesis | none — confirmatory replication of a verified published technique (`docs/kaggle/gap-analysis.md`); H1–H4 are closed |
| Notebook | `notebooks/kaggle/k06_full_aggregation/` |
| Predecessor | EXP-004 (the minimal-UID model — this isolates *aggregation richness*, holding model class fixed) |
| Feature diff | Replaces EXP-004's 4-aggregate block with Deotte's verified ~47-feature engine: (a) combine keys `card1_addr1`, `card1_addr1_P_emaildomain`; (b) `aggregate_group` mean/std of `TransactionAmt, D9, D11` at `card1 / card1_addr1 / card1_addr1_P_emaildomain`; (c) at the D1-based `uid`: mean/std of `TransactionAmt, D4, D9, D10, D15`, mean of `C1..C14` (except C3), mean of `M1..M9`, std of `C14`; (d) `aggregate_nunique` of `P_emaildomain, dist1, DT_M, id_02, cents, C13, V314, V127, V136, V309, V307, V320` at `uid`; (e) `outsider15 = (|D1−D15|>3)`; (f) frequency-encode all UIDs. NaN left native (deviation from Deotte's fillna(-1), consistent with our EXP-001+). Model/params unchanged (LightGBM, seed 42). |
| Config | LightGBM identical to EXP-001..004. Aggregations label-free over the train+test union; categorical/UID frequency encoders keep the per-fit discipline. |
| Expected | Confirmatory prediction: closes most of the verified +0.029 private gap to Deotte's single XGB (0.9324). Target private ~0.92–0.93; holdout Δ large vs EXP-004. This is the one experiment that tests "richness, not the key, is the lever." |
| A (holdout AUC) | **0.9421** |
| B (GroupKFold) | **0.9561 ± 0.0121** — per-fold: [0.9320, 0.9602, 0.9608, 0.9571, 0.9666, 0.9461, 0.9702] |
| DeLong vs EXP-004 | ΔAUC **+0.0122** [95% CI +0.0099, +0.0145], z = 10.32, **p = 5.6e-25 — highly significant** |
| LB public / private | **0.9377 / 0.9078** (SUB-006, 2026-07-11) — series best on every metric |
| Verdict / notes | **Confirmatory result: aggregation richness is a real lever internally — but the drift-decay pattern strikes again, and it points to the true missing piece.** The engine (64 features vs EXP-004's 4) lifts the holdout by **+0.0122 (p=5.6e-25)** — the gap analysis was right that aggregation breadth carries large internal signal. **BUT** private LB gained only +0.0046 (0.9078 vs 0.9032), ~1/3 of the internal gain — the same internal-overstates-private pattern as H1/H2. Decisive sub-finding: the **CV−LB gap WIDENED** with more features (\|A−private\| rose 0.0267 → **0.0343**; \|B−private\| 0.0443 → **0.0483**). More features ⇒ more overfit to the training period ⇒ wider gap. **We faithfully replicated Deotte's aggregation engine (even richer: 64 vs 47) yet sit at private 0.9078 vs his 0.9324 — a ~0.025 gap that is NOT aggregation.** The remaining gap is what he (time-consistency selection) and the 2nd place (train/test screening, permutation-importance selection, forward-CV-with-gap) all did and we do not: **drop time-unstable features.** This makes EXP-007 (feature selection for temporal stability) the single highest-value next step — not more features. |

---

## EXP-005 — SKIPPED (numbering note)

EXP-005 was reserved in the roadmap for "consolidation + seed averaging." It was
**intentionally not run**: after EXP-004 the effort was redirected to the verified
winning-solution gap analysis (`docs/kaggle/gap-analysis.md`), which produced the
confirmatory EXP-006 (aggregation engine) as a higher-value use of the slot. The
EXP-005 id is retired, not reused, to keep the experiment ledger append-only and
traceable. Consolidation/seed-averaging, if done, will be a later id.

---

## EXP-007 — Confirmatory: temporal-stability feature selection

| Field | Value |
|---|---|
| Registered | 2026-07-11 (before running) |
| Hypothesis | none — confirmatory replication of the winners' validation/selection discipline (`docs/kaggle/validation-and-selection-playbook.md`) |
| Notebook | `notebooks/kaggle/k07_feature_selection/` |
| Predecessor | EXP-006 (same feature engine — this isolates the *selection* step) |
| Feature diff | No new features. Applies selection to EXP-006's set: (a) **train/test screening** — drop raw high-cardinality ids `card1, card2, card3, card5, addr1, addr2` (keep their frequency encodings); (b) **adversarial validation** — fit a train-vs-test LightGBM, report AUC + top drifting features (diagnostic); (c) **time-consistency filter** (`src/features/selection.py`) — drop features whose signed univariate AUC flips sign or decays to noise between the first and last training month; (d) **seen/unseen-UID holdout segmentation** (diagnostic). Retrain LightGBM (same params) on the surviving features. |
| Config | LightGBM identical to EXP-001..006. Selection computed on the Scheme-A train partition, then applied uniformly. |
| Expected | The decisive test: selection should **shrink the CV−LB gap** (EXP-006: \|A−private\| = 0.0343) and *raise or hold* the private LB while possibly *lowering* the internal holdout — the signature of trading training-period overfit for temporal generalization. Success = private LB ≥ EXP-006's 0.9078 with a smaller CV−LB gap; the seen/unseen split should show the gain concentrated on unseen UIDs. |
| A (holdout AUC) | **0.9383** (EXP-006: 0.9421 — selection *lowered* the holdout) |
| B (GroupKFold) | **0.9557 ± 0.0133** (EXP-006: 0.9561 — flat) |
| DeLong vs EXP-006 | ΔAUC **−0.0039** [95% CI −0.0056, −0.0022], z = −4.51, p = 6.5e-06 — a significant internal *drop* |
| LB public / private | **0.9344 / 0.9077** (SUB-007) — private tied with EXP-006 (0.9078), public lower |
| Verdict / notes | **The "selection is the missing lever" thesis (my post-EXP-006 claim) is REFUTED — and the refutation is the project's key finding.** What selection did, mechanically: removed overfit — holdout −0.0038, and the **CV−LB gap shrank 0.0343 → 0.0306**, exactly the "trade training-period overfit for stability" signature, *without* costing private LB. What it did NOT do: raise the private LB (0.9077 ≈ 0.9078). Why: the **adversarial train-vs-test AUC was 0.5065** — train and test are nearly indistinguishable by these features, so there was barely any drift to select away. Only 25/562 features were unstable (6 raw ids + 19 flip/decay: mostly `id_24/25/30/33/34`, some V and D-norm). **The real ceiling, revealed by the seen/unseen-UID split, is entity linkability:** holdout AUC on clients seen in train = **0.9897**, on new clients = **0.8990**. The private test set is dominated by new clients, so private ≈ the unseen-client ceiling (~0.90). This single diagnostic explains the entire drift-decay pattern across H1–H4 and EXP-006: the internal holdout is optimistic because ~40% of it is memorised known clients; the private LB is mostly new clients. **Corrected conclusion: the gap to Deotte's 0.9324 is not feature selection and not more aggregation — it is better ENTITY RESOLUTION (more/precise UIDs to make more test clients linkable) plus client-level label propagation.** See `docs/kaggle/gap-analysis.md` (corrected). |
