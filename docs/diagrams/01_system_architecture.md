# System Architecture Diagrams

---

## 1. High-Level System Topology

End-to-end view of all components and their relationships. Arrows represent
data flow; labels describe the artifact or protocol at each boundary.

```mermaid
graph TB
    subgraph Offline["Offline — Batch Training"]
        RAW[("data/raw/\ntrain_transaction.csv\ntrain_identity.csv")]
        SCHEMA["Schema Validation\nsrc/data/schema.py"]
        SPLIT["Temporal Split\nsrc/data/split.py\n(quantile=0.80)"]
        FEAT_TR["Feature Pipeline\nsrc/features/pipeline.py"]
        TRAIN["Model Training\nsrc/models/factory.py"]
        METRICS["Metric Computation\nsrc/models/metrics.py\nEML · ROC-AUC · PR-AUC"]
        ARTIFACT[("artifacts/models/\ngb_v1_<ts>.pkl\ngb_v1_<ts>_meta.json")]
        RUN[("artifacts/runs/\nrun_<ts>.json")]
    end

    subgraph Online["Online — Inference Service"]
        API["FastAPI\napp/main.py\nPOST /predict\nGET /health"]
        STATE["app.state\nmodel · feature_list · threshold"]
    end

    subgraph Ops["Operational — Simulation & Monitoring"]
        SIM["Transaction Simulator\nscripts/simulate_transactions.py"]
        PREDS[("artifacts/monitoring/\npredictions/predictions_<ts>.csv")]
        MON["Monitor\nscripts/monitor_model.py\nPSI · EML tracking"]
        DRIFT[("artifacts/monitoring/\ndrift/drift_report_<ts>.json")]
        PERF[("artifacts/monitoring/\nperformance/perf_report_<ts>.json")]
        RETRAIN["Retraining Trigger\nscripts/retrain_model.py\nmax_psi > threshold"]
    end

    RAW --> SCHEMA --> SPLIT
    SPLIT -->|X_train| FEAT_TR
    SPLIT -->|X_val + y_val + amounts| METRICS
    FEAT_TR -->|X + feature_list| TRAIN
    TRAIN -->|fitted model| METRICS
    METRICS -->|best_threshold| ARTIFACT
    TRAIN --> ARTIFACT
    ARTIFACT --> RUN

    ARTIFACT -->|startup load| STATE
    STATE --> API

    SIM -->|POST /predict\nbatch JSON| API
    API -->|predictions| SIM
    SIM --> PREDS

    PREDS --> MON
    RAW -->|reference| MON
    MON --> DRIFT
    MON --> PERF

    DRIFT -->|max_psi| RETRAIN
    RETRAIN -->|triggers| TRAIN
```

---

## 2. `src/` Module Dependencies

Component-level view of the `src/` package showing which modules depend on
which, and what each module's public interface is.

```mermaid
graph LR
    subgraph data["src/data/"]
        LOADER["loader.py\nload_full_training_dataset()"]
        SCHEMA["schema.py\nvalidate_schema()"]
        SPLIT2["split.py\ntemporal_train_val_split()"]
    end

    subgraph features["src/features/"]
        REGISTRY["feature_registry.py\nget_feature_list()"]
        PIPELINE["pipeline.py\nbuild_features()"]
    end

    subgraph models["src/models/"]
        FACTORY["factory.py\nget_model()"]
        METRICS2["metrics.py\ncompute_classification_metrics()\nexpected_loss()\nthreshold_sweep()"]
        ARTIFACTS2["artifacts.py\nsave_model_artifact()"]
    end

    subgraph utils["src/utils/"]
        CONFIG["config.py\nDATA_PATH · MODELS_DIR\nC_FP · RANDOM_STATE"]
        TRACKING["tracking.py\nstart_run() · log_metrics()\nend_run()"]
        DRIFT2["drift.py\ncompute_psi()"]
    end

    subgraph pipelines["src/pipelines/"]
        TPIPELINE["training_pipeline.py\nrun_training_pipeline()"]
    end

    TPIPELINE --> LOADER
    TPIPELINE --> SCHEMA
    TPIPELINE --> SPLIT2
    TPIPELINE --> PIPELINE
    TPIPELINE --> FACTORY
    TPIPELINE --> METRICS2
    TPIPELINE --> ARTIFACTS2
    TPIPELINE --> CONFIG
    TPIPELINE --> TRACKING

    PIPELINE --> REGISTRY
    PIPELINE --> CONFIG
    METRICS2 --> CONFIG
```

---

## 3. POST /predict Request Lifecycle

Sequence diagram for a single batch scoring request, including the happy
path and the two error paths (model absent, missing features).

```mermaid
sequenceDiagram
    participant C as Client
    participant API as FastAPI app/main.py
    participant STATE as app.state
    participant DF as pandas DataFrame

    C->>API: POST /predict {transactions: [...]}
    API->>STATE: read model, feature_list, threshold

    alt model is None
        API-->>C: HTTP 503 "Model not loaded"
    else feature_list is None
        API-->>C: HTTP 503 "Feature list not available"
    end

    API->>DF: pd.DataFrame(t.model_dump() for t in transactions)
    API->>DF: check missing = [c for c in feature_list if c not in df.columns]

    alt missing columns detected
        API-->>C: HTTP 422 "Missing required features: [...]"
    end

    API->>DF: df[feature_list].fillna(0.0)
    API->>STATE: model.predict_proba(X)[:, 1]
    STATE-->>API: proba array
    API->>API: fraud_flag = proba >= threshold
    API-->>C: HTTP 200 {model_name, threshold, predictions: [{transaction_id, fraud_probability, fraud_flag}]}
    note over API: logger.info("Scored N transactions flagged=K")
```

---

## 4. API Startup Sequence

What happens between `uvicorn app.main:app` and the first request being
served successfully.

```mermaid
sequenceDiagram
    participant UV as uvicorn
    participant APP as FastAPI lifespan
    participant FS as artifacts/models/
    participant STATE as app.state

    UV->>APP: enter lifespan context
    APP->>APP: _load_deployed_model(app)
    APP->>FS: glob("gb_v1_*.pkl") → sorted, take last
    alt no files found
        APP-->>UV: raise RuntimeError → startup fails
        UV-->>UV: exit with error
    end
    APP->>FS: joblib.load(model_path)
    FS-->>APP: fitted GradientBoostingClassifier
    APP->>FS: read _meta.json → feature_list, threshold
    APP->>STATE: model, feature_list, threshold, model_name, model_version
    note over APP: logger.info("Model loaded: gb_v1_<ts>.pkl threshold=0.0200")
    APP-->>UV: lifespan yield → server ready
    UV->>UV: Uvicorn running on http://0.0.0.0:8000
```

---

## 5. Training Pipeline Data Flow

Detailed view of how data moves through the training pipeline, from raw
CSV to serialized artifact.

```mermaid
flowchart TD
    A[/"data/raw/train_transaction.csv\ndata/raw/train_identity.csv"/]
    B["load_full_training_dataset()\n→ merge on TransactionID\n→ ~590k rows × ~400 cols"]
    C["validate_schema()\n→ check required columns\n→ check dtypes\n→ check fraud rate 0 < r < 0.5"]
    D["temporal_train_val_split(split_quantile=0.80)\n→ cutoff = TransactionDT.quantile(0.80)\n→ train: rows before cutoff\n→ val:   rows after cutoff"]

    E1["X_train_raw\n~80% of rows"]
    E2["X_val_raw + y_val + val_amount\n~20% of rows"]

    F1["build_features(X_train_raw)\n→ get_feature_list(): all numeric cols\n   except isFraud, TransactionID, TransactionDT\n→ fillna(0.0)\n→ returns X_train (N×380), feature_list"]
    F2["build_features(X_val_raw)\n→ same feature_list\n→ fillna(0.0)"]

    G["get_model('gb', config)\n→ GradientBoostingClassifier\n   n_estimators=80, max_depth=5\n   learning_rate=0.1, subsample=0.8\n   min_samples_leaf=100"]
    H["model.fit(X_train, y_train)"]
    I["model.predict_proba(X_val)[:, 1]\n→ val_proba"]

    J["compute_classification_metrics(y_val, val_proba, val_amount, c_fp=5.0)\n→ roc_auc, pr_auc\n→ baseline_loss = Σ(fraud × amount)\n→ threshold_sweep [0.01..0.99]\n→ best_threshold=0.02, best_loss\n→ precision=TP/(TP+FP), fpr=FP/(FP+TN)"]

    K[/"artifacts/models/gb_v1_<ts>.pkl\nartifacts/models/gb_v1_<ts>_meta.json\n(feature_list + metrics + config_file)"/]
    L[/"artifacts/runs/run_<ts>.json\n(run_id, metrics, artifact_paths, timestamps)"/]

    A-->B-->C-->D
    D-->E1 & E2
    E1-->F1-->G-->H-->I
    E2-->F2-->I
    I-->J
    J-->K
    H-->K
    K-->L
```
