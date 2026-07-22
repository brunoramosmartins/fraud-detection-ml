"""API contract tests for the v2 serving path (ADR-006).

The v2 artifact is a sklearn Pipeline (FeatureBuilderV2 + classifier) whose
predict_proba takes the RAW request columns; the API must pass them through
without imputation (native_nan) so LightGBM's NaN handling stays intact.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from app.main import app
from src.features.pipeline import FeatureBuilderV2


def _make_raw_df() -> pd.DataFrame:
    """Small synthetic frame mimicking the raw IEEE-CIS merge schema."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [3600.0, 90000.0, 180000.0, 260000.0],
            "isFraud": [0, 1, 0, 0],
            "TransactionAmt": [50.0, 117.53, 30.0, np.nan],
            "card1": [1000, 2000, 1000, 3000],
            "addr1": [204.0, 325.0, np.nan, 204.0],
            "D1": [0.0, 10.0, np.nan, 3.0],
            "D9": [0.5, np.nan, 0.25, 0.0],
            "ProductCD": ["W", "C", "W", np.nan],
            "P_emaildomain": ["gmail.com", "mail.co.uk", np.nan, "gmail.com"],
            "R_emaildomain": [np.nan, "yahoo.com", np.nan, np.nan],
            "V1": [0.5, np.nan, 1.5, 0.0],
        }
    )


class RecordingModel(ClassifierMixin, BaseEstimator):
    """Stub classifier that records the matrix it receives.

    A real sklearn estimator (BaseEstimator + fitted ``classes_``) because
    Pipeline.predict_proba validates the final step via check_is_fitted
    and the estimator-tags machinery.
    """

    def __init__(self) -> None:
        self.received = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        self.received = X.copy()
        return np.tile([0.9, 0.1], (len(X), 1))


def _v2_state(app_instance, clf) -> FeatureBuilderV2:
    """Inject a fitted v2 artifact (builder + clf pipeline) into app.state."""
    builder = FeatureBuilderV2().fit(_make_raw_df())
    app_instance.state.model = Pipeline(steps=[("features", builder), ("clf", clf)])
    app_instance.state.feature_list = builder.input_columns_
    app_instance.state.threshold = 0.5
    app_instance.state.model_name = "lgbm"
    app_instance.state.model_version = "v2"
    app_instance.state.native_nan = True
    return builder


def _raw_transaction(**overrides) -> dict:
    tx = {
        "TransactionID": 10,
        "TransactionDT": 90000.0,
        "TransactionAmt": 117.53,
        "card1": 2000,
        "addr1": 325.0,
        "D1": 10.0,
        "D9": None,
        "ProductCD": "C",
        "P_emaildomain": "mail.co.uk",
        "R_emaildomain": "yahoo.com",
        "V1": None,
    }
    tx.update(overrides)
    return tx


@pytest.fixture()
def v2_client(monkeypatch):
    clf = RecordingModel()
    holder = {}

    def loader(app_instance):
        holder["builder"] = _v2_state(app_instance, clf)

    monkeypatch.setattr("app.main._load_deployed_model", loader)
    with TestClient(app) as c:
        yield c, clf, holder["builder"]


def test_v2_predict_scores_raw_transaction(v2_client):
    client, clf, builder = v2_client
    response = client.post("/predict", json={"transactions": [_raw_transaction()]})
    assert response.status_code == 200

    data = response.json()
    assert data["model_name"] == "lgbm"
    assert data["model_version"] == "v2"
    pred = data["predictions"][0]
    assert pred["fraud_probability"] == pytest.approx(0.1)
    # The classifier received the ENGINEERED matrix, in contract order
    assert list(clf.received.columns) == builder.feature_list_


def test_v2_nan_reaches_model_unimputed(v2_client):
    """native_nan: the API must not fillna(0) — LightGBM handles NaN itself."""
    client, clf, _ = v2_client
    response = client.post(
        "/predict", json={"transactions": [_raw_transaction(V1=None)]}
    )
    assert response.status_code == 200
    assert np.isnan(clf.received["V1"].iloc[0])


def test_v2_unseen_category_scores_without_error(v2_client):
    client, clf, _ = v2_client
    response = client.post(
        "/predict",
        json={"transactions": [_raw_transaction(ProductCD="NEVER_SEEN")]},
    )
    assert response.status_code == 200
    assert clf.received["ProductCD_le"].iloc[0] == -1
    assert clf.received["ProductCD_freq"].iloc[0] == 0.0


def test_v2_missing_raw_column_returns_422(v2_client):
    client, _, _ = v2_client
    tx = _raw_transaction()
    del tx["card1"]
    response = client.post("/predict", json={"transactions": [tx]})
    assert response.status_code == 422
    assert "card1" in response.json()["detail"]


def test_v1_path_still_imputes_zero(monkeypatch):
    """Without native_nan, the legacy fillna(0) contract is preserved."""
    clf = RecordingModel()

    def loader(app_instance):
        app_instance.state.model = clf
        app_instance.state.feature_list = ["TransactionAmt", "V1"]
        app_instance.state.threshold = 0.5
        app_instance.state.model_name = "gb"
        app_instance.state.model_version = "v1"
        app_instance.state.native_nan = False

    monkeypatch.setattr("app.main._load_deployed_model", loader)
    with TestClient(app) as c:
        response = c.post(
            "/predict",
            json={"transactions": [{"TransactionAmt": 50.0, "V1": None}]},
        )
    assert response.status_code == 200
    assert clf.received["V1"].iloc[0] == 0.0
