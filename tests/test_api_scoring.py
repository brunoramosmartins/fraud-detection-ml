import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, _load_deployed_model


class DummyModel:
    """Stub model that always returns fraud_probability=0.1 (non-fraud)."""

    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (len(X), 1))


class DummyModelHighRisk:
    """Stub model that always returns fraud_probability=0.95 (fraud)."""

    def predict_proba(self, X):
        return np.tile([0.05, 0.95], (len(X), 1))


def _inject_state(app_instance, model=None, feature_list=None, threshold=0.5):
    app_instance.state.model = model or DummyModel()
    app_instance.state.feature_list = feature_list if feature_list is not None else ["TransactionAmt"]
    app_instance.state.threshold = threshold
    app_instance.state.model_name = "gb"
    app_instance.state.model_version = "v1"


@pytest.fixture()
def client(monkeypatch):
    """
    TestClient with the lifespan artifact loading replaced by a lightweight stub.
    Monkeypatching _load_deployed_model before the TestClient context manager
    ensures the lifespan does not attempt to read real artifact files.
    """
    monkeypatch.setattr(
        "app.main._load_deployed_model",
        lambda app_instance: _inject_state(app_instance),
    )
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_predict_single_transaction(client):
    payload = {
        "transactions": [
            {"TransactionID": 1, "TransactionDT": 12345.0, "TransactionAmt": 100.0}
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1

    pred = data["predictions"][0]
    assert pred["fraud_probability"] == pytest.approx(0.1)
    assert pred["fraud_flag"] is False  # 0.1 < threshold 0.5


def test_predict_batch_returns_correct_count(client):
    payload = {
        "transactions": [
            {"TransactionID": i, "TransactionAmt": float(i * 10)} for i in range(1, 6)
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 5


def test_predict_high_risk_model_flags_fraud(monkeypatch):
    monkeypatch.setattr(
        "app.main._load_deployed_model",
        lambda app_instance: _inject_state(app_instance, model=DummyModelHighRisk()),
    )
    with TestClient(app) as c:
        payload = {"transactions": [{"TransactionID": 99, "TransactionAmt": 500.0}]}
        response = c.post("/predict", json=payload)
    assert response.status_code == 200
    pred = response.json()["predictions"][0]
    assert pred["fraud_probability"] == pytest.approx(0.95)
    assert pred["fraud_flag"] is True  # 0.95 >= threshold 0.5


def test_health_endpoint_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_predict_missing_required_feature(monkeypatch):
    """Payload missing a feature that the model contract requires → 422."""
    monkeypatch.setattr(
        "app.main._load_deployed_model",
        lambda app_instance: _inject_state(
            app_instance, feature_list=["TransactionAmt", "required_extra"]
        ),
    )
    with TestClient(app) as c:
        payload = {"transactions": [{"TransactionID": 1, "TransactionAmt": 100.0}]}
        response = c.post("/predict", json=payload)
    assert response.status_code == 422
    assert "required_extra" in response.json()["detail"]


def test_predict_model_not_loaded(monkeypatch):
    """If the model is absent from state, the API must return 503."""
    monkeypatch.setattr(
        "app.main._load_deployed_model",
        lambda app_instance: _inject_state(app_instance, model=None),
    )
    with TestClient(app) as c:
        # Force state to unloaded after lifespan runs
        app.state.model = None
        payload = {"transactions": [{"TransactionID": 1, "TransactionAmt": 100.0}]}
        response = c.post("/predict", json=payload)
    assert response.status_code == 503
