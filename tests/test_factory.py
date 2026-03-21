import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.factory import get_model


def test_get_model_lr_returns_pipeline():
    model = get_model("lr", {"lr": {"C": 0.5}})
    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "clf" in model.named_steps


def test_get_model_rf_returns_random_forest():
    model = get_model("rf", {"rf": {"n_estimators": 10}})
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10


def test_get_model_gb_returns_gradient_boosting():
    model = get_model("gb", {"gb": {"n_estimators": 20, "max_depth": 3}})
    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 20
    assert model.max_depth == 3


def test_get_model_case_insensitive():
    model = get_model("GB", {"gb": {}})
    assert isinstance(model, GradientBoostingClassifier)


def test_get_model_invalid_name_raises():
    with pytest.raises(ValueError, match="Unsupported model_name"):
        get_model("xgb", {})


def test_get_model_uses_defaults_when_params_missing():
    model = get_model("gb", {})
    assert isinstance(model, GradientBoostingClassifier)
    assert model.n_estimators == 80
    assert model.max_depth == 5
