import numpy as np
import pandas as pd
import pytest

from src.features.feature_registry import get_feature_list
from src.features.pipeline import FeatureBuilderV2
from src.models.factory import get_model


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


def _fitted_builder() -> FeatureBuilderV2:
    return FeatureBuilderV2().fit(_make_raw_df())


def test_registry_v2_rejects_static_list():
    with pytest.raises(ValueError, match="FeatureBuilderV2"):
        get_feature_list(_make_raw_df(), feature_set="v2")


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        FeatureBuilderV2().transform(_make_raw_df())


def test_output_matches_feature_list_and_order():
    builder = _fitted_builder()
    X = builder.transform(_make_raw_df())
    assert list(X.columns) == builder.feature_list_
    # Engineered blocks are present
    for col in ("tx_hour", "tx_dow", "amt_log1p", "amt_cents", "D1_norm"):
        assert col in X.columns
    # Excluded columns never leak into the matrix
    for col in ("isFraud", "TransactionID", "TransactionDT"):
        assert col not in X.columns
    # D9 is an hour fraction, not a day counter — never normalized
    assert "D9_norm" not in X.columns


def test_native_nan_preserved():
    """v2 must NOT impute — LightGBM handles NaN natively (ADR-006/007)."""
    builder = _fitted_builder()
    X = builder.transform(_make_raw_df())
    assert np.isnan(X.loc[1, "V1"])
    assert np.isnan(X.loc[2, "D1_norm"])


def test_missing_input_column_fails_fast():
    """ADR-004: fail-fast on missing columns, matching build_features."""
    builder = _fitted_builder()
    with pytest.raises(ValueError, match="Features missing"):
        builder.transform(_make_raw_df().drop(columns=["card1"]))


def test_unseen_category_maps_to_sentinels():
    builder = _fitted_builder()
    new = _make_raw_df()
    new.loc[0, "ProductCD"] = "H"  # never seen in training
    X = builder.transform(new)
    assert X.loc[0, "ProductCD_le"] == -1
    assert X.loc[0, "ProductCD_freq"] == 0.0


def test_encoders_fit_on_train_only():
    """Frozen tables must reflect the fit partition, not the scored rows."""
    builder = _fitted_builder()
    other = _make_raw_df()
    other["ProductCD"] = ["C", "C", "C", "C"]
    X = builder.transform(other)
    # freq of "C" in training was 1/4 regardless of the scored rows
    assert X["ProductCD_freq"].to_numpy() == pytest.approx(0.25)


def test_email_split_features_present_and_encoded():
    builder = _fitted_builder()
    X = builder.transform(_make_raw_df())
    assert "P_email_provider_le" in X.columns
    assert "P_email_suffix_freq" in X.columns
    # gmail.com rows share one provider code; mail.co.uk differs
    assert X.loc[0, "P_email_provider_le"] == X.loc[3, "P_email_provider_le"]
    assert X.loc[0, "P_email_provider_le"] != X.loc[1, "P_email_provider_le"]


def test_transform_is_deterministic_across_calls():
    builder = _fitted_builder()
    a = builder.transform(_make_raw_df())
    b = builder.transform(_make_raw_df())
    pd.testing.assert_frame_equal(a, b)


def test_single_row_scoring_contract():
    """The ADR-006 boundary condition: one raw row is transformable."""
    builder = _fitted_builder()
    row = _make_raw_df().iloc[[1]]
    X = builder.transform(row)
    assert X.shape == (1, len(builder.feature_list_))


def test_input_columns_cover_transform_needs():
    builder = _fitted_builder()
    raw = _make_raw_df()
    X = builder.transform(raw[[c for c in raw.columns if c in builder.input_columns_]])
    assert list(X.columns) == builder.feature_list_


def test_builder_joblib_roundtrip(tmp_path):
    """Frozen encoder state must survive artifact serialization (ADR-002/006)."""
    joblib = pytest.importorskip("joblib")
    builder = _fitted_builder()
    path = tmp_path / "builder.pkl"
    joblib.dump(builder, path)
    restored = joblib.load(path)
    pd.testing.assert_frame_equal(
        builder.transform(_make_raw_df()), restored.transform(_make_raw_df())
    )


def test_factory_lgbm_entry_defaults():
    lightgbm = pytest.importorskip("lightgbm")
    model = get_model("lgbm", config={})
    assert isinstance(model, lightgbm.LGBMClassifier)
    params = model.get_params()
    # EXP-003 configuration (ADR-006/007)
    assert params["learning_rate"] == 0.05
    assert params["num_leaves"] == 192
    assert params["min_data_in_leaf"] == 100
    assert params["feature_fraction"] == 0.8
    assert params["random_state"] == 42


def test_factory_lgbm_config_overrides():
    pytest.importorskip("lightgbm")
    model = get_model("lgbm", config={"lgbm": {"n_estimators": 7, "num_leaves": 31}})
    params = model.get_params()
    assert params["n_estimators"] == 7
    assert params["num_leaves"] == 31
