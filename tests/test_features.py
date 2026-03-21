import numpy as np
import pandas as pd

from src.features.pipeline import build_features


def _make_sample_df():
    """Create a small synthetic DataFrame mimicking the IEEE-CIS schema."""
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 200, 300],
            "isFraud": [0, 1, 0],
            "TransactionAmt": [50.0, 100.0, np.nan],
            "card1": [1000, 2000, 3000],
            "V1": [0.5, np.nan, 1.5],
        }
    )


def test_build_features_drops_excluded_columns():
    df = _make_sample_df()
    X, feature_list = build_features(df)
    assert "isFraud" not in feature_list
    assert "TransactionID" not in feature_list
    assert "TransactionDT" not in feature_list


def test_build_features_fills_nan_with_zero():
    df = _make_sample_df()
    X, _ = build_features(df)
    assert not X.isna().any().any()
    # TransactionAmt was NaN in row 3, V1 was NaN in row 2
    assert X.loc[df.index[2], "TransactionAmt"] == 0.0
    assert X.loc[df.index[1], "V1"] == 0.0


def test_build_features_returns_correct_shape():
    df = _make_sample_df()
    X, feature_list = build_features(df)
    assert X.shape == (3, len(feature_list))
    assert len(feature_list) > 0
