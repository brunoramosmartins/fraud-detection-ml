import pandas as pd

from src.data.loader import merge_transaction_identity


def test_merge_transaction_identity_left_join():
    transaction = pd.DataFrame(
        {"TransactionID": [1, 2, 3], "amount": [10.0, 20.0, 30.0]}
    )
    identity = pd.DataFrame(
        {"TransactionID": [1, 3], "device": ["mobile", "desktop"]}
    )
    merged = merge_transaction_identity(transaction, identity)

    assert len(merged) == 3
    assert "device" in merged.columns
    # Row 2 has no identity match → NaN
    assert pd.isna(merged.loc[merged["TransactionID"] == 2, "device"].iloc[0])


def test_merge_preserves_all_transaction_rows():
    transaction = pd.DataFrame(
        {"TransactionID": [10, 20], "amount": [5.0, 15.0]}
    )
    identity = pd.DataFrame({"TransactionID": [], "device": []})
    merged = merge_transaction_identity(transaction, identity)

    assert len(merged) == 2
    assert list(merged["TransactionID"]) == [10, 20]
