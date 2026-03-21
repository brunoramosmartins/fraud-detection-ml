from typing import Dict, Iterable

import pandas as pd

REQUIRED_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
}

NO_MISSING_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "isFraud",
}

EXPECTED_DTYPES: Dict[str, str] = {
    "TransactionID": "int64",
    # In the raw dataset TransactionDT is typically an integer-like time index.
    # We only require it to be stored as 64-bit integer here.
    "TransactionDT": "int64",
    "TransactionAmt": "float64",
    "isFraud": "int64",
}


def _check_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _check_no_missing(df: pd.DataFrame, columns: Iterable[str]) -> None:
    cols_with_na = [c for c in columns if df[c].isna().any()]
    if cols_with_na:
        raise ValueError(f"Columns with missing values but expected none: {cols_with_na}")


def _check_dtypes(df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> None:
    wrong = {}
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            continue
        if str(df[col].dtype) != expected:
            wrong[col] = f"{df[col].dtype} != {expected}"
    if wrong:
        raise ValueError(f"Unexpected dtypes: {wrong}")


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate basic schema constraints for the IEEE-CIS dataset.

    Raises ValueError if invalid.
    """
    _check_required_columns(df, REQUIRED_COLUMNS)
    _check_no_missing(df, NO_MISSING_COLUMNS)
    _check_dtypes(df, EXPECTED_DTYPES)

    # Simple sanity check on target distribution.
    fraud_rate = df["isFraud"].mean()
    if not 0.0 < fraud_rate < 0.5:
        raise ValueError(f"Unexpected fraud rate: {fraud_rate:.4f}")

