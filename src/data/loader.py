from pathlib import Path
from typing import Optional

import pandas as pd


def load_transaction(path: Path) -> pd.DataFrame:
    """Load transaction table."""
    return pd.read_csv(path)


def load_identity(path: Path) -> pd.DataFrame:
    """Load identity table."""
    return pd.read_csv(path)


def merge_transaction_identity(
    transaction: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    """Merge transaction and identity tables on TransactionID."""
    return transaction.merge(identity, on="TransactionID", how="left")


def load_full_training_dataset(
    base_path: Path,
    transaction_filename: str = "train_transaction.csv",
    identity_filename: str = "train_identity.csv",
) -> pd.DataFrame:
    """
    Load and merge the full IEEE-CIS training dataset.

    Parameters
    ----------
    base_path:
        Directory containing the raw CSV files.
    """
    trans_path = base_path / transaction_filename
    id_path = base_path / identity_filename

    transaction = load_transaction(trans_path)
    identity: Optional[pd.DataFrame]
    if id_path.exists():
        identity = load_identity(id_path)
        df = merge_transaction_identity(transaction, identity)
    else:
        df = transaction

    return df

