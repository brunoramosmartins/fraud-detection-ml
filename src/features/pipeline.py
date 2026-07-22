from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.features.engineering import (
    add_amount_features,
    add_time_features,
    apply_frequency_table,
    apply_label_map,
    fit_frequency_table,
    fit_label_map,
    normalize_d_columns,
    split_email_domain,
)
from src.features.feature_registry import (
    V2_D_NORM_COLS,
    V2_EMAIL_SPLITS,
    V2_FREQ_NUMERIC_CATS,
    get_feature_list,
)

# Columns never used as features (mirrors infer_numeric_feature_list).
_EXCLUDE_COLS = {"isFraud", "TransactionID", "TransactionDT"}


def build_features(df: pd.DataFrame, feature_set: str = "v1") -> Tuple[pd.DataFrame, List[str]]:
    """
    Build feature matrix X for the given feature set.

    Responsibilities:
    - select columns from the feature registry
    - apply simple imputations (fillna(0))
    - return X and the ordered feature list actually used

    Note: v1 only. Feature set "v2" carries fitted encoder state and is
    built with :class:`FeatureBuilderV2` instead (ADR-006).
    """
    feature_list = get_feature_list(df, feature_set=feature_set)
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from DataFrame: {missing[:10]}")
    X = df[feature_list].copy()
    X = X.fillna(0.0)
    return X, feature_list


class FeatureBuilderV2:
    """Fit/transform builder for feature set "v2" (ADR-006).

    Builds the serving-feasible subset of the Phase 8 Kaggle blocks — the
    EXP-003 configuration:

    - numeric base (native NaN, no imputation — the model is LightGBM)
    - email-domain provider/suffix split (row-local)
    - label encoding of every object-dtype categorical (frozen dictionaries)
    - frequency encoding of the same columns plus ``V2_FREQ_NUMERIC_CATS``
      (frozen tables)
    - time features (``tx_hour``, ``tx_dow``) and amount features
      (``amt_log1p``, ``amt_cents``) — row-local
    - D-column normalization (``D{n}_norm``) — row-local

    ``fit`` derives the column lists and freezes the encoder tables from
    training rows only; ``transform`` is then computable for any rows —
    a validation fold, the holdout, or a single serving-time transaction —
    using exclusively (a) the row's own values and (b) the frozen state.
    That property is the ADR-006 boundary condition, and it makes the
    fitted builder serializable inside the model artifact via joblib
    (plain dicts and lists only).

    Attributes (set by fit):
        numeric_features_: Ordered numeric base columns.
        label_cols_: Object-dtype columns that get label encoding.
        freq_cols_: Columns that get frequency encoding.
        label_maps_: Frozen category -> code dicts per label column.
        freq_tables_: Frozen category -> frequency dicts per freq column.
        feature_list_: Ordered output columns of ``transform``.
        input_columns_: Raw columns ``transform`` requires (API contract).
    """

    def __init__(
        self,
        email_splits: Optional[List[Tuple[str, str]]] = None,
        freq_numeric_cats: Optional[List[str]] = None,
        d_norm_cols: Optional[List[str]] = None,
    ) -> None:
        self.email_splits = list(V2_EMAIL_SPLITS if email_splits is None else email_splits)
        self.freq_numeric_cats = list(
            V2_FREQ_NUMERIC_CATS if freq_numeric_cats is None else freq_numeric_cats
        )
        self.d_norm_cols = list(V2_D_NORM_COLS if d_norm_cols is None else d_norm_cols)

    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilderV2":
        """Derive column lists and freeze encoder state from training rows.

        Args:
            train_df: Raw training partition (transaction + identity merge).
                Only these rows enter the encoder tables — the fit/transform
                boundary is the leakage control.

        Returns:
            self, fitted.
        """
        df = self._with_email_splits(train_df)

        self.numeric_features_ = [
            c for c in df.columns if df[c].dtype != "O" and c not in _EXCLUDE_COLS
        ]
        self.label_cols_ = [c for c in df.columns if df[c].dtype == "O"]
        self.freq_cols_ = self.label_cols_ + [
            c for c in self.freq_numeric_cats if c in df.columns
        ]

        self.label_maps_: Dict[str, Dict[str, int]] = {
            col: fit_label_map(df[col]) for col in self.label_cols_
        }
        self.freq_tables_: Dict[str, Dict[str, float]] = {
            col: fit_frequency_table(df[col]) for col in self.freq_cols_
        }

        self.d_norm_cols_ = [c for c in self.d_norm_cols if c in df.columns]

        self.feature_list_ = (
            list(self.numeric_features_)
            + ["tx_hour", "tx_dow", "amt_log1p", "amt_cents"]
            + [f"{c}_norm" for c in self.d_norm_cols_]
            + [f"{c}_le" for c in self.label_cols_]
            + [f"{c}_freq" for c in self.freq_cols_]
        )

        email_outputs = {
            f"{p}_{part}" for _, p in self.email_splits for part in ("provider", "suffix")
        }
        raw_needed = set(self.numeric_features_) - email_outputs
        raw_needed |= {src for src, _ in self.email_splits}
        raw_needed |= set(self.label_cols_) - email_outputs
        raw_needed |= set(self.freq_cols_) - email_outputs
        raw_needed |= set(self.d_norm_cols_)
        raw_needed |= {"TransactionDT", "TransactionAmt"}
        self.input_columns_ = sorted(raw_needed)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the v2 feature matrix for any rows (fail-fast, ADR-004).

        Args:
            df: Raw rows to featurize. Must contain ``input_columns_``.

        Returns:
            DataFrame with exactly ``feature_list_`` columns, in order.
            NaNs are preserved (native handling by LightGBM) — no
            imputation happens here.
        """
        if not hasattr(self, "feature_list_"):
            raise RuntimeError("FeatureBuilderV2.transform called before fit")
        missing = [c for c in self.input_columns_ if c not in df.columns]
        if missing:
            raise ValueError(f"Features missing from DataFrame: {missing[:10]}")

        work = self._with_email_splits(df)

        out = work.reindex(columns=self.numeric_features_).astype("float32")
        out = pd.concat(
            [
                out,
                add_time_features(work["TransactionDT"]),
                add_amount_features(work["TransactionAmt"]),
                normalize_d_columns(work, work["TransactionDT"], self.d_norm_cols_),
            ],
            axis=1,
        )
        for col in self.label_cols_:
            out[f"{col}_le"] = apply_label_map(self.label_maps_[col], work[col])
        for col in self.freq_cols_:
            out[f"{col}_freq"] = apply_frequency_table(self.freq_tables_[col], work[col])

        return out[self.feature_list_]

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``train_df`` and return its transformed matrix."""
        return self.fit(train_df).transform(train_df)

    def _with_email_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with provider/suffix columns appended (row-local)."""
        parts = [
            split_email_domain(df[src], prefix)
            for src, prefix in self.email_splits
            if src in df.columns
        ]
        if not parts:
            return df
        return pd.concat([df] + parts, axis=1)
