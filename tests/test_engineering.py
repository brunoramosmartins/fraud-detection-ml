"""Tests for the categorical encoding blocks (src/features/engineering.py)."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    MISSING_TOKEN,
    build_categorical_block,
    frequency_encode,
    label_encode,
    split_email_domain,
)


@pytest.fixture()
def train_col():
    return pd.Series(["a", "a", "a", "b", np.nan, "b", "c", "a"])


class TestFrequencyEncode:
    def test_frequencies_computed_on_train_only(self, train_col):
        scoring = pd.Series(["a", "b", "c"])
        encoded = frequency_encode(train_col, scoring)
        assert encoded.iloc[0] == pytest.approx(4 / 8)  # 'a' appears 4/8 in train
        assert encoded.iloc[1] == pytest.approx(2 / 8)
        assert encoded.iloc[2] == pytest.approx(1 / 8)

    def test_unseen_category_maps_to_zero(self, train_col):
        encoded = frequency_encode(train_col, pd.Series(["zzz"]))
        assert encoded.iloc[0] == 0.0

    def test_missing_is_its_own_category(self, train_col):
        encoded = frequency_encode(train_col, pd.Series([np.nan]))
        assert encoded.iloc[0] == pytest.approx(1 / 8)  # one NaN in train

    def test_scoring_distribution_does_not_affect_encoding(self, train_col):
        skewed = pd.Series(["c"] * 100)
        encoded = frequency_encode(train_col, skewed)
        assert (encoded == pytest.approx(1 / 8)).all()

    def test_dtype_is_float32(self, train_col):
        assert frequency_encode(train_col, train_col).dtype == np.float32


class TestLabelEncode:
    def test_mapping_is_deterministic_and_sorted(self, train_col):
        encoded = label_encode(train_col, pd.Series(["a", "b", "c"]))
        # sorted unique train categories: __missing__, a, b, c
        assert encoded.tolist() == [1, 2, 3]

    def test_unseen_category_maps_to_minus_one(self, train_col):
        encoded = label_encode(train_col, pd.Series(["zzz"]))
        assert encoded.iloc[0] == -1

    def test_missing_seen_in_train_gets_a_code(self, train_col):
        encoded = label_encode(train_col, pd.Series([np.nan]))
        assert encoded.iloc[0] == 0  # MISSING_TOKEN sorts first

    def test_missing_unseen_in_train_maps_to_minus_one(self):
        train = pd.Series(["a", "b"])
        encoded = label_encode(train, pd.Series([np.nan]))
        assert encoded.iloc[0] == -1

    def test_dtype_is_int32(self, train_col):
        assert label_encode(train_col, train_col).dtype == np.int32

    def test_numeric_categories_are_stable(self):
        train = pd.Series([1111.0, 2222.0, np.nan])
        encoded = label_encode(train, pd.Series([2222.0, 3333.0]))
        assert encoded.iloc[0] >= 0
        assert encoded.iloc[1] == -1


class TestSplitEmailDomain:
    def test_two_part_domain(self):
        out = split_email_domain(pd.Series(["gmail.com"]), "P")
        assert out.loc[0, "P_provider"] == "gmail"
        assert out.loc[0, "P_suffix"] == "com"

    def test_multi_part_domain(self):
        out = split_email_domain(pd.Series(["mail.co.uk"]), "P")
        assert out.loc[0, "P_provider"] == "mail"
        assert out.loc[0, "P_suffix"] == "uk"

    def test_missing_yields_missing_token(self):
        out = split_email_domain(pd.Series([np.nan]), "P")
        assert out.loc[0, "P_provider"] == MISSING_TOKEN
        assert out.loc[0, "P_suffix"] == MISSING_TOKEN

    def test_index_preserved(self):
        s = pd.Series(["gmail.com", "yahoo.com"], index=[10, 20])
        out = split_email_domain(s, "R")
        assert list(out.index) == [10, 20]


class TestBuildCategoricalBlock:
    def test_column_names_and_shape(self):
        train = pd.DataFrame({"cat": ["a", "b"], "card": [1.0, 2.0]})
        block = build_categorical_block(
            train, train, label_cols=["cat"], freq_cols=["cat", "card"]
        )
        assert sorted(block.columns) == ["card_freq", "cat_freq", "cat_le"]
        assert len(block) == 2

    def test_fit_on_train_apply_on_test(self):
        train = pd.DataFrame({"cat": ["a", "a", "b"]})
        test = pd.DataFrame({"cat": ["b", "zzz"]}, index=[100, 101])
        block = build_categorical_block(train, test, ["cat"], ["cat"])
        assert block.loc[101, "cat_le"] == -1
        assert block.loc[101, "cat_freq"] == 0.0
        assert list(block.index) == [100, 101]

    def test_only_new_columns_returned(self):
        train = pd.DataFrame({"cat": ["a"], "other": [1]})
        block = build_categorical_block(train, train, ["cat"], [])
        assert list(block.columns) == ["cat_le"]
