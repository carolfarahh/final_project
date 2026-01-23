import pandas as pd
import pytest

from src.eda import (
    assert_required_columns,
    assert_allowed_values,
    basic_overview,
    missingness_table,
    duplicates_info,
    numeric_summary,
    categorical_summary,
    group_descriptives,
    crosstab_counts,
)


def test_assert_required_columns_ok():
    df = pd.DataFrame({"A": [1], "B": [2]})
    assert_required_columns(df, ["A", "B"])  # should not raise


def test_assert_required_columns_raises_keyerror():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError):
        assert_required_columns(df, ["A", "B"])


def test_assert_allowed_values_raises_valueerror_on_extras():
    df = pd.DataFrame({"G": ["x", "y", "z"]})
    with pytest.raises(ValueError):
        assert_allowed_values(df, "G", allowed_values=["x", "y"])


def test_basic_overview_returns_expected_keys():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    out = basic_overview(df)
    assert out["n_rows"] == 2
    assert out["n_cols"] == 2
    assert "dtypes" in out
    assert "n_unique" in out


def test_missingness_table_counts_and_pct():
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, "x"]})
    out = missingness_table(df, sort=False)
    assert out.loc["A", "missing_count"] == 1
    assert out.loc["B", "missing_count"] == 2
    assert out.loc["A", "missing_pct"] == (1 / 3) * 100
    assert out.loc["B", "missing_pct"] == (2 / 3) * 100


def test_duplicates_info_full_row_duplicates():
    df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
    out = duplicates_info(df)
    assert out["n_duplicate_rows"] == 1.0
    assert out["duplicate_pct"] == (1 / 3) * 100


def test_numeric_summary_raises_on_non_numeric_cols():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    with pytest.raises(TypeError):
        numeric_summary(df, cols=["B"])


def test_numeric_summary_auto_selects_numeric():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    out = numeric_summary(df)
    assert "A" in out.index
    assert "B" not in out.index


def test_categorical_summary_returns_tables_per_column():
    df = pd.DataFrame({"C": ["a", "a", "b"], "D": [True, False, True]})
    out = categorical_summary(df, cols=["C"])
    assert "C" in out
    assert set(out["C"].columns) == {"count", "pct"}
    assert out["C"].loc["a", "count"] == 2


def test_group_descriptives_basic_output():
    df = pd.DataFrame({"G": ["m", "m", "f"], "Y": [1.0, 3.0, 10.0]})
    out = group_descriptives(df, group_col="G", value_col="Y")
    assert set(out.columns) == {"n", "mean", "median", "iqr"}
    assert out.loc["m", "n"] == 2


def test_group_descriptives_raises_if_value_not_numeric():
    df = pd.DataFrame({"G": ["m", "f"], "Y": ["bad", "10"]})
    with pytest.raises(ValueError):
        group_descriptives(df, group_col="G", value_col="Y")


def test_crosstab_counts_basic():
    df = pd.DataFrame({"A": ["x", "x", "y"], "B": ["p", "q", "p"]})
    tab = crosstab_counts(df, row_col="A", col_col="B")
    assert tab.loc["x", "p"] == 1
    assert tab.loc["x", "q"] == 1
    assert tab.loc["y", "p"] == 1
