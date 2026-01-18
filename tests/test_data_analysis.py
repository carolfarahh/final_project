import pandas as pd
import pytest
from src.statistical_analysis import levene_test


def test_levene_test_returns_dataframe_with_expected_columns():
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 10, 11, 12, 13],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    out = levene_test(df, "y", "group")

    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 6)
    assert set(["test", "center", "stat", "pval", "n_groups", "group_sizes"]).issubset(out.columns)
    assert out.loc[0, "test"] == "levene"
    assert 0.0 <= out.loc[0, "pval"] <= 1.0


def test_levene_test_raises_keyerror_if_column_missing():
    df = pd.DataFrame({"y": [1, 2, 3], "other": ["A", "A", "B"]})
    with pytest.raises(KeyError):
        levene_test(df, "y", "group")


def test_levene_test_raises_valueerror_if_only_one_group():
    df = pd.DataFrame({"y": [1, 2, 3], "group": ["A", "A", "A"]})
    with pytest.raises(ValueError):
        levene_test(df, "y", "group")


def test_levene_test_raises_valueerror_if_group_too_small():
    df = pd.DataFrame({"y": [1, 2, 3], "group": ["A", "A", "B"]})
    with pytest.raises(ValueError):
        levene_test(df, "y", "group", min_group_size=2)


def test_levene_test_drops_nan_when_dropna_true():
    df = pd.DataFrame({
        "y": [1, 2, None, 4, 10, 11, 12, 13],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    out = levene_test(df, "y", "group", dropna=True)

    assert out.shape[0] == 1
    assert 0.0 <= out.loc[0, "pval"] <= 1.0
