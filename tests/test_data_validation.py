import pandas as pd
import pytest

from src.data_validation import unique_values, value_counts_report


def test_unique_values_returns_sorted_unique_non_null_values():
    df = pd.DataFrame({"Stage": ["Early", None, "Late", "Early", "Pre"]})
    out = unique_values(df, "Stage")
    assert out == ["Early", "Late", "Pre"]


def test_unique_values_missing_column_raises_keyerror():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(KeyError):
        unique_values(df, "Stage")


def test_value_counts_report_includes_nan_counts():
    df = pd.DataFrame({"Sex": ["female", "male", None, "female"]})
    out = value_counts_report(df, "Sex")
    assert out.to_dict() == {"female": 2, "male": 1, None: 1}


def test_value_counts_report_missing_column_raises_keyerror():
    df = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(KeyError):
        value_counts_report(df, "Sex")
