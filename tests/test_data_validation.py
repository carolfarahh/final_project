import pandas as pd
from src.data_validation import unique_values, value_counts_report


def test_unique_values_returns_sorted_unique_values():
    df = pd.DataFrame({"Sex": ["Male", "Female", "Male", None]})
    out = unique_values(df, "Sex")
    assert out == ["Female", "Male"]


def test_value_counts_report_includes_nan_when_dropna_false():
    df = pd.DataFrame({"Disease_Stage": ["Early", "Early", None]})
    counts = value_counts_report(df, "Disease_Stage")
    assert counts["Early"] == 2
    assert counts.isna().sum() == 0  # counts series itself should not contain NaN
    assert counts.index.isna().sum() == 1  # NaN appears as an index entry
