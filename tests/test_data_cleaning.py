import pandas as pd
import numpy as np
import pytest
from src.data_cleaning import (_require_columns,
                        select_columns, 
                        strip_spaces_columns, 
                        normalize_case_columns, 
                        gene_filter,convert_numeric_columns, 
                        drop_missing_required, 
                        remove_influential_by_cooks)

def test_require_columns_success():
    df = pd.DataFrame({"a": [1, 2],"b": [3, 4],"c": [5, 6]})
    # Should NOT raise an error
    _require_columns(df, ["a", "b"])

def test_select_columns_success():
    df = pd.DataFrame({"a": [1, 2],"b": [3, 4],"c": [5, 6]})
    result = select_columns(df, ["a", "c"])
    # Correct columns
    assert list(result.columns) == ["a", "c"]
    # Correct values
    pd.testing.assert_frame_equal(result,df[["a", "c"]])
    # Ensure it's a copy, not a view
    result.iloc[0, 0] = 999
    assert df.loc[0, "a"] != 999

def test_strip_spaces_columns_success():
    df = pd.DataFrame({"name": ["  Adi  ", "Amani", "  Carol"], "city": ["  NY", "LA  ", "  Paris  "] })
    result = strip_spaces_columns(df, ["name", "city"])
    expected = pd.DataFrame({
        "name": ["Adi", "Amani", "Carol"],
        "city": ["NY", "LA", "Paris"]
    }).astype("string")
    pd.testing.assert_frame_equal(result, expected)

def test_normalize_case_columns_lower():
    df = pd.DataFrame({
        "name": ["Adi", "AMANI", "Carol"],
        "city": ["NY", "la", "Paris"]
    })
    result = normalize_case_columns(df, ["name", "city"], method="lower")
    expected = pd.DataFrame({
        "name": pd.Series(["adi", "amani", "carol"], dtype="string"),
        "city": pd.Series(["ny", "la", "paris"], dtype="string")
    })
    pd.testing.assert_frame_equal(result, expected)

def test_gene_filter_lower():
    df = pd.DataFrame({"gene": ["MLH1", "MSH3", "HTT", "HTT3"]})
    result = gene_filter(df, "gene", ["mlh1", "msh3"], method="lower")
    expected = pd.DataFrame({"cleaned_gene": ["MLH1", "MSH3"]})
    # If you want lowercase values in expected:
    expected["cleaned_gene"] = expected["cleaned_gene"].str.lower()
    result = result.reset_index(drop=True)
    expected = expected.reset_index(drop=True)

def test_convert_numeric_columns_success():
    df = pd.DataFrame({
        "a": ["1", "2", "3"],
        "b": ["4.5", "5.5", "6.5"]
    })
    result = convert_numeric_columns(df, ["a", "b"])
    expected = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4.5, 5.5, 6.5]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_drop_missing_required_drops_na():
    df = pd.DataFrame({"a": [1, 2, None, 4],"b": [5, None, 7, 8],"c": [9, 10, 11, 12]})
    result = drop_missing_required(df, ["a", "b"])
    expected = pd.DataFrame({
        "a": ([1, 4]),
        "b": ([5, 8]),
        "c": ([9, 12])
    }).reset_index(drop=True)
    result = result.reset_index(drop=True)
    for col in result.columns:
        result[col] = result[col].astype("int64") 
    pd.testing.assert_frame_equal(result, expected)

def test_drop_missing_required_keeps_complete_rows():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = drop_missing_required(df, ["a", "b"])
    pd.testing.assert_frame_equal(result, df)

def test_drop_missing_required_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        drop_missing_required(df, ["a", "b"])

def test_drop_missing_required_original_df_unchanged():
    df = pd.DataFrame({"a": [1, None],"b": [3, 4]})
    _ = drop_missing_required(df, ["a", "b"])
    # Original DataFrame still has the NaN row
    assert pd.isna(df.loc[1, "a"])