import pandas as pd
import pytest
from src.EDA import duplicates_info, numeric_summary, categorical_summary, group_descriptives


def test_duplicates_info():
    # Case 1: Some duplicates 
    df = pd.DataFrame({"A": [1, 2, 2, 3], "B": ["x", "y", "y", "z"]})
    result = duplicates_info(df)
    # There is 1 duplicate row: the row with (2, "y")
    assert result["n_duplicate_rows"] == 1.0
    assert result["duplicate_pct"] == 25.0  # 1 out of 4 rows
    # Case 2: No duplicates 
    df2 = pd.DataFrame({"A": [1, 2, 3],"B": ["x", "y", "z"]})
    result2 = duplicates_info(df2)
    assert result2["n_duplicate_rows"] == 0.0
    assert result2["duplicate_pct"] == 0.0
    # Case 3: Empty DataFrame 
    df3 = pd.DataFrame(columns=["A", "B"])
    result3 = duplicates_info(df3)
    assert result3["n_duplicate_rows"] == 0.0
    assert result3["duplicate_pct"] == 0.0
    # Case 4: Invalid input 
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        duplicates_info([1, 2, 3])

def test_numeric_summary():

    df_basic = pd.DataFrame({"Age": [20, 21, 22, 23],"Score": [88, 92, 95, 85],"Gender": ["F", "M", "F", "M"]})
    # Case 1: automatic numeric selection when cols=None
    result_auto = numeric_summary(df_basic, None)
    for col in ["Age", "Score"]:
        assert col in result_auto.columns or col in result_auto.index
    # Case 2: reject non-numeric columns
    with pytest.raises(TypeError):
        numeric_summary(df_basic, ["Age", "Gender"])
    # Case 3: dataframe with no numeric columns at all
    df_non_numeric = pd.DataFrame({"A": ["x", "y"], "B": ["p", "q"]})
    result_none = numeric_summary(df_non_numeric, None)
    assert isinstance(result_none, pd.DataFrame)
    assert result_none.empty


def test_categorical_summary():
    # Case 1 basic functionality
    df_basic = pd.DataFrame({"Gender": ["F", "M", "F", "F", "M"],"Group": ["A", "A", "B", "B", "B"]})
    result = categorical_summary(df_basic, ["Gender"])
    assert "Gender" in result
    assert isinstance(result["Gender"], pd.DataFrame)
    assert "count" in result["Gender"].columns
    assert "pct" in result["Gender"].columns
    assert result["Gender"].loc["F", "count"] == 3
    assert result["Gender"].loc["M", "count"] == 2
    assert result["Gender"]["pct"].sum() == pytest.approx(100.0)
    # Case 2 numeric column rejection
    df_numeric = pd.DataFrame({"Age": [20, 21, 22, 23]})
    with pytest.raises(TypeError):
        categorical_summary(df_numeric, ["Age"])
    # Case 3 empty column edge case
    df_empty = pd.DataFrame({"Category": pd.Series([], dtype="object")})
    result_empty = categorical_summary(df_empty, ["Category"])
    assert "Category" in result_empty
    assert isinstance(result_empty["Category"], pd.DataFrame)

def test_group_descriptives():
    # Case 1: Normal usage
    df = pd.DataFrame({"group": ["A", "A", "A", "B", "B", "C"],"value": [1, 2, 3, 4, 5, 6]})
    result = group_descriptives(df, group_col="group", value_col="value")
    # Check that all groups are included
    assert set(result.index) == {"A", "B", "C"}
    # Check statistics for group A
    assert result.loc["A", "n"] == 3
    assert result.loc["A", "mean"] == 2.0
    assert result.loc["A", "median"] == 2.0
    assert result.loc["A", "iqr"] == 1.0  # Q3=2.5, Q1=1.5 thus IQR=1
    # Check statistics for group B
    assert result.loc["B", "n"] == 2
    assert result.loc["B", "mean"] == 4.5
    assert result.loc["B", "median"] == 4.5
    assert result.loc["B", "iqr"] == 0.5  # Q3=4.75, Q1=4.25 thus IQR=0.5
    #  Case 2: Non-numeric value_col 
    df_invalid = pd.DataFrame({"group": ["A", "A", "B"],"value": ["x", "y", "z"]})
    with pytest.raises(ValueError, match=f"'value' must be numeric and convertible to float"):
        group_descriptives(df_invalid, group_col="group", value_col="value")