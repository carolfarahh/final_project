import pandas as pd
import pytest
from EDA import duplicates_info, numeric_summary, categorical_summary, group_descriptives, crosstab_counts
from validation import assert_required_columns


def test_duplicates_info():
    # --- Case 1: Some duplicates ---
    df = pd.DataFrame({
        "A": [1, 2, 2, 3],
        "B": ["x", "y", "y", "z"]
    })
    result = duplicates_info(df)
    # There is 1 duplicate row: the row with (2, "y")
    assert result["n_duplicate_rows"] == 1.0
    assert result["duplicate_pct"] == 25.0  # 1 out of 4 rows

    # --- Case 2: No duplicates ---
    df2 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })
    result2 = duplicates_info(df2)
    assert result2["n_duplicate_rows"] == 0.0
    assert result2["duplicate_pct"] == 0.0

    # --- Case 3: Empty DataFrame ---
    df3 = pd.DataFrame(columns=["A", "B"])
    result3 = duplicates_info(df3)
    assert result3["n_duplicate_rows"] == 0.0
    assert result3["duplicate_pct"] == 0.0

    # --- Case 4: Invalid input ---
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        duplicates_info([1, 2, 3])


def test_numeric_summary():
    # --- Case 1: Default numeric summary ---
    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": [10.0, 20.0, 30.0, 40.0],
        "C": ["x", "y", "z", "w"]  # non-numeric
    })

    result = numeric_summary(df)
    # Only numeric columns should be included
    assert set(result.index) == {"A", "B"}
    assert "mean" in result.columns
    assert result.loc["A", "mean"] == 2.5
    assert result.loc["B", "max"] == 40.0

    # --- Case 2: Specific numeric columns ---
    result2 = numeric_summary(df, cols=["A"])
    assert set(result2.index) == {"A"}
    assert result2.loc["A", "mean"] == 2.5

    # --- Case 3: Non-numeric column passed ---
    with pytest.raises(TypeError, match="Non-numeric columns passed to numeric_summary:"):
        numeric_summary(df, cols=["C"])

    # --- Case 4: Empty DataFrame ---
    df_empty = pd.DataFrame()
    result_empty = numeric_summary(df_empty)
    assert result_empty.empty

    # --- Case 5: Invalid input ---
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        numeric_summary([1, 2, 3])



def test_categorical_summary():
    # --- Case 1: Default categorical summary ---
    df = pd.DataFrame({
        "A": ["x", "y", "x", "z", "x"],
        "B": ["cat", "dog", "cat", "dog", "dog"],
        "C": [1, 2, 3, 4, 5]  # numeric column
    })

    result = categorical_summary(df)
    # Only non-numeric columns should be included
    assert set(result.keys()) == {"A", "B"}
    # Each value count should sum to 5
    assert result["A"]["count"].sum() == 5
    assert result["B"]["count"].sum() == 5
    # Percentages sum to 100
    assert round(result["A"]["pct"].sum(), 5) == 100.0
    assert round(result["B"]["pct"].sum(), 5) == 100.0

    # --- Case 2: Explicit columns ---
    result2 = categorical_summary(df, cols=["A"])
    assert set(result2.keys()) == {"A"}
    assert result2["A"]["count"].sum() == 5

    # --- Case 3: Numeric column passed ---
    with pytest.raises(TypeError, match="Numeric columns passed to categorical_summary:"):
        categorical_summary(df, cols=["C"])

    # --- Case 4: Empty DataFrame ---
    df_empty = pd.DataFrame()
    result_empty = categorical_summary(df_empty)
    assert result_empty == {}

    # --- Case 5: Invalid input ---
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        categorical_summary([1, 2, 3])

#pytest src/EDA_testing.py::test_group_descriptives


def test_group_descriptives():
    # --- Case 1: Normal usage ---
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "C"],
        "value": [1, 2, 3, 4, 5, 6]
    })

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

    # --- Case 2: Non-numeric value_col ---
    df_invalid = pd.DataFrame({
        "group": ["A", "A", "B"],
        "value": ["x", "y", "z"]
    })

    with pytest.raises(ValueError, match=f"'value' must be numeric and convertible to float"):
        group_descriptives(df_invalid, group_col="group", value_col="value")

 
#pytest src/EDA_testing.py::test_crosstab_counts

def test_crosstab_counts():
    # Case 1: Normal usage ---
    df = pd.DataFrame({
        "Gender": ["M", "F", "F", "M", "F", "M"],
        "Outcome": ["Yes", "No", "Yes", "No", "Yes", "Yes"]
    })

    result = crosstab_counts(df, row_col="Gender", col_col="Outcome")

    # Check shape
    assert result.shape == (2, 2)  # 2 genders x 2 outcomes

    # Check counts manually
    assert result.loc["M", "Yes"] == 2
    assert result.loc["M", "No"] == 1
    assert result.loc["F", "Yes"] == 2
    assert result.loc["F", "No"] == 1



    # Empty DataFrame
    df_empty = pd.DataFrame(columns=["Gender", "Outcome"])
    result_empty = crosstab_counts(df_empty, row_col="Gender", col_col="Outcome")
    # Should return empty DataFrame
    assert result_empty.empty


