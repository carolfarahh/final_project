import pandas as pd
import pytest
from data_cleaning import _require_columns, select_columns, strip_spaces_columns, normalize_case_columns, gene_filter,convert_numeric_columns, drop_missing_required, remove_influential_by_cooks

def test_require_columns_success():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    # Should NOT raise an error
    _require_columns(df, ["a", "b"])


def test_select_columns_success():
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
        "c": [5, 6]
    })

    result = select_columns(df, ["a", "c"])

    # Correct columns
    assert list(result.columns) == ["a", "c"]

    # Correct values
    pd.testing.assert_frame_equal(
        result,
        df[["a", "c"]]
    )

    # Ensure it's a copy, not a view
    result.iloc[0, 0] = 999
    assert df.loc[0, "a"] != 999

def test_strip_spaces_columns_success():
    df = pd.DataFrame({
        "name": ["  Alice  ", "Bob", "  Carol"],
        "city": ["  NY", "LA  ", "  Paris  "]
    })

    result = strip_spaces_columns(df, ["name", "city"])

    expected = pd.DataFrame({
        "name": ["Alice", "Bob", "Carol"],
        "city": ["NY", "LA", "Paris"]
    }).astype("string")

    pd.testing.assert_frame_equal(result, expected)

def test_normalize_case_columns_lower():
    df = pd.DataFrame({
        "name": ["Alice", "BOB", "Carol"],
        "city": ["NY", "la", "Paris"]
    })

    result = normalize_case_columns(df, ["name", "city"], method="lower")

    expected = pd.DataFrame({
        "name": pd.Series(["alice", "bob", "carol"], dtype="string"),
        "city": pd.Series(["ny", "la", "paris"], dtype="string")
    })

    pd.testing.assert_frame_equal(result, expected)

def test_gene_filter_lower():
    df = pd.DataFrame({
        "gene": ["MLH1", "MSH3", "HTT", "HTT3"]
    })

    result = gene_filter(df, "gene", ["mlh1", "msh3"], method="lower")

    expected = pd.DataFrame({
        "cleaned_gene": ["MLH1", "MSH3"]
    })

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
    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": [5, None, 7, 8],
        "c": [9, 10, 11, 12]
    })

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
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4]
    })

    result = drop_missing_required(df, ["a", "b"])

    pd.testing.assert_frame_equal(result, df)

def test_drop_missing_required_missing_column():
    df = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(KeyError):
        drop_missing_required(df, ["a", "b"])

def test_drop_missing_required_original_df_unchanged():
    df = pd.DataFrame({
        "a": [1, None],
        "b": [3, 4]
    })

    _ = drop_missing_required(df, ["a", "b"])

    # Original DataFrame still has the NaN row
    assert pd.isna(df.loc[1, "a"])



import pandas as pd
import numpy as np

def test_remove_influential_by_cooks_returns_three_outputs():
    # 5 IV levels, 20 observations per level
    IV_levels = ["A", "B", "C", "D", "E"]
    df = pd.DataFrame({
        "DV": np.concatenate([np.arange(1, 21) + i*0 for i in range(5)]) ,  # 20*5 = 100 rows
        "IV": np.repeat(IV_levels, 20)
    })

    # Add an extreme outlier
    df.loc[len(df)] = [1000, "E"]

    # Call the function
    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", statistical_test="ANOVA"
    )

    # Check types
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(influential, pd.DataFrame)
    assert isinstance(threshold, float)

    # Check columns
    assert "DV" in cleaned.columns
    assert "IV" in cleaned.columns

    # Check that outlier was detected
    assert 1000 in influential["DV"].values
    assert 1000 not in cleaned["DV"].values


def test_remove_influential_by_cooks_removes_outlier():
    # Create a larger dataset to ensure Cook's distance works
    n_per_group = 20
    df = pd.DataFrame({
        "DV": list(range(1, n_per_group + 1)) + list(range(1, n_per_group + 1)) + [200],  # last row is outlier
        "IV": ["A"] * n_per_group + ["B"] * n_per_group + ["B"]  # last row is outlier
    })

    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", statistical_test="ANOVA"
    )

    # The extreme value should be flagged as influential
    assert 200 in influential["DV"].values

    # It should not appear in the cleaned dataset
    assert 200 not in cleaned["DV"].values



def test_remove_influential_by_cooks_ancova():
    # Create a larger dataset to ensure Cook's distance works
    n_per_group = 20
    df = pd.DataFrame({
        "DV": list(range(1, n_per_group + 1)) + list(range(1, n_per_group + 1)) + [200],  # extreme outlier
        "IV": ["A"] * n_per_group + ["B"] * n_per_group + ["B"],  # last row is outlier
        "cov": list(range(10, 10 + n_per_group)) + list(range(20, 20 + n_per_group)) + [50]
    })

    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", covariate="cov", statistical_test="ANCOVA"
    )

    # Check types
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(influential, pd.DataFrame)
    assert isinstance(threshold, float)

    # Optional: check that the extreme outlier was removed
    assert 200 in influential["DV"].values
    assert 200 not in cleaned["DV"].values

def test_remove_influential_by_cooks_moderated_regression():
    n_per_group = 20

    # Create a larger dataset
    df = pd.DataFrame({
        "DV": list(range(1, n_per_group + 1)) + list(range(1, n_per_group + 1)) + [200],  # last row is outlier
        "IV": ["A"] * n_per_group + ["B"] * n_per_group + ["B"],  # last row is outlier
        "cov": list(range(10, 10 + n_per_group)) + list(range(20, 20 + n_per_group)) + [50]
    })

    # Call the function
    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", covariate="cov", statistical_test="Moderated Regression"
    )

    # Check types
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(influential, pd.DataFrame)
    assert isinstance(threshold, float)

    # Optional: check that the outlier was flagged
    assert 200 in influential["DV"].values
    assert 200 not in cleaned["DV"].values


def test_remove_influential_by_cooks_invalid_test():
    df = pd.DataFrame({"DV": [1, 2], "IV": ["A", "B"]})

    with pytest.raises(ValueError, match="Values must either be 'ANOVA', 'ANCOVA' or 'Moderated Regression'"):
        remove_influential_by_cooks(df, DV="DV", IV="IV", statistical_test="t-test")


