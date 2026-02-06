import pandas as pd
from src.data_cleaning import _require_columns, select_columns, strip_spaces_columns, normalize_case_columns, gene_filter,convert_numeric_columns, drop_missing_required, remove_influential_by_cooks

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
        "name": ["alice", "bob", "carol"],
        "city": ["ny", "la", "paris"]
    })

    pd.testing.assert_frame_equal(result, expected)


def test_gene_filter_lower():
    df = pd.DataFrame({
        "gene": ["BRCA1", "TP53", "EGFR", "BRCA2"]
    })

    result = gene_filter(df, "gene", ["brca1", "egfr"], method="lower")

    expected = pd.DataFrame({
        "gene": ["BRCA1", "EGFR"]
    }).reset_index(drop=True)

    result = result.reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)

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
        "a": [1, 4],
        "b": [5, 8],
        "c": [9, 12]
    }).reset_index(drop=True)

    result = result.reset_index(drop=True)
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

def test_remove_influential_by_cooks_returns_three_outputs():
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4, 100],
        "IV": ["A", "A", "B", "B", "B"]
    })

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

def test_remove_influential_by_cooks_removes_outlier():
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4, 100],
        "IV": ["A", "A", "B", "B", "B"]
    })

    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", statistical_test="ANOVA"
    )

    # The row with 100 should be in influential
    assert 100 in influential["DV"].values

    # It should not be in cleaned
    assert 100 not in cleaned["DV"].values


def test_remove_influential_by_cooks_ancova():
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4, 100],
        "IV": ["A", "A", "B", "B", "B"],
        "cov": [10, 20, 30, 40, 50]
    })

    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", covariate="cov", statistical_test="ANCOVA"
    )

    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(influential, pd.DataFrame)
    assert isinstance(threshold, float)

def test_remove_influential_by_cooks_moderated_regression():
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4, 100],
        "IV": ["A", "A", "B", "B", "B"],
        "cov": [10, 20, 30, 40, 50]
    })

    cleaned, influential, threshold = remove_influential_by_cooks(
        df, DV="DV", IV="IV", covariate="cov", statistical_test="Moderated Regression"
    )

    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(influential, pd.DataFrame)
    assert isinstance(threshold, float)

def test_remove_influential_by_cooks_invalid_test():
    df = pd.DataFrame({"DV": [1, 2], "IV": ["A", "B"]})

    with pytest.raises(ValueError, match="Values must either be 'ANOVA', 'ANCOVA' or 'Moderated Regression'"):
        remove_influential_by_cooks(df, DV="DV", IV="IV", statistical_test="t-test")


