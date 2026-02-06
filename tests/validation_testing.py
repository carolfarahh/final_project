
import pandas as pd
import pytest
from validation import assert_allowed_values, assert_required_columns, validate_missing_values, validate_variance_in_dv, validate_category_levels_n, validate_group_size, validate_variable_type
from app_logger import logger


def test_assert_required_columns():
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })

    # --- Case 1: All columns present ---
    # Should not raise any exception
    assert_required_columns(df, ["A", "B"])
    assert_required_columns(df, ["A", "B", "C"])

    # --- Case 2: Some columns missing ---
    with pytest.raises(KeyError, match="Missing required columns:"):
        assert_required_columns(df, ["A", "D"])  # D is missing

    with pytest.raises(KeyError) as e:
        assert_required_columns(df, ["X", "Y"])
    assert "X" in str(e.value) and "Y" in str(e.value)

    # --- Case 3: Invalid input (not a DataFrame) ---
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        assert_required_columns([1, 2, 3], ["A"])


def test_assert_allowed_values():
    df = pd.DataFrame({
        "Color": ["red", "blue", "green", "red", None]
    })

    # --- Case 1: All values allowed (dropna=True) ---
    assert_allowed_values(df, col="Color", allowed_values=["red", "blue", "green"])

    # --- Case 2: Some unexpected values (dropna=True) ---
    df2 = pd.DataFrame({
        "Color": ["red", "blue", "yellow", "green"]
    })
    with pytest.raises(ValueError, match="Unexpected values in 'Color'"):
        assert_allowed_values(df2, col="Color", allowed_values=["red", "blue", "green"])

    # --- Case 3: Unexpected value with dropna=False (NaN included) ---
    df3 = pd.DataFrame({
        "Color": ["red", "blue", None, "green", "pink"]
    })
    with pytest.raises(ValueError) as e:
        assert_allowed_values(df3, col="Color", allowed_values=["red", "blue", "green"], dropna=False)
    assert "pink" in str(e.value)

    # --- Case 4: Column missing ---
    df_missing = pd.DataFrame({
        "Other": [1, 2, 3]
    })
    with pytest.raises(KeyError):
        assert_allowed_values(df_missing, col="Color", allowed_values=["red", "blue"])

    # --- Case 5: dropna=True ignores NaN ---
    df_nan = pd.DataFrame({
        "Color": ["red", "blue", None]
    })
    assert_allowed_values(df_nan, col="Color", allowed_values=["red", "blue"], dropna=True)




def test_validate_missing_values():
    # --- Case 1: No missing values ---
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [7, 8, 9]
    })
    result = validate_missing_values(df, ["A", "B"])
    for col in result.columns:
        result[col] = result[col].astype("int64") 
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

    # --- Case 2: Some missing values ---
    df2 = pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": [4, None, 6, 7],
        "C": [7, 8, 9, 10]
    })

    result2 = validate_missing_values(df2, ["A", "B"])
    for col in result.columns:
        result2[col] = result2[col].astype("int64") 
    # Only rows without NaNs in A or B should remain
    expected2 = pd.DataFrame({
        "A": [1, 4],
        "B": [4, 7],
        "C": [7, 10]
    })
    pd.testing.assert_frame_equal(result2.reset_index(drop=True), expected2.reset_index(drop=True))

    # --- Case 3: All rows missing ---
    df3 = pd.DataFrame({
        "A": [None, None],
        "B": [None, None]
    })
    result3 = validate_missing_values(df3, ["A", "B"])
    assert result3.empty

    # --- Case 4: Column missing ---
    df_missing = pd.DataFrame({
        "X": [1, 2, 3]
    })
    with pytest.raises(KeyError):
        validate_missing_values(df_missing, ["A"])



def test_validate_variance_in_dv():
    # --- Case 1: Normal case (DV has variation) ---
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4],
        "IV": ["A", "A", "B", "B"]
    })
    # Should not raise any error
    validate_variance_in_dv(df, dv="DV")

    # --- Case 2: No variation in DV ---
    df_no_var = pd.DataFrame({
        "DV": [5, 5, 5, 5],
        "IV": ["A", "A", "B", "B"]
    })
    with pytest.raises(ValueError, match="Dependent variable 'DV' has no variation"):
        validate_variance_in_dv(df_no_var, dv="DV")

    # --- Case 3: Missing DV column ---
    df_missing = pd.DataFrame({
        "X": [1, 2, 3]
    })
    with pytest.raises(KeyError):
        validate_variance_in_dv(df_missing, dv="DV")

#pytest src/validation_testing.py::test_validate_category_levels_n
def test_validate_category_levels_n():
    # Case 1: Normal case (enough categories for ANOVA) ---
    df = pd.DataFrame({
        "Factor1": ["A", "B", "C", "A", "B", "C"],
        "Factor2": ["X", "Y", "Z", "X", "Y", "Z"]
    })
    # Should not raise error
    validate_category_levels_n(df, factors_list=["Factor1", "Factor2"], test="ANOVA")

    # --- Case 2: Too few categories for ANOVA/ANCOVA ---
    df2 = pd.DataFrame({
        "Factor1": ["A", "B", "A", "B"],  # only 2 levels
        "Factor2": ["X", "Y", "X", "Y"]   # only 2 levels
    })
    with pytest.raises(ValueError, match="Validation Failed: Factor 'Factor1' must have at least 3 categories"):
        validate_category_levels_n(df2, factors_list=["Factor1"], test="ANOVA")

    # --- Case 3: Too few categories for moderated_regression (min 2) ---
    df3 = pd.DataFrame({
        "Factor1": ["A", "A", "B", "B"]  # 2 levels, which is enough
    })
    # Should not raise error
    validate_category_levels_n(df3, factors_list=["Factor1"], test="moderated_regression")

    df4 = pd.DataFrame({
        "Factor1": ["A", "A", "A"]  # 1 level only
    })
    with pytest.raises(ValueError, match="Validation Failed: Factor 'Factor1' must have at least 2 categories"):
        validate_category_levels_n(df4, factors_list=["Factor1"], test="moderated_regression")

 #pytest src/validation_testing.py::test_validate_group_size

def test_validate_group_size():
    # --- Case 1: Normal case (all groups ≥ n_min for default test) ---
    df = pd.DataFrame({
        "IV": ["A", "A", "B", "B", "B", "C", "C"],
        "Factor2": ["X", "X", "Y", "Y", "Y", "Z", "Z"],
        "DV": [1,2,3,4,5,6,7]
    })
    # n_min default is 2 thus, all groups meet it
    validate_group_size(df, iv="IV", factor2="Factor2", test="ANOVA")

    # --- Case 2: Group below minimal size (default test, n_min=2) ---
    df2 = pd.DataFrame({
        "IV": ["A", "A", "B"],  # B has only 1 row
        "Factor2": ["X", "X", "Y"],
        "DV": [1,2,3]
    })
    with pytest.raises(ValueError, match="Problematic cells"):
        validate_group_size(df2, iv="IV", factor2="Factor2", test="ANOVA")

    # --- Case 3: anova_tukey requires n_min=5 ---
    df3 = pd.DataFrame({
        "IV": ["A"]*4 + ["B"]*5,
        "Factor2": ["X"]*4 + ["Y"]*5,
        "DV": list(range(9))
    })
    # A/X has 4 rows <5 → should raise error
    with pytest.raises(ValueError, match="Problematic cells"):
        validate_group_size(df3, iv="IV", factor2="Factor2", test="anova_tukey")

    # --- Case 4: anova_tukey with sufficient group size ---
    df4 = pd.DataFrame({
        "IV": ["A"]*5 + ["B"]*5,
        "Factor2": ["X"]*5 + ["Y"]*5,
        "DV": list(range(10))
    })
    # Should pass without error
    validate_group_size(df4, iv="IV", factor2="Factor2", test="anova_tukey")


# pytest src/validation_testing.py::test_validate_variable_type

def test_validate_variable_type():
    df = pd.DataFrame({
        "cat1": ["A", "B", "A", "C"],
        "num1": ["1", "2", "3", "4"],
        "num2": [10, 20, 30, 40],
        "cat2": ["X", "Y", "X", "Y"]
    })

    # --- Case 1: Convert cat1 and cat2 to categorical, num1 to numeric ---
    result = validate_variable_type(
        df,
        categorical_list=["cat1", "cat2"],
        numeric_list=["num1"]
    )

    assert isinstance(result["cat1"].dtype, CategoricalDtype)
    assert isinstance(result["cat2"].dtype, CategoricalDtype)
    assert pd.api.types.is_numeric_dtype(result["num1"])
    assert pd.api.types.is_numeric_dtype(result["num2"])  # already numeric

    # --- Case 2: Numeric column already numeric ---
    result2 = validate_variable_type(df, numeric_list=["num2"])
    assert pd.api.types.is_numeric_dtype(result2["num2"])

    # --- Case 3: Categorical passed as numeric → should raise ValueError ---
    df_invalid = pd.DataFrame({
        "cat": ["A", "B", "C"]
    }) 
    with pytest.raises(ValueError, match="Validation Failed: Variable 'cat' could not be converted to numeric values"):
        validate_variable_type(df_invalid, numeric_list=["cat"])

    # --- Case 4: Non-convertible numeric column → should raise ValueError ---
    df_nonconvert = pd.DataFrame({
        "num": ["a", "b", "c"]
    })
    with pytest.raises(ValueError, match="Variable 'num' could not be converted to numeric"):
        validate_variable_type(df_nonconvert, numeric_list=["num"])

    # --- Case 5: Columns already correct type (no error) ---
    df_correct = pd.DataFrame({
        "cat": pd.Series(["X", "Y"], dtype="category"),
        "num": pd.Series([1.0, 2.0])
    })
    result5 = validate_variable_type(df_correct, categorical_list=["cat"], numeric_list=["num"])
    assert pd.api.types.is_categorical_dtype(result5["cat"])
    assert pd.api.types.is_numeric_dtype(result5["num"])
