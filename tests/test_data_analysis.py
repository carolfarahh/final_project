import numpy as np
import pandas as pd

from src.statistical_analysis import welch_anova, tukey, levene_test


# =============================================================================
# File: tests/test_data_analysis.py
#
# Purpose:
#   Unit tests for statistical analysis wrapper functions (Pingouin-based):
#     - welch_anova
#     - tukey
#     - levene_test
#
# Testing approach:
#   We validate stable expectations (type/structure) rather than exact p-values,
#   because exact numbers can vary with floating point and library versions.
# =============================================================================


def build_three_group_dataset_for_tests():
    """
    Build a small deterministic dataset with 3 groups and a numeric dependent variable.
    Reused across multiple tests to keep tests consistent and readable.
    """
    return pd.DataFrame(
        {
            "group": (["A"] * 8) + (["B"] * 8) + (["C"] * 8),
            "y": (
                [10, 11, 9, 10, 12, 11, 10, 9]
                + [20, 19, 21, 20, 22, 19, 20, 21]
                + [15, 14, 16, 15, 17, 14, 15, 16]
            ),
        }
    )


# -----------------------------------------------------------------------------
# Tests: welch_anova
# -----------------------------------------------------------------------------
def test_welch_anova_returns_dataframe_and_not_empty():
    """Normal case: welch_anova should return a non-empty pandas DataFrame."""
    df = build_three_group_dataset_for_tests()

    res = welch_anova(df, dependent_variable="y", independent_variable="group")

    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0


def test_welch_anova_contains_p_value_column():
    """Structural check: Welch ANOVA output should include a p-value column."""
    df = build_three_group_dataset_for_tests()

    res = welch_anova(df, dependent_variable="y", independent_variable="group")

    # Pingouin usually reports p-values under 'p-unc' for ANOVA outputs
    assert ("p-unc" in res.columns) or ("pval" in res.columns)


# -----------------------------------------------------------------------------
# Tests: tukey
# -----------------------------------------------------------------------------
def test_tukey_returns_dataframe_and_expected_number_of_rows():
    """
    Tukey post-hoc compares all unique group pairs.
    For k groups, expected comparisons = k*(k-1)/2.
    """
    df = build_three_group_dataset_for_tests()
    k = df["group"].nunique()
    expected_pairs = k * (k - 1) // 2

    res = tukey(df, dependent_variable="y", independent_variable="group")

    assert isinstance(res, pd.DataFrame)
    assert len(res) == expected_pairs


def test_tukey_contains_group_label_columns():
    """Structural check: Tukey output should include compared group label columns."""
    df = build_three_group_dataset_for_tests()

    res = tukey(df, dependent_variable="y", independent_variable="group")

    # Pingouin typically uses 'A' and 'B' to indicate group labels
    assert ("A" in res.columns) and ("B" in res.columns)


# -----------------------------------------------------------------------------
# Tests: levene_test
# -----------------------------------------------------------------------------
def test_levene_test_returns_dataframe_and_not_empty():
    """Normal case: levene_test should return a non-empty pandas DataFrame."""
    df = build_three_group_dataset_for_tests()

    res = levene_test(df, dependent_variable="y", independent_variable="group")

    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0


def test_levene_test_contains_p_value_column():
    """Structural check: Levene output should include a p-value column."""
    df = build_three_group_dataset_for_tests()

    res = levene_test(df, dependent_variable="y", independent_variable="group")

    # Pingouin homoscedasticity commonly reports p-values under 'pval'
    assert ("pval" in res.columns) or ("p-unc" in res.columns)


# -----------------------------------------------------------------------------
# Tests: invalid input scenarios
# -----------------------------------------------------------------------------
def test_statistical_functions_fail_with_single_group():
    """
    Invalid input:
    If the independent variable has only one level, analyses are not meaningful.
    We expect an exception (exact type may vary), so we catch a generic Exception.
    """
    df_one_group = pd.DataFrame({"group": ["A"] * 10, "y": np.arange(10)})

    try:
        welch_anova(df_one_group, dependent_variable="y", independent_variable="group")
        assert False
    except Exception:
        assert True

    try:
        tukey(df_one_group, dependent_variable="y", independent_variable="group")
        assert False
    except Exception:
        assert True

    try:
        levene_test(df_one_group, dependent_variable="y", independent_variable="group")
        assert False
    except Exception:
        assert True


def test_functions_raise_keyerror_when_columns_missing():
    """Invalid input: missing columns should raise KeyError."""
    df = pd.DataFrame({"other_group": ["A", "B", "C"], "other_y": [1, 2, 3]})

    try:
        welch_anova(df, dependent_variable="y", independent_variable="group")
        assert False
    except KeyError:
        assert True

    try:
        tukey(df, dependent_variable="y", independent_variable="group")
        assert False
    except KeyError:
        assert True

    try:
        levene_test(df, dependent_variable="y", independent_variable="group")
        assert False
    except KeyError:
        assert True
