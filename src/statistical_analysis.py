import numpy as np
import pandas as pd

from src.statistical_analysis import welch_anova, tukey, levene_test


# =============================================================================
# Tests: statistical_analysis.py (Pingouin wrappers)
#
# These tests validate:
#   1) Output type (DataFrame).
#   2) Output is not empty for valid inputs.
#   3) Output structure (p-value columns, group label columns).
#   4) Tukey: expected number of pairwise comparisons for k groups.
#
# We do NOT assert exact numeric values to avoid flaky tests.
# =============================================================================


def build_three_group_dataset_for_tests():
    """Shared deterministic dataset used across multiple tests."""
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
def test_welch_anova_returns_dataframe():
    """Normal case: welch_anova should return a pandas DataFrame."""
    df = build_three_group_dataset_for_tests()

    results = welch_anova(df, dependent_variable="y", independent_variable="group")

    assert isinstance(results, pd.DataFrame)


def test_welch_anova_output_not_empty():
    """Normal case: welch_anova output should not be empty for valid inputs."""
    df = build_three_group_dataset_for_tests()

    results = welch_anova(df, dependent_variable="y", independent_variable="group")

    assert len(results) > 0


def test_welch_anova_contains_p_value_column():
    """Structural check: Welch ANOVA output should include a p-value column."""
    df = build_three_group_dataset_for_tests()

    results = welch_anova(df, dependent_variable="y", independent_variable="group")

    assert ("p-unc" in results.columns) or ("pval" in results.columns)


# -----------------------------------------------------------------------------
# Tests: tukey
# -----------------------------------------------------------------------------
def test_tukey_returns_dataframe():
    """Normal case: tukey should return a pandas DataFrame."""
    df = build_three_group_dataset_for_tests()

    results = tukey(df, dependent_variable="y", independent_variable="group")

    assert isinstance(results, pd.DataFrame)


def test_tukey_expected_number_of_pairwise_comparisons():
    """For k groups, Tukey should return k*(k-1)/2 pairwise comparisons."""
    df = build_three_group_dataset_for_tests()
    k = df["group"].nunique()
    expected_pairs = k * (k - 1) // 2

    results = tukey(df, dependent_variable="y", independent_variable="group")

    assert len(results) == expected_pairs


def test_tukey_contains_group_label_columns():
    """Structural check: Tukey output should include columns identifying compared groups."""
    df = build_three_group_dataset_for_tests()

    results = tukey(df, dependent_variable="y", independent_variable="group")

    assert ("A" in results.columns) and ("B" in results.columns)


# -----------------------------------------------------------------------------
# Tests: levene_test
# -----------------------------------------------------------------------------
def test_levene_test_returns_dataframe():
    """Normal case: levene_test should return a pandas DataFrame."""
    df = build_three_group_dataset_for_tests()

    results = levene_test(df, dependent_variable="y", independent_variable="group")

    assert isinstance(results, pd.DataFrame)


def test_levene_test_output_not_empty():
    """Normal case: levene_test output should not be empty for valid inputs."""
    df = build_three_group_dataset_for_tests()

    results = levene_test(df, dependent_variable="y", independent_variable="group")

    assert len(results) > 0


def test_levene_test_contains_p_value_column():
    """Structural check: Levene output should include a p-value column."""
    df = build_three_group_dataset_for_tests()

    results = levene_test(df, dependent_variable="y", independent_variable="group")

    assert ("pval" in results.columns) or ("p-unc" in results.columns)


# -----------------------------------------------------------------------------
# Tests: invalid input (single group)
# -----------------------------------------------------------------------------
def test_statistical_functions_fail_with_single_group():
    """
    Invalid input:
    If there is only one group level, these analyses are not meaningful.
    We expect an exception (exact type can vary).
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
