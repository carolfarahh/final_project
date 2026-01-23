import pandas as pd
import pytest

from src.eda import (
    row_count_log,
    log_step,
    log_to_df,
    missingness_table,
    sample_for_plot,
    group_descriptives,
    sex_by_stage_table,
    age_by_stage_summary,
    assert_filtered_gene_values,
)


# ============================================================
# Row-count logging tests
# ============================================================
def test_row_count_log_flow():
    df1 = pd.DataFrame({"A": [1, 2, 3]})
    df2 = pd.DataFrame({"A": [1, 2]})

    log = row_count_log()
    log_step(log, "raw", df1)
    log_step(log, "after_filter", df2)

    out = log_to_df(log)

    assert list(out["step"]) == ["raw", "after_filter"]
    assert list(out["rows"]) == [3, 2]
    assert out.loc[1, "pct_remaining"] == (2 / 3) * 100


# ============================================================
# Missingness + sampling tests
# ============================================================
def test_missingness_table_counts():
    df = pd.DataFrame({"A": [1, None], "B": [None, None]})
    out = missingness_table(df)

    assert out.loc["B", "missing_count"] == 2
    assert out.loc["A", "missing_count"] == 1


def test_sample_for_plot_small_returns_copy():
    df = pd.DataFrame({"A": [1, 2, 3]})
    out = sample_for_plot(df, n=10, random_state=0)

    # Ensure it's a copy (mutating output should not change original)
    out.loc[0, "A"] = 999
    assert df.loc[0, "A"] == 1


def test_sample_for_plot_raises_for_nonpositive_n():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError):
        sample_for_plot(df, n=0)


# ============================================================
# Group summaries tests
# ============================================================
def test_group_descriptives_has_expected_columns():
    df = pd.DataFrame(
        {
            "Disease_Stage": ["early", "early", "late"],
            "Brain_Volume_Loss": [1.0, 3.0, 10.0],
        }
    )
    out = group_descriptives(df, "Disease_Stage", "Brain_Volume_Loss")

    assert "n" in out.columns
    assert "mean" in out.columns
    assert "median" in out.columns
    assert "iqr" in out.columns
    assert out.loc["early", "n"] == 2
    assert out.loc["late", "n"] == 1


def test_sex_by_stage_table_counts():
    df = pd.DataFrame(
        {
            "Disease_Stage": ["early", "early", "late"],
            "Sex": ["F", "M", "F"],
        }
    )
    out = sex_by_stage_table(df, stage_col="Disease_Stage", sex_col="Sex")

    assert out.loc["early", "F"] == 1
    assert out.loc["early", "M"] == 1
    assert out.loc["late", "F"] == 1


def test_age_by_stage_summary_uses_group_descriptives():
    df = pd.DataFrame(
        {
            "Disease_Stage": ["early", "late", "late"],
            "Age": [20, 30, 50],
        }
    )
    out = age_by_stage_summary(df, stage_col="Disease_Stage", age_col="Age")

    assert out.loc["early", "n"] == 1
    assert out.loc["late", "n"] == 2


# ============================================================
# Sanity check tests
# ============================================================
def test_assert_filtered_gene_values_passes_when_ok():
    df = pd.DataFrame({"Gene/Factor": ["mlh1", "msh3", "htt (somatic expansion)"]})
    assert_filtered_gene_values(
        df, "Gene/Factor", ["mlh1", "msh3", "htt (somatic expansion)"]
    )


def test_assert_filtered_gene_values_raises_when_unexpected_present():
    df = pd.DataFrame({"Gene/Factor": ["mlh1", "other"]})
    with pytest.raises(ValueError):
        assert_filtered_gene_values(df, "Gene/Factor", ["mlh1"])
