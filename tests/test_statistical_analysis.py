
import pytest
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from unittest.mock import patch, MagicMock
from src.statistical_analysis import two_way_anova_test,simple_effects_anova,run_posthoc_on_significant_simple_effects,additive_anova_posthoc,run_ancova,run_ancova_posthoc,run_moderated_regression, run_spotlight_analysis

#pytest tests/test_statistical_analysis.py


def test_two_way_anova_test():
    # Create a small sample dataframe
    df = pd.DataFrame({
        "IV": ["A", "A", "B", "B", "A", "B", "A", "B"],
        "Factor2": ["X", "Y", "X", "Y", "X", "Y", "Y", "X"],
        "DV": [5, 6, 7, 8, 5, 9, 6, 8]
    })

    # Test additive ANOVA with Levene p > alpha (no robust adjustment)
    anova_table, robust_type = two_way_anova_test(df, dv="DV", iv="IV", factor2="Factor2", levene_p=0.2, check_interaction=False, alpha=0.05)
    # Check robust_type is None
    assert robust_type is None
    # Check output is a DataFrame and has expected ANOVA columns
    assert isinstance(anova_table, pd.DataFrame)
    assert "F" in anova_table.columns
    assert "PR(>F)" in anova_table.columns

    # Test interactive ANOVA with Levene p < alpha (robust adjustment)
    anova_table2, robust_type2 = two_way_anova_test(df, dv="DV", iv="IV", factor2="Factor2", levene_p=0.01, check_interaction=True, alpha=0.05)
    # Check robust_type is "hc3"
    assert robust_type2 == "hc3"
    # Check output is a DataFrame
    assert isinstance(anova_table2, pd.DataFrame)
    assert "F" in anova_table2.columns
    assert "PR(>F)" in anova_table2.columns

def test_simple_effects_anova():
    # Sample dataset
    df = pd.DataFrame({
        "Factor1": ["A", "A", "B", "B", "A", "B", "A", "B"],
        "Factor2": ["X", "Y", "X", "Y", "X", "Y", "Y", "X"],
        "DV": [5, 6, 7, 8, 5, 9, 6, 8]
    })

    results = simple_effects_anova(df, dv="DV", factor1="Factor1", factor2="Factor2", alpha=0.05)

    # Test that results is a DataFrame
    assert isinstance(results, pd.DataFrame)

    # Test that it has the expected columns (without 'significant')
    expected_cols = ["Level", "F", "p"]
    for col in expected_cols:
        assert col in results.columns

    # Test that 'Level' contains all unique levels of Factor2
    assert set(results["Level"]) == set(df["Factor2"].unique())

    # Test that 'F' and 'p' are numeric
    assert pd.api.types.is_numeric_dtype(results["F"])
    assert pd.api.types.is_numeric_dtype(results["p"])



def test_run_posthoc_on_significant_simple_effects_no_sig():
    # Sample data
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4],
        "IV": ["A", "A", "B", "B"],
        "Factor2": ["X", "Y", "X", "Y"]
    })

    simple_effects_results = pd.DataFrame({
        "Level": ["A", "B"],
        "F": [1.2, 0.5],
        "p": [0.2, 0.7],
        "significant": [False, False]
    })

    result = run_posthoc_on_significant_simple_effects(
        df, dv="DV", iv="IV", factor2="Factor2", 
        simple_effects_results=simple_effects_results, levene_p=0.1
    )

    # Should return empty dictionary if no significant effects
    assert result == {}

def test_run_posthoc_on_significant_simple_effects_sig():
    df = pd.DataFrame({
        "DV": [1, 2, 3, 4],
        "IV": ["A", "A", "B", "B"],
        "Factor2": ["X", "Y", "X", "Y"]
    })

    simple_effects_results = pd.DataFrame({
        "Level": ["A", "B"],
        "F": [5.2, 0.5],
        "p": [0.02, 0.7],
        "significant": [True, False]
    })

    # Mock the post-hoc functions to avoid actually running stats
    with patch("src.statistical_analysis.pairwise_tukeyhsd") as mock_tukey, \
         patch("src.statistical_analysis.pg.pairwise_gameshowell") as mock_gh:

        # Setup return values
        mock_tukey.return_value.summary.return_value.data = [["group1","group2","p"], ["A","B",0.03]]
        mock_gh.return_value = pd.DataFrame({"group1":["A"], "group2":["B"], "pval":[0.03]})

        # Run function
        result = run_posthoc_on_significant_simple_effects(
            df, dv="DV", iv="IV", factor2="Factor2", 
            simple_effects_results=simple_effects_results, 
            levene_p=0.2, alpha=0.05
        )

        # Check that we got one key for level "A"
        assert list(result.keys()) == ["A"]
        # Check the returned dataframe columns
        df_result = result["A"]
        assert "group1" in df_result.columns
        assert "group2" in df_result.columns
        assert "p" in df_result.columns or "pval" in df_result.columns

def test_additive_anova_posthoc():
    # Case 1: Main effect not significant, should return None
    df1 = pd.DataFrame({
        "DV": [1, 2, 3, 4],
        "Factor": ["A", "A", "B", "B"]
    })
    result1 = additive_anova_posthoc(df1, "DV", "Factor", main_effect_p=0.1, robust_type=None)
    assert result1 is None

    # Case 2: Factor has only 2 levels, should return None
    df2 = pd.DataFrame({
        "DV": [1, 2],
        "Factor": ["A", "B"]
    })
    result2 = additive_anova_posthoc(df2, "DV", "Factor", main_effect_p=0.01, robust_type=None)
    assert result2 is None

    # Case 3: Equal variances, should call Tukey HSD
    df3 = pd.DataFrame({
        "DV": [1, 2, 3, 4, 5, 6],
        "Factor": ["A", "A", "B", "B", "C", "C"]
    })
    with patch("src.statistical_analysis.pairwise_tukeyhsd") as mock_tukey:
        mock_tukey.return_value = "tukey_result"
        result3 = additive_anova_posthoc(df3, "DV", "Factor", main_effect_p=0.01, robust_type=None)
        mock_tukey.assert_called_once_with(endog=df3["DV"], groups=df3["Factor"], alpha=0.05)
        assert result3 == "tukey_result"

    # Case 4: Unequal variances, should call Games–Howell
    with patch("src.statistical_analysis.pg.pairwise_gameshowell") as mock_gh:
        mock_gh.return_value = "gameshowell_result"
        result4 = additive_anova_posthoc(df3, "DV", "Factor", main_effect_p=0.01, robust_type="hc3")
        mock_gh.assert_called_once_with(data=df3, dv="DV", between="Factor")
        assert result4 == "gameshowell_result"


def test_run_ancova():
    # Sample dataset
    df = pd.DataFrame({
        "DV": [5,6,7,5,6,7,8,9,10,9],
        "IV": ["A","A","A","B","B","B","C","C","C","C"],
        "Cov": [1,2,3,1,2,3,1,2,3,4]
    })

    # Case 1: linearity ok, equal variances (levene_test > alpha)
    model1, table1 = run_ancova(df, dv="DV", iv="IV", covariate="Cov", levene_test=0.2, linearity_p_value=0.01, alpha=0.05)
    assert "partial_eta_sq" in table1.columns
    assert model1 is not None

    # Case 2: linearity violated, levene test not significant
    with patch("src.statistical_analysis.quadratic_model_adjustment") as mock_quad:
        mock_model = smf.ols("DV ~ C(IV) + Cov", data=df).fit()
        mock_quad.return_value = mock_model
        model2, table2 = run_ancova(df, dv="DV", iv="IV", covariate="Cov", levene_test=0.2, linearity_p_value=0.1, alpha=0.05)
        mock_quad.assert_called_once()
        assert "partial_eta_sq" in table2.columns
        assert model2 is not None

    # Case 3: linearity ok, variance violated (levene_test < alpha)
    model3, table3 = run_ancova(df, dv="DV", iv="IV", covariate="Cov", levene_test=0.01, linearity_p_value=0.01, alpha=0.05)
    assert "partial_eta_sq" in table3.columns
    assert hasattr(model3, "resid")  # Should still be a model object


def test_run_ancova_posthoc():
    df = pd.DataFrame({
        "DV": [10, 12, 14, 20, 22, 24, 30, 32, 34],
        "IV": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "Cov": [1, 2, 3, 1, 2, 3, 1, 2, 3]
    })

    model = smf.ols("DV ~ C(IV) + Cov", data=df).fit()

    posthoc = run_ancova_posthoc(
        df=df,
        model=model,
        iv="IV",
        levene_test=0.10,
        alpha=0.05
    )

    assert posthoc is not None

    # Version-safe extraction
    if hasattr(posthoc, "summary_frame"):
        result_df = posthoc.summary_frame()
        assert isinstance(result_df, pd.DataFrame)

    elif hasattr(posthoc, "summary"):
        summary = posthoc.summary()
        assert summary is not None

    elif hasattr(posthoc, "result_frame"):
        result_df = posthoc.result_frame
        assert isinstance(result_df, pd.DataFrame)

    else:
        raise AssertionError("Unknown posthoc result type")

def test_run_moderated_regression_returns_dataframe():
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    n = 100
    df = pd.DataFrame({
        "DV": np.random.normal(size=n),
        "IV": np.random.normal(size=n),
        "Moderator": np.random.normal(size=n)
    })

    df["DV"] = (
        0.5 * df["IV"]
        + 0.3 * df["Moderator"]
        + 0.4 * df["IV"] * df["Moderator"]
        + np.random.normal(scale=0.5, size=n)
    )

    result = run_moderated_regression(
        df=df,
        dv="DV",
        iv="IV",
        moderator="Moderator",
        alpha=0.05
    )

    # ---- Assertions ----
    assert isinstance(result, pd.DataFrame)

    # Core columns always present
    assert "Coef." in result.columns
    assert "Std.Err." in result.columns

    # Accept either t or z stats
    assert ("t" in result.columns) or ("z" in result.columns)

    # Accept either p column
    assert ("P>|t|" in result.columns) or ("P>|z|" in result.columns)

    # Interaction term exists
    assert any("IV:Moderator" in idx for idx in result.index)

def test_run_spotlight_analysis_returns_valid_dataframe():
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # ---- Create synthetic dataset ----
    n = 120
    df = pd.DataFrame({
        "DV": np.random.normal(size=n),
        "IV": np.random.normal(size=n),
        "Moderator": np.random.normal(size=n)
    })

    # Add real interaction signal (important for stable regression)
    df["DV"] = (
        0.6 * df["IV"]
        + 0.4 * df["Moderator"]
        + 0.5 * df["IV"] * df["Moderator"]
        + np.random.normal(scale=0.5, size=n)
    )

    # ---- Run function ----
    result = run_spotlight_analysis(
        df=df,
        dv="DV",
        iv="IV",
        moderator="Moderator"
    )

    # ---- Assertions ----
    assert isinstance(result, pd.DataFrame)

    expected_columns = [
        "Level",
        "Covariate Value",
        "Group Difference (B)",
        "Std. Error",
        "t-stat",
        "p-value"
    ]

    assert all(col in result.columns for col in expected_columns)

    # Should have exactly 3 spotlight rows
    assert len(result) == 3

    # Check correct labels exist
    expected_levels = {
        "Low (-1 SD)",
        "Average (Mean)",
        "High (+1 SD)"
    }

    assert set(result["Level"]) == expected_levels

    # Check numeric outputs are valid
    numeric_cols = [
        "Covariate Value",
        "Group Difference (B)",
        "Std. Error",
        "t-stat",
        "p-value"
    ]

    assert result[numeric_cols].notna().all().all()
