import pytest
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from unittest.mock import patch, MagicMock
from src.statistical_analysis import(
    factor_categorical,
    two_way_anova_test,
    simple_effects_anova,
    report_simple_effects,
    run_posthoc_on_significant_simple_effects,
    additive_anova_posthoc,
    report_significant_posthoc_simple_effects,
    run_ancova,
    run_ancova_posthoc
)

#pytest tests/test_statistical_analysis.py

def test_factor_categorical():
    # Create a small sample dataframe
    df = pd.DataFrame({
        "IV": [1, 2, 1, 2],
        "Factor2": ["A", "B", "A", "B"],
        "DV": [10, 15, 20, 25]
    })

    # Run the function
    df_out = factor_categorical(df, iv="IV", factor2="Factor2")

    # Check that the columns are now categorical
    assert pd.api.types.is_categorical_dtype(df_out["IV"])
    assert pd.api.types.is_categorical_dtype(df_out["Factor2"])

    # Check that other columns remain unchanged
    assert (df_out["DV"] == df["DV"]).all()

    # Check that the function returns a new copy (not the original dataframe)
    assert df_out is not df

def test_two_way_anova_test():
    # Create a small sample dataframe
    df = pd.DataFrame({
        "IV": ["A", "A", "B", "B", "A", "B", "A", "B"],
        "Factor2": ["X", "Y", "X", "Y", "X", "Y", "Y", "X"],
        "DV": [5, 6, 7, 8, 5, 9, 6, 8]
    })

    # Test additive ANOVA with Levene p > alpha (no robust adjustment)
    anova_table, robust_type = two_way_anova_test(df, dv="DV", iv="IV", factor2="Factor2",
                                                  levene_p=0.2, check_interaction=False, alpha=0.05)
    # Check robust_type is None
    assert robust_type is None
    # Check output is a DataFrame and has expected ANOVA columns
    assert isinstance(anova_table, pd.DataFrame)
    assert "F" in anova_table.columns
    assert "PR(>F)" in anova_table.columns

    # Test interactive ANOVA with Levene p < alpha (robust adjustment)
    anova_table2, robust_type2 = two_way_anova_test(df, dv="DV", iv="IV", factor2="Factor2",
                                                    levene_p=0.01, check_interaction=True, alpha=0.05)
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

def test_report_simple_effects(caplog):
    # Sample DataFrame simulating output from simple_effects_anova
    results_df = pd.DataFrame({
        "Level": ["X", "Y"],
        "F": [5.123, 0.456],
        "p": [0.025, 0.65]
    })

    with caplog.at_level("INFO"):
        report_simple_effects(results_df, alpha=0.05)

    # There should be two log messages
    assert len(caplog.records) == 2

    # Check content of the messages
    assert "Simple effect of factor at X" in caplog.records[0].message
    assert "Significant" in caplog.records[0].message

    assert "Simple effect of factor at Y" in caplog.records[1].message
    assert "Not significant" in caplog.records[1].message


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

def test_report_significant_posthoc_simple_effects_cases():
    # Case 1: Empty dictionary, should just return None
    empty_dict = {}
    result1 = report_significant_posthoc_simple_effects(empty_dict)
    assert result1 is None

    # Case 2: Dictionary with no significant results, should log but return None
    df_no_sig = pd.DataFrame({
        "group1": ["A", "A"],
        "group2": ["B", "C"],
        "meandiff": [0.1, 0.2],
        "stat": [1.2, 0.5],
        "p-adj": [0.6, 0.7],
        "lower": [0, 0],
        "upper": [1, 1]
    })
    posthoc_dict_no_sig = {"Level1": df_no_sig}
    result2 = report_significant_posthoc_simple_effects(posthoc_dict_no_sig)
    assert result2 is None

    # Case 3: Dictionary with some significant results, should log but return None
    df_sig = pd.DataFrame({
        "group1": ["A"],
        "group2": ["B"],
        "meandiff": [1.5],
        "stat": [3.2],
        "p-adj": [0.01],
        "lower": [0.5],
        "upper": [2.5]
    })
    posthoc_dict_sig = {"Level1": df_sig}
    result3 = report_significant_posthoc_simple_effects(posthoc_dict_sig)
    assert result3 is None

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
    # Minimal dataset
    df = pd.DataFrame({
        "DV": [10, 12, 15, 20, 22, 25],
        "IV": ["A", "A", "B", "B", "C", "C"],
        "Cov": [1, 2, 1, 2, 1, 2]
    })

    # Fit ANCOVA model
    model = smf.ols("DV ~ C(IV) + Cov", data=df).fit()

    # Run posthoc
    posthoc = run_ancova_posthoc(df, model, iv="IV", levene_test=0.1, alpha=0.05)

    # Instead of checking summary(), check that it has summary_frame()
    summary_df = posthoc.summary_frame()  # this works for pairwise t-test
    assert isinstance(summary_df, pd.DataFrame)
    assert "group1" in summary_df.columns
    assert "group2" in summary_df.columns


