from src.statistical_analysis import gameshowell, one_way_anova
import warnings
from scipy.stats._stats_py import DegenerateDataWarning
import numpy as np
import scipy.stats


import pandas as pd
import pytest

def test_factor_categorical():
    # 1. Create dummy data
    data = {'Group': ['A', 'B', 'A', 'C'], 'Condition': ['High', 'Low', 'High', 'Low'], 'Score': [10, 20, 30, 40]}
    df = pd.DataFrame(data)
    # 2. Run the function
    f1, f2 = factor_categorical(df, 'Group', 'Condition')

    # 3. Assertions
    # Check if the returned objects are Categorical dtypes
    assert f1.dtype == 'category'
    assert f2.dtype == 'category'
    # Check if the change reflected in the original DataFrame
    assert df['Group'].dtype == 'category'
    assert df['Condition'].dtype == 'category'
    # Ensure values remain the same
    assert list(df['Group']) == ['A', 'B', 'A', 'C']

    print("Test Passed!")

def test_summarize():
    # 1. Setup: Known data
    # For [10, 20], Mean=15, SD=7.071, N=2
    test_data = pd.Series([10, 20])
    
    # 2. Execution
    result = summarize(test_data, ci=0.95)
    
    # 3. Assertions
    assert result["N"] == 2
    assert result["Mean_DV"] == 15.0
    assert np.isclose(result["SD_DV"], 7.071067, atol=1e-5)
    
    # Check if CI is calculated (should not be NaN)
    assert not np.isnan(result["CI_95_Lower"])
    
    # Check that SE is correctly derived from SD/sqrt(N)
    expected_se = 7.071067 / np.sqrt(2) # which is 5.0
    assert np.isclose(result["SE_DV"], expected_se)
    
    print("Summarize function passed statistical validation!")

def test_descriptive_table_two_way():
    # 1. Setup: 2x2 design
    data = {
        'Gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
        'Treatment': ['Drug', 'Placebo', 'Drug', 'Placebo', 'Drug', 'Placebo', 'Drug', 'Placebo'],
        'Score': [10, 12, 14, 16, 11, 13, 15, 17]
    }
    df = pd.DataFrame(data)

    # 2. Execution
    result = descriptive_table_two_way(df, 'Score', 'Gender', 'Treatment', ci=0.95)

    # 3. Assertions
    # Check if we have 4 rows (2 genders * 2 treatments)
    assert len(result['Gender'].unique()) == 2
    assert len(result['Treatment'].unique()) == 2
    
    # Check if the summary statistics columns exist
    assert "Mean_DV" in result.columns
    assert "N" in result.columns

    print("Two-way descriptive table function passed!")

def test_descriptive_table_ancova():
    # 1. Setup
    data = {
        'Group': ['A', 'A', 'B', 'B'],
        'Score': [10, 20, 30, 40],       # DV
        'PreTest': [5, 7, 10, 12]        # Covariate
    }
    df = pd.DataFrame(data)
    
    # 2. Execution
    result = descriptive_table_ancova(df, 'Score', 'Group', 'PreTest')

    # 3. Assertions
    # Check if we have both DV and Covariate info
    assert "Mean_DV" in result.columns
    assert "Mean_PreTest" in result.columns
    
    # Check calculation: Group B Mean_PreTest should be (10+12)/2 = 11
    group_b_cov = result.loc[result['Group'] == 'B', 'Mean_PreTest'].values[0]
    assert group_b_cov == 11.0
    
    print("ANCOVA descriptive table function passed!")

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def test_anova_model_logic():
    # 1. Setup dummy data
    data = {
        'Score': [10, 12, 11, 20, 22, 21, 30, 32, 31, 40, 42, 41],
        'GroupA': ['A1', 'A1', 'A1', 'A1', 'A1', 'A1', 'A2', 'A2', 'A2', 'A2', 'A2', 'A2'],
        'GroupB': ['B1', 'B1', 'B1', 'B2', 'B2', 'B2', 'B1', 'B1', 'B1', 'B2', 'B2', 'B2']
    }
    df = pd.DataFrame(data)

    # 2. Test Case 1: Interaction + Robust (Levene Significant)
    res_robust = anova_model(df, 'Score', 'GroupA', 'GroupB', 'significant', True)
    # Check if interaction term exists in index
    assert 'C(GroupA):C(GroupB)' in res_robust.index
    
    # 3. Test Case 2: No Interaction + Standard (Levene Insignificant)
    res_std = anova_model(df, 'Score', 'GroupA', 'GroupB', 'insignificant', False)
    assert 'C(GroupA):C(GroupB)' not in res_std.index

    print("ANOVA model selector function passed logic test!")

def test_simple_effects():
    # 1. Setup: Group A has a difference, Group B does not (Interaction)
    data = {
        'Score': [10, 11, 12, 10, 11, 12, 50, 52, 51, 10, 11, 12],
        'FactorA': ['Level1']*6 + ['Level2']*6,
        'FactorB': (['B1']*3 + ['B2']*3) * 2
    }
    df = pd.DataFrame(data)
    df['FactorA'] = df['FactorA'].astype('category')
    df['FactorB'] = df['FactorB'].astype('category')

    # 2. Execution
    # Testing the 'equal variance' path (negative Levene)
    results = simple_effects_tukey(df, 'Score', 'FactorA', 'FactorB', levine_test="negative")

    # 3. Assertions
    assert 'FactorA' in results
    assert 'FactorB' in results
    # Check if ANOVA was performed for FactorA at Level1
    assert 'anova' in results['FactorA']['Level1']
    
    # Check if Tukey was triggered for FactorA at Level2 (where the difference is huge)
    assert 'posthoc' in results['FactorA']['Level2']

    print("Simple effects function passed validation!")


from src.statistical_analysis import simple_effects_tukey
print(test_simple_effects)

import pandas as pd
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def test_posthoc_main_effect():
    # Setup: 3 levels so post-hoc is eligible
    data = {
        'Score': [10, 11, 12, 20, 21, 22, 50, 51, 52],
        'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    }
    df = pd.DataFrame(data)

    # Test Case 1: Non-significant main effect (p = 0.10)
    print("\nTest 1:")
    res1 = posthoc_main_effect(df, 'Score', 'Group', main_effect_p=0.10, levene_test=0.50)
    assert res1 is None

    # Test Case 2: Only 2 levels
    print("\nTest 2:")
    df_small = df[df['Group'] != 'C']
    res2 = posthoc_main_effect(df_small, 'Score', 'Group', main_effect_p=0.01, levene_test=0.50)
    assert res2 is None

    # Test Case 3: Significant + Equal Variance (Tukey)
    print("\nTest 3:")
    res3 = posthoc_main_effect(df, 'Score', 'Group', main_effect_p=0.01, levene_test=0.60)
    assert isinstance(res3, type(pairwise_tukeyhsd(df['Score'], df['Group'])))

    # Test Case 4: Significant + Unequal Variance (Games-Howell)
    print("\nTest 4:")
    res4 = posthoc_main_effect(df, 'Score', 'Group', main_effect_p=0.01, levene_test=0.01)
    assert isinstance(res4, pd.DataFrame) # Pingouin returns a DataFrame for Games-Howell

    print("\nPost-hoc logic function passed all branch tests!")

def test_run_ancova():
    # 1. Setup: Create data with 1 IV and 1 Covariate
    data = {
        'Score': [10, 15, 12, 20, 25, 22, 30, 35, 32], # DV
        'Group': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'], # IV
        'Age': [20, 22, 21, 20, 22, 21, 20, 22, 21] # Covariate
    }
    df = pd.DataFrame(data)

    # 2. Execution
    model, table = run_ancova(df, 'Score', 'Group', ['Age'], levene_test="insignificant")

    # 3. Assertions
    assert "partial_eta_sq" in table.columns
    assert "C(Group)" in table.index
    assert "Age" in table.index
    # Check if Partial Eta Squared is between 0 and 1
    assert 0 <= table.loc["C(Group)", "partial_eta_sq"] <= 1

    print("ANCOVA runner function passed validation!")

import pandas as pd
import statsmodels.formula.api as smf

def test_run_moderated_regression():
    # 1. Setup data
    data = {'Y': [1, 2, 3, 4, 5, 6], 'X': [0, 0, 0, 1, 1, 1], 'Z': [2, 4, 6, 8, 10, 12]}
    df = pd.DataFrame(data)

    # 2. Test Case: Robust Model
    # We pass 'Y', 'X', 'Z' and 'significant'
    model = run_moderated_regression(df, 'Y', 'X', 'Z', 'significant')

    # 3. Assertions
    # Check if interaction term 'X:Z' exists in the model params
    assert 'X:Z' in model.params.index
    # Check if cov_type is correctly set to HC3
    assert model.cov_type == 'HC3'

    print("Moderated regression function passed validation!")


To test a Spotlight Analysis function, we need a dataset where the effect of the Independent Variable (IV) changes depending on the level of the Covariate.In this test, I have designed the data so that there is no difference between groups when the covariate is low, but a huge difference when the covariate is high. This will allow us to verify that your function correctly identifies different $p$-values for each "spot."Python Test for Spotlight AnalysisPythonimport pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def test_run_spotlight_analysis():
    # 1. Create a synthetic dataset
    # We want Group B to be higher than Group A ONLY as 'IQ' increases
    np.random.seed(42)
    n = 100
    iq = np.random.normal(100, 15, n)
    group = np.random.choice(['Control', 'Treatment'], n)
    
    # Generate Score: Baseline + Group Effect + IQ effect + Interaction
    # The interaction (group_binary * iq) ensures slopes are NOT parallel
    group_binary = (group == 'Treatment').astype(int)
    score = 10 + (2 * group_binary) + (0.5 * iq) + (0.8 * group_binary * (iq - 100)) + np.random.normal(0, 2, n)
    
    df = pd.DataFrame({'Score': score, 'Group': group, 'IQ': iq})

    # 2. Run your function
    # We'll test both the standard and robust (HC3) paths
    print("Testing Standard Path...")
    results_std = run_spotlight_analysis(df, 'Score', 'Group', 'IQ', 'insignificant')
    
    print("\nTesting Robust Path (Levene Significant)...")
    results_robust = run_spotlight_analysis(df, 'Score', 'Group', 'IQ', 'significant')

    # 3. Assertions
    # Check that we got 3 rows (Low, Mean, High)
    assert len(results_std) == 3
    assert 'Group Difference (B)' in results_std.columns
    
    # Check logic: In this dataset, the 'High' spot should have a 
    # much larger 'Group Difference (B)' than the 'Low' spot.
    high_diff = results_std.loc[results_std['Level'] == 'High (+1 SD)', 'Group Difference (B)'].values[0]
    low_diff = results_std.loc[results_std['Level'] == 'Low (-1 SD)', 'Group Difference (B)'].values[0]
    
    assert high_diff > low_diff
    
    print("\nSpotlight Analysis test passed! Logic and variance paths are working.")
