import logging

logger = logging.getLogger(__name__)

import pingouin as pg
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#TODO: things in common between both tests: Data describe
def factor_categorical(df, factor1, factor2):
    df = df.copy()
    df[factor1] = df[factor1].astype("category")
    df[factor2] = df[factor2].astype("category")
    return df




#TODO: TWO_WAY_ANOVA homogenous variance and welch
def anova_model(df, dv, factor1, factor2, levene_test, check_interaction, alpha=0.05):

    if check_interaction == True: #defines which model to use

        # Fit OLS model(Ordinary Least Squares), we define which sum of squares to includes
        model = ols(f'{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})', data=df).fit() #two-way factorial model
        if levene_test < alpha: #typ=3 runs anova model with interaction
            print("Running Two Way ANOVA with equal variances assumed")
            anova_table = anova_lm(model, typ=3, robust='hc3')  # if robust = hc3, it runs the test with adjustment to the group variance 
        else:
            print("Running Two Way ANOVA adjusted for unequal variance")
            anova_table = anova_lm(model, typ=3)  
    else:
        model = ols( f'{dv} ~ C({factor1}) + C({factor2})', data=df).fit() #additive model, typ=2 ignores interactions
        if levene_test < alpha:
            print("Running Additive Two Way ANOVA adjusted for unequal variance")           
            anova_table = anova_lm(model, typ=2, robust='hc3')  
        else:
            print("Running Additive Two Way ANOVA with equal variance assumed")
            anova_table = anova_lm(model, typ=2)  
    return anova_table    

def simple_effects_tukey(df, dv, factor1, factor2, alpha=0.05, levine_test): #in case the interaction is significant, check simple effects
    factors = [factor1, factor2]
    results = {}
    for factor in factors:
        #creates a new list for the values that aren't equal to the 
        # factor we're working on, and selects the first value

        other_factor = [f for f in factors if f != factor][0] 

        #checks if factor has more than one category, which is essential for checking simple effects 
        if df[factor].astype("category").cat.categories.size > 1: 
            results.setdefault(factor, {})  #creates a dictionary which opens with each factor value

            for level in df[factor].cat.categories: #goes over the levels in the factor
                sub_df = df[df[factor] == level]
                # Simple effect ANOVA for factor1 at this level
                model_sub = ols(f'{dv} ~ C({other_factor})', data=sub_df).fit()                  
                if levine_test > alpha:
                    print("Running simple effect ANOVA with variance assumed")
                    anova_sub = anova_lm(model_sub, typ=3)                
                else:
                    print("Running simple effect ANOVA adjusted for unequal variance")                    
                    anova_sub = anova_lm(model_sub, typ=3, robust="hc3")
                    

                results[factor][level] = {'anova': anova_sub}
                print(anova_sub)


                if anova_sub.loc[f"C({other_factor})", "PR(>F)"] < 0.05:
                    print(f"Simple effect detected significant.\n")
                    if levine_test > alpha:
                    # Tukey post-hoc for all pairwise comparisons of factor1
                        print(f"Running Tukey test with equal variences assumed")                     
                        tukey = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[other_factor], alpha=alpha)                    
                        results[factor][level]['posthoc'] = tukey
                    else:
                        (f"Running Games-Howell test for samples in need of varience adjustment")
                        gameshowell = pg.pairwise_gameshowell(data=sub_df,dv=dv,between=other_factor)
                        results[factor][level]['posthoc'] = gameshowell
    return results


def posthoc_main_effect(df,dv,factor,main_effect_p,levene_test,alpha=0.05):
    #Main effect must be significant
    if main_effect_p >= alpha:
        print(f"No post-hoc tests: main effect of '{factor}'\n")
        print(f"is not significant (p = {main_effect_p:.3f}).")
        return None
    # Two levels indicate that no post-hoc needed
    n_levels = df[factor].nunique()

    if n_levels == 2:
        print(
            f"No post-hoc tests needed: '{factor}' has only two levels.")
        return None
    # If we have equal variances we'll conduct tukey
    if levene_test >= alpha:
        print(f"Running Tukey HSD for '{factor} with equal variance assumed'.")
        return pairwise_tukeyhsd(endog=df[dv],groups=df[factor], alpha=alpha)
    #Unequal variances Games–Howell
    else:
        print(f"Running Games–Howell for '{factor}' adjusted for unequal variance.")
        return pg.pairwise_gameshowell(data=df,dv=dv,between=factor)
    
#TODO: ANCOVA

#Adjusted quadratic model:

def quadratic_model_adjustment(df, dv, iv,covariate):
    df = df.copy()

    # Center the covariate (VERY important for stability)
    cov_c = f"{covariate}_c"
    cov_c_sq = f"{covariate}_c_sq"

    df[cov_c] = df[covariate] - df[covariate].mean()
    df[cov_c_sq] = df[cov_c] ** 2
    model = ols(f"{dv} ~ C({iv}) + {cov_c} + {cov_c_sq}", data=df).fit()

    return model



# def run_ancova(data, dv, iv, covariate, levene_test, alpha=0.05, linearity_p_value):
#     if linearity_p_value < 0.05:
#         # Fit model
#         model = smf.ols(f"{dv} ~ C({iv}) + {covariate}", data=data).fit()
#     else:
#         model = quadratic_model_adjustment(data, dv, iv, covariate)
#         # Variance adjustment
#         if levene_test < alpha:
#             model = model.get_robustcov_results(cov_type="HC3")
#             print("Running ANCOVA adjusted for unequal variance")
#         else:
#             print(f"Equal variances assumed (Levene p = {levene_test:.3f}).\n")
#             print("Running ANCOVA test")

        
#     #Run ANCOVA test
#     ancova_table = sm.stats.anova_lm(model, typ=2)

#     # Calculate partial eta squared
#     ancova_table["partial_eta_sq"] = (ancova_table["sum_sq"] /(ancova_table["sum_sq"] + ancova_table.loc["Residual", "sum_sq"]))


#     anova_table = sm.stats.anova_lm(model, typ=2)
        
#     return model, ancova_table

def run_ancova(data, dv, iv, covariate, levene_test, linearity_p_value, alpha=0.05):

    # If linearity is violated → use quadratic model
    if linearity_p_value < alpha:
        model = quadratic_model_adjustment(data, dv, iv, covariate)
        model_type = "Quadratic ANCOVA"
        
    else:
        model = smf.ols(f"{dv} ~ C({iv}) + {covariate}", data=data).fit()
        model_type = "Linear ANCOVA"


    # Variance adjustment (apply to BOTH models)
    if levene_test < alpha:
        model = model.get_robustcov_results(cov_type="HC3")
        print(f"Running {model_type} using HC3 robust SEs")
    else:
        print(f"{model_type} with equal variances assumed")

    # ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)

    # Partial eta squared
    ancova_table["partial_eta_sq"] = (
        ancova_table["sum_sq"] /
        (ancova_table["sum_sq"] + ancova_table.loc["Residual", "sum_sq"])
    )

    return model, ancova_table


def run_ancova_with_statsmodels_posthoc(data, dv, iv, covariate, levene_test=None, alpha=0.05):
    # Check that IV has at least 3 levels
    k = data[iv].nunique()
    if k < 3:
        raise ValueError(f"Post-hoc requires IV to have at least 3 levels. Found only {k} level(s).")
    
    # Fit the model (using C() to ensure IV is categorical)
    if levene_test is not None and levene_test < alpha:
        model = smf.ols(f"{dv} ~ C({iv}) + {covariate}", data=data).fit(cov_type="HC3")
    else:
        model = smf.ols(f"{dv} ~ C({iv}) + {covariate}", data=data).fit()
    
    # Perform Pairwise T-Tests (Post-hoc)
    posthoc = model.t_test_pairwise(term_name=f"C({iv})", method="bonferroni")
    
    return posthoc.result_frame





def run_moderated_regression(df, dv, iv, covariate): #No need for levene test because robust regression is always preffered
    model = smf.ols(formula=f"{dv} ~ {iv} * {covariate}", data=df).fit(cov_type="HC3")
    print("Running moderated regression test")

    #Create a table for the results
    moderated_regression_table = model.summary2().tables[1]  
    print("\nModerated Regression Results Table")
    print(table)
    return moderated_regression_table



def run_spotlight_analysis(df, dv, iv, covariate):
    # Check that IV has at least 3 levels
    k = df[iv].nunique()
    if k < 3:
        raise ValueError(f"Spotlight analysis requires IV to have at least 3 levels. Found only {k} level(s).")
    
    # Calculate the 'spots' (Mean, +1SD, -1SD)
    mean_cov = df[covariate].mean()
    sd_cov = df[covariate].std()
    
    spots = {
        'Low (-1 SD)': mean_cov - sd_cov,
        'Average (Mean)': mean_cov,
        'High (+1 SD)': mean_cov + sd_cov
    }
    
    results = []

    # Run the model for each spot by centering the covariate
    for label, value in spots.items():
        # Center the covariate at the specific spot
        df['temp_centered'] = df[covariate] - value
        
        # Fit the interaction model using robust SEs
        model = smf.ols(f"{dv} ~ {iv} * temp_centered", data=df).fit(cov_type='HC3')
        
        # Extract the coefficient and stats for the IV
        iv_row = [row for row in model.params.index if iv in row and ':' not in row and 'Intercept' not in row][0]
        
        results.append({
            'Level': label,
            'Covariate Value': round(value, 3),
            'Group Difference (B)': round(model.params[iv_row], 4),
            'Std. Error': round(model.bse[iv_row], 4),
            't-stat': round(model.tvalues[iv_row], 4),
            'p-value': round(model.pvalues[iv_row], 4)
        })

    # Display Results
    results_df = pd.DataFrame(results)
    return results_df
