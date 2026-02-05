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
from src.app_logger import logger


#TODO: things in common between both tests: Data describe
def factor_categorical(df, iv, factor2):
    logger.debug("Variable converted to factor")
    df = df.copy()

    # Converts variables into categorical variables
    df[iv] = df[iv].astype("category")
    df[factor2] = df[factor2].astype("category")

    return df



#TODO: TWO_WAY_ANOVA homogenous variance and welch
def two_way_anova_test(df, dv, iv, factor2, levene_p, check_interaction, alpha=0.05):

    """

    Function conducts 2-Way ANOVA.
    Based on the input of check_interaction, it conducts either an interactive model
    or an additive model, where typ=3 -> interactive, typ=2 -> additive, while also 
    adjesting variances in case levene's test p-value was significant, where:
    robust='hc3'-> adjusts variance.
    Function then returns a datafram of the ANOVA test.

    
    """

    robust_type = "hc3" if levene_p < alpha else None

    if check_interaction == True: #Interactive model

        formula = f'{dv} ~ C({iv}) + C({factor2}) + C({iv}):C({factor2})' #Interactive model formula
        anova_typ = 3 # Important for running interactive model in anova_lm

        if robust_type != None:
            logger.debug("Running Interactive Two Way ANOVA model adjusted for unequal variance")
        else:
            logger.debug("Running Interactive Two Way ANOVA model with equal variances assumed")

    else: #Additive model

        formula = f'{dv} ~ C({iv}) + C({factor2})'
        anova_typ = 2 # Important for running additive model in anova_lm

        if robust_type != None:
            logger.debug("Running Additive Two Way ANOVA model adjusted for unequal variance")
        else:
            logger.debug("Running Additive Two Way ANOVA model with equal variances assumed")

    model = ols(formula, data=df).fit() 

    anova_table = anova_lm(model, typ=anova_typ, robust=robust_type)

    return anova_table



# Two Way ANOVA post hoc
def simple_effects_tukey(df, dv, factor1, iv, alpha=0.05, levine_p=None): 
    """
    
    In case the interaction is significant, check simple effects

    """
    robust_type = "hc3" if levene_p < alpha else None

    factors = [factor1, iv]

    results = {}
    for factor in factors:

        # Select the other factor we're looking into, useful for conducting simple effect test
        other_factor = [f for f in factors if f != factor][0] 

        # Checks if factor has more than one categorywhich is essential for checking simple effects 

        validation_check = validate_anova_inputs(df, dv, factors)

        results.setdefault(factor, {})  #creates a dictionary which opens with each factor value

        for level in df[factor].unique(): #goes over the levels in the factor
            sub_df = df[df[factor] == level]

            # Simple effect ANOVA for factor1 at this level

            model_sub = ols(f'{dv} ~ C({other_factor})', data=sub_df).fit()

            anova_sub = anova_lm(model_sub, typ=2, robust = robust_type)

            if robust_type == None:
                logger.debug("Running simple effect ANOVA with variance assumed")
            else:
                logger.debug("Running simple effect ANOVA adjusted for unequal variance")

            results[factor][level] = {'anova': anova_sub}

    return results
        
                # print(anova_sub)
            
            #return anova_sub, other_factor, sub_df, robust_type,
        

def anova_simple_effects(df, results, factors, robust_type):

    for factor in factors:

        # Select the other factor we're looking into, useful for conducting simple effect test
        other_factor = [f for f in factors if f != factor][0] 

        for level in df[factor].unique():

        if results[factor][level]['anova'].loc[f"C({other_factor})", "PR(>F)"] < 0.05:
             print(f"Simple effect detected significant.\n")
            if robust_type == None:
            # Tukey post-hoc for all pairwise comparisons of factor1

                logger.debug("Simple effect detected significant. Running Tukey test with equal variences assumed") 

                tukey = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[other_factor], alpha=alpha)

                results[factor][level]['posthoc'] = tukey

            else:
                logger.debug("Simple effect detected significant. " \
                "Running Games-Howell test for samples in need of varience adjustment")    

                gameshowell = pg.pairwise_gameshowell(data=sub_df,dv=dv,between=other_factor)

                results[factor][level]['posthoc'] = gameshowell

    return results
        
 def tukey_simple_effects(df, results, dv, factors, robust_type, alpha=0.05):
    for factor in factors:
        # Identify the factor being compared
        other_factor = [f for f in factors if f != factor][0] 

        for level in df[factor].unique():
            # Pull the ANOVA p-value from the results dictionary
            anova_res = results[factor][level]['anova']
            p_val = anova_res.loc[f"C({other_factor})", "PR(>F)"]

            if p_val < alpha:
                print(f"Simple effect for {factor} at level {level} is significant.")
                
                # Re-create the subset of data for this level
                sub_df = df[df[factor] == level]

                if robust_type is None:
                    # Equal Variances, thus we conduct Tukey
                    logger.debug(f"Running Tukey for {other_factor} at {factor}={level}") 
                    tukey = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[other_factor], alpha=alpha)
                    results[factor][level]['posthoc'] = tukey
                else:
                    # Unequal Variances, thus we conduct Games-Howell
                    logger.debug(f"Running Games-Howell for {other_factor} at {factor}={level}")    
                    gameshowell = pg.pairwise_gameshowell(data=sub_df, dv=dv, between=other_factor)
                    results[factor][level]['posthoc'] = gameshowell

    return results
   



def tukey_additive_anova(df,dv,factor,main_effect_p,robust_type, alpha = 0.05):

    validate_anova_inputs(df, dv, factor)

    #Main effect must be significant
    if main_effect_p >= alpha:
        logger.debug("No post-hoc tests: main effect of '{factor}'\n 
                     f"{factor} is not significant (p = {main_effect_p:.3f}).")
        return None
    # Two levels indicate that no post-hoc needed
    n_levels = df[factor].nunique()

    if n_levels == 2:
        logger.debug(f"No post-hoc tests needed: '{factor}' has only two levels.")
        return None
    
    # If we have equal variances we'll conduct tukey
    if robust_type == None:
        logger.debug(f"Running Tukey HSD for '{factor} with equal variance assumed'.")
        return pairwise_tukeyhsd(endog=df[dv],groups=df[factor], alpha=alpha)
    
    #Unequal variances Games–Howell
    else:
        print(f"Running Games–Howell for '{factor}' adjusted for unequal variance.")
        logger.debug("Running Games-Howel")
        return pg.pairwise_gameshowell(data=df,dv=dv,between=factor)
    

#Adjusted quadratic model:

def quadratic_model_adjustment(df, dv, iv,covariate):
    df = df.copy()

    # Center the covariate (VERY important for stability)
    cov_c = f"{covariate}_c"
    cov_c_sq = f"{covariate}_c_sq"

    df[cov_c] = df[covariate] - df[covariate].mean()
    df[cov_c_sq] = df[cov_c] ** 2
    model = ols(f"{dv} ~ C({iv}) + {cov_c} + {cov_c_sq}", data=df).fit()
    logger.debug  ("Running quadratic model adjustment")

    return model



def run_ancova(data, dv, iv, covariate, levene_test, linearity_p_value, alpha=0.05):

    # If linearity is violated use quadratic model
    if linearity_p_value > alpha:
        model = quadratic_model_adjustment(data, dv, iv, covariate)
        model_type = "Quadratic ANCOVA"
    else:
        model = smf.ols(f"{dv} ~ C({iv}) + {covariate}", data=data).fit()
        model_type = "Linear ANCOVA"
    # Variance adjustment (apply to BOTH models)
    if levene_test < alpha:
        variance_assumption = "using HC3 robust SEs"
        model = model.get_robustcov_results(cov_type="HC3")
    else:
        variance_assumption = "with equal variances assumed"
    logger.debug(f"{model_type} {variance_assumption}")

    # ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)
    # Partial eta squared
    ancova_table["partial_eta_sq"] = (
        ancova_table["sum_sq"] /
        (ancova_table["sum_sq"] + ancova_table.loc["Residual", "sum_sq"])
    )
    return model, ancova_table


def run_ancova_with_statsmodels_posthoc(data, model, dv, iv, covariate, levene_test=None, alpha=0.05):
    
    df = validate_ancova_inputs(df, dv, iv, covariate)

    # Fit the model (using C() to ensure IV is categorical)
    if levene_test is not None and levene_test < alpha:
        model = model.get_robustcov_results(cov_type="HC3")
    
    # Perform Pairwise T-Tests (Post-hoc)
    posthoc = model.t_test_pairwise(term_name=f"C({iv})", method="bonferroni")

    logger.debug ("Running Pairwise T-Tests")
    
    return posthoc




#No need for levene test because robust regression is always preffered
def run_moderated_regression(df, dv, iv, covariate): 
    model = smf.ols(formula=f"{dv} ~ {iv} * {covariate}", data=df).fit(cov_type="HC3")
    logger.debug  ("Running moderated regression test")

    #Create a table for the results
    moderated_regression_table = model.summary2().tables[1]  
    logger.debug  ("Returning a moderated regression table")
    return moderated_regression_table



def run_spotlight_analysis(df, dv, iv, covariate):
    # Check that IV has at least 3 levels
    # k = df[iv].nunique()
    # if k < 3:
    #     raise ValueError(f"Spotlight analysis requires IV to have at least 3 levels. Found only {k} level(s).")
    
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
    logger.debug("Conducting spotlight analysis")
    return results_df
