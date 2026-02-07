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

    return anova_table, robust_type


def report_two_way_anova(anova_table, iv, factor2, alpha=0.05, check_interaction=True):
    """
    Runs a two-way additive ANOVA and reports the results.

    Parameters
    ----------
    dv : str
        Dependent variable.
    iv : str
        First independent variable (factor).
    factor2 : str
        Second independent variable (factor).
    alpha : float
        Significance threshold.
    Returns
    -------
    anova_table : pd.DataFrame
        ANOVA table with sum_sq, df, F, PR(>F) columns.
    """

    # Report results
    if check_interaction == True:
        f_val_int = anova_table.loc[f"C({iv}):C({factor2})", "F"]
        p_val_int = anova_table.loc[f"C({iv}):C({factor2})", "PR(>F)"]
        p_eta_sq_int = anova_table.loc[f"C({iv}):C({factor2})", "partial_eta_sq"]

        if p_val_int < alpha:
            message = f"{C({iv}):C({factor2})}: F = {f_val_int:.3f}, p = {p_val_int:.4f}, partial η² = {p_eta_sq_int},  Significant at alpha={alpha}"
        else:
            message = f"{C({iv}):C({factor2})}: F = {f_val_int:.3f}, p = {p_val:.4f}, partial η² = {p_eta_sq_int},  Not significant at alpha={alpha}"


    for term in [iv, factor2]:
        f_val = anova_table.loc[f"C({term})", "F"]
        p_val = anova_table.loc[f"C({term})", "PR(>F)"]
        p_eta_sq = anova_table.loc[f"C{term})", "partial_eta_sq"]


        if p_val < alpha:
            message = f"{term}: F = {f_val:.3f}, p = {p_val:.4f}, partial η² = {p_eta_sq},  Significant at alpha={alpha}"
        else:
            message = f"{term}: F = {f_val:.3f}, p = {p_val:.4f}, partial η² = {p_eta_sq},  Not significant at alpha={alpha}"

        logger.info(message)


# Two Way ANOVA post hoc        

def simple_effects_anova(df, dv, factor1, factor2, alpha=0.05):
    """
    Computes simple effects for a 2-way ANOVA interaction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Your dataset.
    dv : str
        Dependent variable.
    factor1 : str
        Factor for which we want simple effects.
    factor2 : str
        Moderator factor (levels to subset).
    alpha : float, optional
        Significance threshold, by default 0.05.
        
    Returns
    -------
    results_df : pd.DataFrame
        A dataframe with factor1 effect at each level of factor2.
        Columns: ['Level', 'F', 'p', 'significant']
    """
    results = []

    for level in df[factor2].unique():
        sub = df[df[factor2] == level]
        
        # Fit one-way ANOVA for factor1 at this level of factor2
        model = smf.ols(f"{dv} ~ C({factor1})", data=sub).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        f_val = anova_table.loc[f"C({factor1})", "F"]
        p_val = anova_table.loc[f"C({factor1})", "PR(>F)"]
        
        
        results.append({
            "Level": level,
            "F": round(f_val, 3),
            "p": round(p_val, 4),
            "significant": sig

        })

    
    results_df = pd.DataFrame(results)
    return results_df


def report_simple_effects(results_df, alpha=0.05):
    """
    Reports the results of simple effects from a 2-way ANOVA interaction.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of simple_effects_anova. Must contain columns: ['Level', 'F', 'p', 'significant'].
    alpha : float
        Significance threshold for reporting.

    Returns
    -------
    None
    """
    if results_df.empty:
        logger.info("No simple effects to report.")
        return

    for _, row in results_df.iterrows():
        level = row["Level"]
        f_val = row["F"]
        p_val = row["p"]
        significant = row["significant"]

        if significant:
            message = (
                f"Simple effect of factor at {level}: "
                f"F = {f_val:.3f}, p = {p_val:.4f} → Significant at alpha={alpha}"
            )
        else:
            message = (
                f"Simple effect of factor at {level}: "
                f"F = {f_val:.3f}, p = {p_val:.4f} → Not significant at alpha={alpha}"
            )

        logger.info(message)




    
def run_posthoc_on_significant_simple_effects(
    df, dv, iv, factor2, simple_effects_results, levene_p, alpha = 0.05):
    """
    Run post-hoc tests only for significant simple effects.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    dv : str
        Dependent variable column name.
    factor : str
        Factor for which simple effects were calculated.
    simple_effects_results : pd.DataFrame
        DataFrame with columns ['Level', 'F', 'p', 'significant'].
    alpha : float
        Significance threshold.
    posthoc_method : str
        Which posthoc to run: "tukey" or "gameshowell".
    second_factor : str, optional
        The second factor to compare within each level of the first factor. Required for Tukey.

    Returns
    -------
    posthoc_dict : dict
        Keys = significant levels, values = posthoc DataFrames
    """
    # Filter only significant simple effects
    sig_effects = simple_effects_results[simple_effects_results["significant"]]

    if sig_effects.empty:
        logger.debug(f"No significant simple effects to run post-hoc at alpha={alpha}.")
        return {}
    
    posthoc_dict = {}

    for _, row in sig_effects.iterrows():
        level = row["Level"]
        logger.debug(f"Running post-hoc for significant simple effect at {level}...")

        # Subset the data to this level
        sub_df = df[df[iv] == level]

        if levene_p > alpha:
            if second_factor is None:
                raise ValueError("second_factor must be provided for Tukey post-hoc")
            posthoc = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[factor2])
            posthoc_df = pd.DataFrame(data=posthoc.summary().data[1:], columns=posthoc.summary().data[0])

        else:
            if second_factor is None:
                second_factor = factor  # Default to factor if no second factor is provided
            posthoc_df = pg.pairwise_gameshowell(data=sub_df, dv=dv, between=factor2)


        posthoc_dict[level] = posthoc_df

    return posthoc_dict

def additive_anova_posthoc(df,dv,factor,main_effect_p,robust_type, alpha = 0.05):

    validate_anova_inputs(df, dv, factor)

    #Main effect must be significant
    if main_effect_p >= alpha:
        logger.debug("No post-hoc tests: main effect of '{factor}'\n {factor} is not significant (p = {main_effect_p:.3f}).")
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

def report_significant_posthoc_simple_effects(posthoc_dict, alpha=0.05):
    """
    Report significant post-hoc results from a dictionary of posthoc DataFrames.

    Parameters
    ----------
    posthoc_dict : dict
        Keys = levels of simple effects, values = posthoc DataFrames
    alpha : float
        Significance threshold
    """
    if not posthoc_dict:
        logger.debug("No post-hoc results to report.")
        return

    for level, df in posthoc_dict.items():
        # Some posthoc tables use 'p-adj', others 'pval'; adjust accordingly
        p_col = "p-adj" if "p-adj" in df.columns else "pval"
        sig_df = df[df[p_col].astype(float) < alpha]

        if sig_df.empty:
            logger.debug(f"No significant post-hoc differences for simple effect level '{level}'.\n")
            continue

        logger.debug(f"Significant post-hoc results for simple effect level '{level}':")
        for _, row in sig_df.iterrows():
            group1 = row.get("group1", row.get("A", ""))
            group2 = row.get("group2", row.get("B", ""))
            meandiff = row.get("meandiff", row.get("diff", float("nan")))
            stat = row.get("stat", float("nan"))
            pval = row.get(p_col, float("nan"))
            lower = row.get("lower", float("nan"))
            upper = row.get("upper", float("nan"))

            logger.debug(f"  Comparison: {group1} vs {group2}")
            logger.debug(f"    Mean Diff: {meandiff:.3f}")
            logger.debug(f"    t-stat: {stat:.3f}, p-value: {pval:.4f}")
            logger.debug(f"    95% CI: [{lower:.3f}, {upper:.3f}]")

def report_posthoc_additive_anova(posthoc_results, alpha=0.05):
    """
    Reports the significant post-hoc results from a Tukey HSD or Games–Howell test.

    Parameters
    ----------
    posthoc_results : statsmodels or pingouin object
        Output from posthoc_additive_anova().
        - Tukey: statsmodels.sandbox.stats.multicomp.TukeyHSDResults
        - Games–Howell: pandas.DataFrame (pingouin output)
    alpha : float
        Significance threshold for reporting.


    """

    # Handle Tukey HSD
    if isinstance(posthoc_results, pairwise_tukeyhsd().__class__):
        # Convert to DataFrame
        df = pd.DataFrame(data=posthoc_results.summary().data[1:], columns=posthoc_results.summary().data[0])
        df['reject'] = df['reject'].astype(bool)
        sig_df = df[df['reject']].copy()

        if sig_df.empty:
            logger.info("No significant post-hoc comparisons found (Tukey HSD).")
        else:
            for _, row in sig_df.iterrows():
                logger.info(f"Significant comparison: {row['group1']} vs {row['group2']}")
                logger.info(f"  Mean difference: {row['meandiff']:.3f}, p = {row['p-adj']:.4f}")
                logger.info(f"  95% CI: [{row['lower']:.3f}, {row['upper']:.3f}]")
    
    # Handle Games–Howell 
    elif isinstance(posthoc_results, pd.DataFrame):
        sig_df = posthoc_results[posthoc_results['p-val'] < alpha].copy()
        if sig_df.empty:
            logger.info("No significant post-hoc comparisons found (Games–Howell).")
        else:
            for _, row in sig_df.iterrows():
                logger.info(f"Significant comparison: {row['A']} vs {row['B']}")
                logger.info(f"  Mean difference: {row['mean diff']:.3f}, p = {row['p-val']:.4f}")
                logger.info(f"  95% CI: [{row['CI95%'][0]:.3f}, {row['CI95%'][1]:.3f}]")
    




    
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


def run_ancova_posthoc(data, model, dv, iv, covariate, levene_test=None, alpha=0.05):
    
    df = validate_ancova_inputs(df, dv, iv, covariate)

    # Fit the model (using C() to ensure IV is categorical)
    if levene_test is not None and levene_test < alpha:
        model = model.get_robustcov_results(cov_type="HC3")
    
    # Perform Pairwise T-Tests (Post-hoc)
    posthoc = model.t_test_pairwise(term_name=f"C({iv})", method="bonferroni")

    logger.debug ("Running Pairwise T-Tests")
    
    return posthoc




#No need for levene test because robust regression is always preffered
def run_moderated_regression(df, dv, iv, moderator, alpha): 
    model = smf.ols(formula=f"{dv} ~ {iv} * {moderator}", data=df).fit(cov_type="HC3")
    logger.debug  ("Running moderated regression test")

    #Create a table for the results
    moderated_regression_table = model.summary2().tables[1]  
    logger.debug  ("Returning a moderated regression table")
    return moderated_regression_table



def run_spotlight_analysis(df, dv, iv, moderator):
    
    # Calculate the 'spots' (Mean, +1SD, -1SD)
    mean_cov = df[moderator].mean()
    sd_cov = df[moderator].std()
    
    spots = {
        'Low (-1 SD)': mean_cov - sd_cov,
        'Average (Mean)': mean_cov,
        'High (+1 SD)': mean_cov + sd_cov
    }
    
    results = []

    # Run the model for each spot by centering the covariate
    for label, value in spots.items():
        # Center the covariate at the specific spot
        df['temp_centered'] = df[moderator] - value
        
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

def report_spotlight_results(results_df, iv, dv, moderator, alpha=0.05):
    """
    Generate a textual report for spotlight (simple effects) analysis
    in a moderated regression.

    """

    sentences = []
    significant_levels = []

    for _, row in results_df.iterrows():
        level = row["Level"]
        value = row["Covariate Value"]
        b = row["Group Difference (B)"]
        se = row["Std. Error"]
        t = row["t-stat"]
        p = row["p-value"]

        if p < alpha:
            significant_levels.append(level)
            sentence = (
                f"At {level} levels of {moderator} "
                f"(value = {value}), {iv} significantly predicted {dv}, "
                f"B = {b:.3f}, SE = {se:.3f}, t = {t:.2f}, p = {p:.3f}."
            )
        else:
            sentence = (
                f"At {level} levels of {moderator} "
                f"(value = {value}), the effect of {iv} on {dv} was not significant, "
                f"B = {b:.3f}, SE = {se:.3f}, t = {t:.2f}, p = {p:.3f}."
            )

        sentences.append(sentence)

    # Summary sentence
    if len(significant_levels) == 0:
        summary = (
            f"Overall, spotlight analysis indicated that the effect of {iv} on {dv} "
            f"was not significant at any level of {moderator}."
        )
    elif len(significant_levels) == len(results_df):
        summary = (
            f"Overall, spotlight analysis indicated that the effect of {iv} on {dv} "
            f"was significant at all examined levels of {moderator}."
        )
    else:
        summary = (
            f"The effect of {iv} on {dv} was significant only at "
            f"{', '.join(significant_levels)} levels of {moderator}."
        )

    return summary + "\n\n" + " ".join(sentences)



def ancova_test_pipeline(df, dv, iv, covariate, levene_test, linearity_p_value, alpha=0.05):
    df_clean= df.copy()

    #Check that the ANCOVA assumptions are met while also conducting sanity ckecks to ensure that all is ready for the test
    df_clean, conduct_moderated_regression,levene_ancova_p = ancova_assumptions_pipeline(df, dv, iv, cov)

    #In case the assumption of homogeneity of slopes isn't violated, conduct ancova
    if conduct_moderated_regression == False:

        ancova_model, ancova_results= run_ancova(df_clean, dv, iv, covariate, levene_ancova_p, linearity_p_value, alpha=0.05)
        
        print(ancova_results)

        # Go over each effect and report stat, p-value and effect size
        for term in ancova_results.index:
            F_val = ancova_results.loc[term, "F"]
            p_val = ancova_results.loc[term, "PR(>F)"]
            eta_sq = ancova_results.loc[term, "partial_eta_sq"]

            if p_val < alpha:
                logger.debug(f"Term: {term} is significant (F = {F_val:.3f}, p = {p_val:.4f}, partial η² = {eta_sq:.3f})")
            else:
                logger.debug(f"Term: {term} is not significant (F = {F_val:.3f}, p = {p_val:.4f}); no effect detected.")
        
        iv_pvalue = ancova_results.loc[iv, "PR(>F)"]    
        cov_pvalue = ancova_results.loc[covariate, "PR(>F)"]

        if iv_pvalue < alpha and cov_pvalue >= alpha:
            logger.debug("The effect of the IV on the dependent variable is statistically significant after controlling for the covariate.")
            logger.debug("The covariate did not have a significant effect, so it didn’t explain much additional variance.")
        elif iv_pvalue < alpha and cov_pvalue < alpha:
            logger.debug("After controlling for the covariate, the independent variable significantly affected the dependent variable.") 
            logger.debug("Additionally, the covariate itself was a significant predictor, indicating it also explains some of the variation in the outcome.")
        elif iv_pvalue >= alpha and cov_pvalue < alpha:
            logger.debug("After accounting for the covariate, the independent variable did not significantly influence the dependent variable.")
            logger.debug("The covariate itself, however, was a significant predictor, explaining some of the variation in the outcome.")
        else:
            logger.debug("After accounting for the covariate, the independent variable did not significantly influence the dependent variable.") 
            logger.debug("The covariate itself was also not a significant predictor, indicating that neither factor explained a meaningful portion of the variation in the outcome.")

        if iv_pvalue < alpha:
            pairwise_ttest_table = run_ancova_posthoc(df_clean, ancova_model, dv, iv, covariate, levene_test=levene_test, alpha=0.05)

            results_df = pairwise_ttest_table.summary_frame()

            # Loop over rows
            for comp, row in results_df.iterrows():
                if row['pvalue'] < alpha:  # significance threshold
                    logger.debug(f"Significant comparison: {comp}")
                    logger.debug(f"  Mean difference: {row['mean']:.3f}")
                    logger.debug(f"  t = {row['t']:.3f}, p = {row['pvalue']:.3f}")
                    logger.debug(f"  95% CI: [{row['lower']:.3f}, {row['upper']:.3f}]")

    return conduct_moderated_regression
        
            
            

def moderated_regression_pipeline(df, dv, iv, moderator, conduct_moderated_regression, alpha=0.05):

    if conduct_moderated_regression == True:
        logger.debug("The assumption of homogeneity of slopes has been violated. Instead running moderated regression")
        df_clean = df.copy()
        #Check assumptions including sanity checks
        df_clean = moderated_regression_assumptions_pipeline(df_clean, dv, iv, moderator)

        #Conduct regression test
        moderated_regression_results = run_moderated_regression(df_clean, dv, iv, moderator)

        for term, row in moderated_regression_table.iterrows():
            if row["P>|t|"] < alpha:
                logger.debug(
                    f"{term} was significant: "
                    f"β = {row['Coef.']:.3f}, "
                    f"t = {row['t']:.2f}, "
                    f"p = {row['P>|t|']:.3f}"
                )
        
        # Reporting the results of the interaction.
        # Conducting spotlight analysis in case of significance
        p_int = moderated_regression_table.loc[f"{iv}:{moderator}", "P>|t|"]

        if p_int > alpha:
            logger.debug(
                f"The interaction between {iv} and {moderator} was not significant "
                f"(p = {p_int:.3f})."
            )
        else:
            logger.debug(
                f"There was a significant interaction between {iv} and {moderator}, "
                f"β = {beta_int:.3f}, t = {t_int:.2f}, p = {p_int:.3f}."
            )

            spotlight_analysis_results = run_spotlight_analysis(df, dv, iv, moderator)
            report_spot = report_spotlight_results(spotlight_analysis_results, iv, dv, moderator, alpha=0.05)

            logger.debug(f"{report_spot}")

def report_simple_effects_results():
    alpha = 0.05
    s_e = results[factor][level]['posthoc']

    sig = gh[gh['pval'] < alpha]

    for _, row in sig.iterrows():
        group1 = row['A']
        group2 = row['B']
        diff = row['diff']
        t = row['T']
        p = row['pval']

        logger.debug(
            f"Significant difference between {group1} and {group2}: "
            f"ΔM = {diff:.3f}, t = {t:.2f}, p = {p:.3f}"
        )

            
def anova_pipeline(df, dv, iv, factor2, levene_p, alpha=0.05):
    df_clean = df.copy()

    #Check whether the assumptions (including the structural assumptions) are met
    df_clean, anova_levene = anova_assumptions_pipeline(df_clean,dv, iv, factor2)
    anova_table, robust_type = two_way_anova_test(df_clean, dv, iv, factor2, levene_p, check_interaction = True, alpha=0.05)

    # Report results

    report_two_way_anova(anova_table, iv, factor2, alpha=0.05, check_interaction=True)
    row = anova_table.loc[f"C({iv}):C({factor2})"]

    F = row["F"]
    p = row["PR(>F)"]
    eta = row.get("partial_eta_sq", None)

    if p < alpha:
        debug.logger(
            f"There was a significant interaction between {iv} and {factor2}, "
            f"F = {F:.2f}, p = {p:.3f}"
            + (f", partial η² = {eta:.3f}." if eta is not None else ".")
        )

        # Conduct simple effects
        simple_effects_results = simple_effects_anova(df_clean, dv, iv, factor2, alpha=0.05)

        # Reporting the results of the simple effects test
        report_simple_effects(simple_effects_results, alpha=0.05)

        #In case there aren;t significant simple effects, conduct posthoc
        if not simple_effects_results.empty: 
            post_hoc_table = run_posthoc_on_significant_simple_effects(df_clean, dv, iv, factor2, simple_effects_results, anova_levene, alpha = 0.05)
            print(post_hoc_table)
            #Report the results of the post hoc
            report_significant_posthoc_simple_effects(post_hoc_table , alpha=0.05)


    else:
        debug.logger(
            f"The interaction between {iv} and {factor2} was not significant, "
            f"F = {F:.2f}, p = {p:.3f}"
            + (f", partial η² = {eta:.3f}." if eta is not None else ".")
        )


        additive_anova_table, robust_type = two_way_anova_test(df_clean, dv, iv, factor2, anova_levene, check_interaction=False, alpha=0.05)

        # Report results of additive anova
        report_two_way_anova(additive_anova_table, iv, factor2, alpha=0.05, robust_type= robust_type)
        for factor in [iv, factor2]:
            row = additive_anova_table.loc[f"C({factor})"]
            if row["PR(>F)"] < alpha:
                #Conduct post-hoc
                posthoc_table = additive_anova_posthoc(df_clean,dv,factor,row["PR(>F)"],robust_type, alpha = 0.05)

                #Report results
                report_posthoc_additive_anova(posthoc_table, alpha=0.05)








    







        


