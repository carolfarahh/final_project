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
    df[factor1] = df[factor1].astype('category')
    df[factor2] = df[factor2].astype('category')
    return df[factor1], df[factor2]

def summarize(x):
    n = x.count()
    mean = x.mean()
    sd = x.std(ddof=1)
    se = sd / np.sqrt(n) if n > 0 else np.nan
    t_crit = stats.t.ppf((1 + ci)/2, df=n-1) if n > 1 else np.nan
    ci_low = mean - t_crit * se if n > 1 else np.nan
    ci_high = mean + t_crit * se if n > 1 else np.nan

    return pd.Series({
        "N": n, "Mean_DV": mean, "SD_DV": sd, "SE_DV": se, f"CI_{int(ci*100)}_Lower": ci_low, f"CI_{int(ci*100)}_Upper": ci_high})
    

def descriptive_table_two_way(df,dv,factor_a,factor_b,ci=0.95):
    table = (
        df
        .groupby([factor_a, factor_b])[dv]
        .apply(summarize)
        .reset_index()
    )
    return table


def descriptive_table_ancova(df, dv, iv, covariate, ci=0.95):
    dv_stats = df.groupby(iv)[dv].apply(summarize).reset_index()
    cov_stats = df.groupby(iv)[covariate].mean().reset_index().rename(columns={covariate: f"Mean_{covariate}"})
    table = pd.merge(dv_stats, cov_stats, on=iv)
    return table

#TODO: TWO_WAY_ANOVA homogenous variance and welch
def anova_model(df, dv, factor1, factor2, levene_test, check_interaction):

    if check_interaction == True: #defines which model to use
        # Fit OLS model(Ordinary Least Squares)
        model = ols(f'{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})', data=df).fit() #two-way factorial model
    else:
        model = ols( f'{dv} ~ C({factor1}) + C({factor2})', data=df).fit() #additive model

    if levene_test == "significant":
    # ANOVA table using Type 2 sum of squares with robust covariance (Welch-style)
        anova_table = anova_lm(model, typ=2, robust='hc3')  # HC3 = heteroscedasticity-consistent
        return anova_table
    elif levene_test == "insignificant":
        anova_table = anova_lm(model, typ=2)  #Without HC3
        return anova_table    

def simple_effects_tukey(df, dv, factor1, factor2, alpha=0.05, levine_test): #in case the interaction is significant, check simple effects

    factors = [factor1, factor2]
    results = {}
    for factor in factors:
        other_factor = [f for f in factors if f != factor][0]
        if df[factor].astype("category").cat.categories.size > 1:   ########### Really understand how simple effects work
            results.setdefault(factor, {})

            for level in df[factor].cat.categories:
                sub_df = df[df[factor] == level]
                # Simple effect ANOVA for factor1 at this level
                model_sub = ols(f'{dv} ~ C({other_factor})', data=sub_df).fit()                   
                if levine_test == "negative":
                    anova_sub = anova_lm(model_sub, typ=2)                
                else:
                    anova_sub = anova_lm(model_sub, typ=2, robust="hc3")

                results[factor][level] = {'anova': anova_sub}


                if anova_sub.loc[f"C({other_factor})", "PR(>F)"] < 0.05:
                    if levine_test == "negative":
                    # Tukey post-hoc for all pairwise comparisons of factor1
                        tukey = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[other_factor], alpha=alpha)                    
                        results[factor][level]['posthoc'] = tukey
                    else:
                        gameshowell = pg.pairwise_gameshowell(data=sub_df,dv=dv,between=other_factor)
                        results[factor][level]['posthoc'] = gameshowell
        else:
            continue

    return results


def posthoc_main_effect(df,dv,factor,main_effect_p,levene_test,alpha=0.05):
    # 1. Main effect must be significant
    if main_effect_p >= alpha:
        print(
            f"No post-hoc tests: main effect of '{factor}' "
            f"is not significant (p = {main_effect_p:.3f}).")
        return None
    # 2. Two levels → no post-hoc needed
    n_levels = df[factor].nunique()
    if n_levels == 2:
        print(
            f"No post-hoc tests needed: '{factor}' has only two levels.")
        return None
    # 3. Equal variances → Tukey
    if levene_test >= alpha:
        print(
            f"Equal variances assumed (Levene p = {levene_test:.3f}). "
            f"Running Tukey HSD for '{factor}'.")
        return pairwise_tukeyhsd(endog=df[dv],groups=df[factor], alpha=alpha)
    # 4. Unequal variances → Games–Howell
    else:
        print(
            f"Unequal variances detected (Levene p = {levene_test:.3f}). "
            f"Running Games–Howell for '{factor}'.")
        return pg.pairwise_gameshowell(data=df,dv=dv,between=factor
        )
#TODO: ANCOVA
def run_ancova(data, dv, iv, covariates, levene_test):

    # Build formula
    covariate_formula = " + ".join(covariates)
    formula = f"{dv} ~ C({iv}) + {covariate_formula}"
    
    # Fit model
    model = smf.ols(formula, data=data).fit()
    
    # Type II ANCOVA table
    if levene_test == "significant":
        model = model.get_robustcov_results(cov_type="HC3")

    ancova_table = sm.stats.anova_lm(model, typ=2)

    
    # Calculate partial eta squared
    ancova_table["partial_eta_sq"] = (
        ancova_table["sum_sq"] /
        (ancova_table["sum_sq"] + ancova_table.loc["Residual", "sum_sq"])
    )
    
    return model, ancova_table


def run_moderated_regression(df, dv, iv, covariate, levene_test):
    # Create the formula: Y ~ IV + Covariate + (IV * Covariate)
    # The '*' operator in statsmodels automatically includes main effects and the interaction
    formula = f"{dv} ~ {iv} * {covariate}"
    # Fit the model

    model = smf.ols(formula=formula, data=df).fit() 

    if levene_test == "significant":
        model = smf.ols(formula="DV ~ IV * Covariate", data=df).fit(cov_type='HC3')

    # Print results
    print("--- Moderated Regression Results ---")
    print(model.summary())
    return model


def run_spotlight_analysis(df, dv, iv, covariate, levene_test):

    # 1. Calculate the 'spots' (Mean, +1SD, -1SD)
    mean_cov = df[covariate].mean()
    sd_cov = df[covariate].std()
    
    spots = {
        'Low (-1 SD)': mean_cov - sd_cov,
        'Average (Mean)': mean_cov,
        'High (+1 SD)': mean_cov + sd_cov
    }
    
    results = []

    # 2. Run the model for each spot by centering the covariate
    # Centering at a 'spot' makes the main effect of the IV represent 
    # the group difference at that specific level of the covariate.
    for label, value in spots.items():
        # Center the covariate at the specific spot
        df['temp_centered'] = df[covariate] - value
        
        # Fit the interaction model
        # Using HC3 for robust standard errors due to unequal variance
        formula = f"{dv} ~ {iv} * temp_centered"
        model = smf.ols(formula, data=df).fit()

        if levene_test == "significant":
            model = smf.ols(formula, data=df).fit(cov_type='HC3')
        
        # Extract the coefficient and stats for the IV (the group difference)
        # Note: If IV is categorical, statsmodels uses 'IV[T.GroupName]'
        # We find the row that matches your IV name
        iv_row = [row for row in model.params.index if iv in row and ':' not in row and 'Intercept' not in row][0]
        
        results.append({
            'Level': label,
            'Covariate Value': round(value, 3),
            'Group Difference (B)': round(model.params[iv_row], 4),
            'Std. Error': round(model.bse[iv_row], 4),
            't-stat': round(model.tvalues[iv_row], 4),
            'p-value': round(model.pvalues[iv_row], 4)
        })

    # 3. Display Results
    results_df = pd.DataFrame(results)
    print("--- Spotlight Analysis (Simple Slopes) ---")
    print(results_df.to_string(index=False))
    
    return results_df

def get_ancova_posthoc(df, dv, between, covariate, method='bonf'):
    """
    Calculates the Pairwise Comparisons and Adjusted Means for an ANCOVA.
    
    Returns:
    - posthoc_results: Table with p-values and effect sizes.
    - adjusted_means: The means of each group after controlling for the covariate.
    """
    
    # 1. Calculate Pairwise Comparisons (Post-Hoc)
    # This automatically uses the covariate to adjust the means before comparing
    posthoc = pg.pairwise_tests( data=df,  dv=dv, between=between, covar=covariate, padjust=method)
    # 2. Calculate Adjusted Means (Estimated Marginal Means)
    # This gives you the 'true' means to report in your results section
    adj_means = pg.ancova(data=df, dv=dv, between=between, covar=covariate)
    
    print(f"--- Post-Hoc Results ({method} correction) ---")
    display_cols = ['A', 'B', 'stats-condition', 'p-corr', 'p-adjust', 'hedges']
    print(posthoc[display_cols])
    
    return posthoc
