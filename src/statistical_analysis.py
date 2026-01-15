import pingouin as pg
import numpy as np

def data_describe(df):
    data_stats = df.describe()
    return data_stats


def gameshowell(df, dependent_variable, independent_variable):
    results = pg.pairwise_gameshowell(data=df, dv= dependent_variable, between= independent_variable)
    return results

#TODO: things in common between both tests: Data describe

import pandas as pd
import numpy as np
from scipy import stats

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

#TODO: TWO_WAY_ANOVA
#TODO: TWO_WAY_ANOVA Welch test
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def welch_two_way_anova(df, dv, factor1, factor2, levene_test):
    # Convert factors to categorical if not already
    df[factor1] = df[factor1].astype('category')
    df[factor2] = df[factor2].astype('category')
    
    # Fit OLS model(Ordinary Least Squares)
    model = ols(f'{dv} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})', data=df).fit()
    
    if levene_test == "significant":

    # ANOVA table using Type 2 sum of squares with robust covariance (Welch-style)
        anova_table = anova_lm(model, typ=2, robust='hc3')  # HC3 = heteroscedasticity-consistent
        return anova_table
    elif levene_test == "insignificant":
        anova_table = anova_lm(model, typ=2)  #Without HC3
        return anova_table    



import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def simple_effects_tukey(df, dv, factor1, factor2, alpha=0.05, levine_test):
    # Interaction is significant → simple effects + Tukey
    results = {}
    
    for level in df[factor2].cat.categories:
        sub_df = df[df[factor2] == level]
        
        # Simple effect ANOVA for factor1 at this level
        model_sub = ols(f'{dv} ~ C({factor1})', data=sub_df).fit()

        if levine_test == "positive":
            anova_sub = anova_lm(model_sub, typ=2, robust="hc3")
        
        else:
            anova_sub = anova_lm(model_sub, typ=2)

        # Tukey post-hoc for all pairwise comparisons of factor1
        tukey = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[factor1], alpha=alpha)
            
        results[level] = {'anova': anova_sub, 'tukey': tukey}
    
    return results

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def additive_anova(df, dv, factor1, factor2, levine_test):

    model = ols(
        f'{dv} ~ C({factor1}) + C({factor2})',
        data=df
    ).fit()
    if levine_test == "significant":
        return anova_lm(model, typ=2, robust="hc3")
    else:
        return anova_lm(model, typ=2)

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def additive_posthoc_tukey(df, dv, factor, alpha=0.05):
    
    tukey = pairwise_tukeyhsd(
        endog=df[dv],
        groups=df[factor],
        alpha=alpha
    )
    
    return tukey
    



