import pingouin as pg
import numpy as np
from scipy.stats import f_oneway


def data_describe(df):
    data_stats = df.describe()
    return data_stats

def levene_test(df, dependent_variable, independent_variable):
    results = pg.homoscedasticity(
        data=df,
        dv=dependent_variable,
        group=independent_variable,
        method="levene"
    )
    return results

def tukey(df, dependent_variable, independent_variable):
    results = pg.pairwise_tukey(data=df, dv=dependent_variable, between=independent_variable)
    return results

def anova(dv, analysis_df, iv):
    groups = [
        g[dv].dropna()
        for _, g in analysis_df.groupby(iv)
    ]

    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def welch_anova(df, dependent_variable, independent_variable):
    results = pg.welch_anova(data=df, dv=dependent_variable, between=independent_variable)
    f_stat = results['F'].item()
    p_value = results['p-unc'].item()

    return f_stat, p_value


def gameshowell(df, dependent_variable, independent_variable):
    results = pg.pairwise_gameshowell(data=df, dv= dependent_variable, between= independent_variable)
    return results






