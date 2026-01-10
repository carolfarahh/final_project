import pingouin as pg


def welch_anova(df, dependent_variable, independent_variable):
    results = pg.welch_anova(data=df, dv=dependent_variable, between=independent_variable)
    return results


def tukey(df, dependent_variable, independent_variable):
    results = pg.pairwise_tukey(data=df, dv=dependent_variable, between=independent_variable)
    return results


def levene_test(df, dependent_variable, independent_variable):
    results = pg.homoscedasticity(
        data=df,
        dv=dependent_variable,
        group=independent_variable,
        method="levene"
    )
    return results

