 



def two_way_anova_test(df, dv, iv, factor2, levene_p, check_interaction, alpha=0.05):
    # hc3 in case assumption of equal variance is violated
    robust_type = "hc3" if levene_p < alpha else None
    # We start with interactive model and if it's insignificant, we conduct additive model
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

    #We fit the model
    model = ols(formula, data=df).fit() 

    anova_table = anova_lm(model, typ=anova_typ, robust=robust_type)

    return anova_table, robust_type

# Two Way ANOVA post hoc        
def simple_effects_anova(df, dv, factor1, factor2, alpha=0.05):
    results = []

    for level in df[factor2].unique():
        sub = df[df[factor2] == level]
        
        # Fit one-way ANOVA for factor1 at this level of factor2
        model = smf.ols(f"{dv} ~ C({factor1})", data=sub).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        #reporting values
        f_val = anova_table.loc[f"C({factor1})", "F"]
        p_val = anova_table.loc[f"C({factor1})", "PR(>F)"]
        
        
        results.append({
            "Level": level,
            "F": round(f_val, 3),
            "p": round(p_val, 4),

        })
    results_df = pd.DataFrame(results)
    logger.debug("Conducting simple effects test")
    return results_df

def run_posthoc_on_significant_simple_effects(
    df, dv, iv, factor2, simple_effects_results, levene_p, alpha = 0.05):
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
            posthoc = pairwise_tukeyhsd(endog=sub_df[dv], groups=sub_df[factor2])
            posthoc_df = pd.DataFrame(data=posthoc.summary().data[1:], columns=posthoc.summary().data[0])
            logger.debug("Conducting tukey for significant simple effect")

        else:
            posthoc_df = pg.pairwise_gameshowell(data=sub_df, dv=dv, between=factor2)
            logger.debug("Conducting games howell for significant simple effect")


        posthoc_dict[level] = posthoc_df
    logger.debug(posthoc_dict)


    return posthoc_dict

def additive_anova_posthoc(df,dv,factor,main_effect_p,robust_type, alpha = 0.05):

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
        logger.debug(f"Running Games–Howell for '{factor}' adjusted for unequal variance.")
        return pg.pairwise_gameshowell(data=df,dv=dv,between=factor)

################## ANCOVA ######################
#Adjusted quadratic model, his is done in case the the assumption of linearity in ancova is violatated
def quadratic_model_adjustment(df, dv, iv,covariate):
    df = df.copy()
    # Center the covariate  and square the covariate
    cov_c = f"{covariate}_c"
    cov_c_sq = f"{covariate}_c_sq"
    df[cov_c] = df[covariate] - df[covariate].mean()
    df[cov_c_sq] = df[cov_c] ** 2
    # fit model
    model = ols(f"{dv} ~ C({iv}) + {cov_c} + {cov_c_sq}", data=df).fit()
    logger.debug("Adjusting model with quadratic model")
    return model

def run_ancova(data, dv, iv, covariate, levene_test, linearity_p_value, alpha=0.05):
    # If linearity is violated use quadratic model
    if linearity_p_value > alpha:
        model = quadratic_model_adjustment(data, dv, iv, covariate)
    else:
        model = smf.ols(f"{dv} ~ C({iv}) + {covariate}", data=data).fit()
    # Variance adjustment (apply to BOTH models)
    if levene_test < alpha:
        variance_assumption = "using HC3 robust SEs"
        model = model.get_robustcov_results(cov_type="HC3")
    else:
        variance_assumption = "with equal variances assumed"
    logger.debug(f"Conducting ANCOVA {variance_assumption}")
    # ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)
    # Partial eta squared because it's better to report the effect sizee in case of significance
    ancova_table["partial_eta_sq"] = (
        ancova_table["sum_sq"] /
        (ancova_table["sum_sq"] + ancova_table.loc["Residual", "sum_sq"])
    )
    logger.debug(ancova_table)
    return model, ancova_table


def run_ancova_posthoc(df, model, iv, levene_test=None, alpha=0.05):
    df_clean = df.copy()
    # Fit the model (It will automatically upgrade as quadratic in case of )
    if levene_test is not None and levene_test < alpha:
        model = model.get_robustcov_results(cov_type="HC3")
    # Perform Pairwise T-Tests (Post-hoc)
    posthoc = model.t_test_pairwise(term_name=f"C({iv})", method="bonferroni")
    logger.debug ("Running Pairwise T-Tests for ancova's significant iv")
    logger.debug(posthoc)
    return posthoc

#No need for levene test because robust regression is always preffered
def run_moderated_regression(df, dv, iv, moderator, alpha): 
    model = smf.ols(formula=f"{dv} ~ {iv} * {moderator}", data=df).fit(cov_type="HC3")
    #Create a table for the results
    moderated_regression_table = model.summary2().tables[1]
    logger.debug  ("Running moderated regression test")
    logger.debug(moderated_regression_table)
    return moderated_regression_table

def run_spotlight_analysis(df, dv, iv, moderator):
    # Calculate the 'spots' (Mean, +1SD, -1SD)
    mean_cov = df[moderator].mean()
    sd_cov = df[moderator].std()
    spots = {'Low (-1 SD)': mean_cov - sd_cov, 'Average (Mean)': mean_cov, 'High (+1 SD)': mean_cov + sd_cov}
    results = []
    # Run the model for each spot by centering the covariate
    for label, value in spots.items():
        # Center the covariate at the specific spot
        df['temp_centered'] = df[moderator] - value
        # Fit the interaction model using robust SEs
        model = smf.ols(f"{dv} ~ {iv} * temp_centered", data=df).fit(cov_type='HC3')
        # Extract the coefficient and stats for the IV
        iv_row = [row for row in model.params.index if iv in row and ':' not in row and 'Intercept' not in row][0]
        #Create dictionary for results
        results.append({
            'Level': label,
            'Covariate Value': round(value, 3),
            'Group Difference (B)': round(model.params[iv_row], 4),
            'Std. Error': round(model.bse[iv_row], 4),
            't-stat': round(model.tvalues[iv_row], 4),
            'p-value': round(model.pvalues[iv_row], 4)
        })
    # convert dictionary to dataset
    results_df = pd.DataFrame(results)
    logger.debug("Conducting spotlight analysis")
    return results_df

def ancova_test_pipeline(df, dv, iv, covariate, alpha=0.05):
    df_clean= df.copy()
    #Check that the ANCOVA assumptions are met while also conducting sanity ckecks to ensure that all is ready for the test
    df_clean, conduct_moderated_regression,levene_ancova_p, ancova_linearity_p = ancova_assumptions_pipeline(df, dv, iv, covariate)
    #In case the assumption of homogeneity of slopes isn't violated, conduct ancova
    if conduct_moderated_regression == False:
        ancova_model, ancova_results= run_ancova(df_clean, dv, iv, covariate, levene_ancova_p, ancova_linearity_p, alpha=0.05)
        # Go over each effect and report stat, p-value and effect size
        for term in ancova_results.index:
            F_val = ancova_results.loc[term, "F"]
            p_val = ancova_results.loc[term, "PR(>F)"]
            eta_sq = ancova_results.loc[term, "partial_eta_sq"]
            if p_val < alpha:
                logger.debug(f"Term: {term} is significant (F = {F_val:.3f}, p = {p_val:.4f}, partial η² = {eta_sq:.3f})")
        iv_row_name = [row for row in ancova_results.index if iv in row][0]
        iv_pvalue = ancova_results.loc[iv_row_name, "PR(>F)"]  
        cov_row_name = [row for row in ancova_results.index if covariate in row][0]  
        cov_pvalue = ancova_results.loc[cov_row_name, "PR(>F)"]
        #Conduct post-hoc on relevant significant effects (Only if the iv is significant)
        if iv_pvalue < alpha:
            pairwise_ttest_table = run_ancova_posthoc(df_clean, ancova_model, dv, iv, covariate, levene_test=levene_test, alpha=0.05)
            results_df = pairwise_ttest_table.summary_frame()
            # Loop over rows
            for comp, row in results_df.iterrows():
                if row['pvalue'] < alpha:  # significance threshold
                    logger.debug(f"Significant comparison: {comp} t = {row['t']:.3f}, p = {row['pvalue']:.3f}")
        #Plot
        adjusted_means_plot(df_clean,dv, iv, covariate)

    return conduct_moderated_regression

def moderated_regression_pipeline(df, dv, iv, moderator, conduct_moderated_regression, alpha=0.05):
    if conduct_moderated_regression == True:
        logger.debug("The assumption of homogeneity of slopes has been violated. Instead running moderated regression")
        df_clean = df.copy()
        #Check assumptions including sanity checks
        df_clean = moderated_regression_assumptions_pipeline(df_clean, dv, iv, moderator)
        #Conduct regression test
        moderated_regression_results = run_moderated_regression(df_clean, dv, iv, moderator, alpha=0.05)

        if "P>|t|" in moderated_regression_results.columns:
            p_col = "P>|t|"
        elif "P>|z|" in moderated_regression_results.columns:
            p_col = "P>|z|"
        elif "pvalue" in moderated_regression_results.columns:
            p_col = "pvalue"
        else:
            raise ValueError("Could not find p-value column")
        for term, row in moderated_regression_results.iterrows():
            if row[p_col] < alpha:
                logger.debug(
                    f"{term} was significant: β = {row['Coef.']:.3f} p = {row[p_col]:.3f}")
        # Conducting spotlight analysis in case of significance of interaction only
        interaction_rows = [i for i in moderated_regression_results.index if ":" in i and moderator in i]
        p_int = moderated_regression_results.loc[interaction_rows, p_col]

        for term, p_val in p_int.items():
            if p_val < alpha:
                logger.debug(f"{term} significant: β = {moderated_regression_results.loc[term, 'coef']:.3f}, p = {p_val:.3f}")
            else:
                logger.debug(f"{term} non-significant: p = {p_val:.3f}")

            spotlight_analysis_results = run_spotlight_analysis(df, dv, iv, moderator)
        adjusted_means_plot_moderated(df_clean, dv, iv, moderator)

def anova_pipeline(df, dv, iv, factor2, levene_p= None, alpha=0.05):
    df_clean = df.copy()
    df_clean=remove_influential_by_cooks(df, dv, iv, "ANOVA", covariate=None, factor2=factor2, check_interaction=None)
    #Check whether the assumptions (including the structural assumptions) are met
    df_clean, anova_levene = anova_assumptions_pipeline(df_clean,dv, iv, factor2)
    anova_table, robust_type = two_way_anova_test(df_clean, dv, iv, factor2, levene_p = anova_levene, check_interaction = True, alpha=0.05)
    row = anova_table.loc[f"C({iv}):C({factor2})"]
    F = row["F"]
    p = row["PR(>F)"]
    eta = row.get("partial_eta_sq", None)
    if p < alpha:
        logger.debug(f"There was a significant interaction between {iv} and {factor2}, p = {p:.3f}")
        # Conduct simple effects
        simple_effects_results = simple_effects_anova(df_clean, dv, iv, factor2, alpha=0.05)
        #In case there aren;t significant simple effects, conduct posthoc
        if not simple_effects_results.empty: 
            post_hoc_table = run_posthoc_on_significant_simple_effects(df_clean, dv, iv, factor2, simple_effects_results, anova_levene, alpha = 0.05)
            logger.debug(post_hoc_table)
    else:
        logger.debug(f"The interaction between {iv} and {factor2} was not significant, p = {p:.3f}")
        df_clean = df.copy()
        df_clean= df_clean=remove_influential_by_cooks(df, dv, iv, "ANOVA", covariate=None, factor2=factor2, check_interaction=True)
        additive_anova_table, robust_type = two_way_anova_test(df_clean, dv, iv, factor2, anova_levene, check_interaction=False, alpha=0.05)
        for term in [iv, factor2]:
            row = anova_table.loc[f"C({term})"]
            p = row["PR(>F)"]
            if p < alpha:
                additive_ph= additive_anova_posthoc(df_clean,dv,term,p,robust_type, alpha = 0.05)
                logger.debug(f"Conducting additive anova post hoc for {term}")
                logger.debug(additive_ph)

    boxplot_two_factor(df, dv, iv, factor2,title=None, ylabel=None, xlabel=None,figsize=(8,6), palette="Set2")