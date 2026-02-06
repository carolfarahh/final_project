import numpy as np
from scipy.stats import pearsonr, levene
from statsmodels.formula.api import ols
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


""" 
    Definitions:
    dv = dependent variable -> Brain volume loss
    iv = independent variable -> Disease Stage 
    cov = covariate -> Age
    factor2 = second factor -> Sex
"""



# Removes duplicated subject IDs as a measure to check the independence of variables
def drop_duplicate_subjects(df, id_col, keep="first"):
    if id_col not in df.columns: # Raises error in case column isn't found
        raise ValueError(f"Column '{id_col}' not found. If you don’t have IDs, independence is judged by study design.")

    return df.drop_duplicates(subset=[id_col], keep=keep) # Keep="first" keeps one row per subject in case of duplication

# Checks linearity between a covariate and dependent variable.
# 'kind' can be "scatter" or "hexbin" for high-density data.
def check_linearity_predictor_dv(df, dv, predictor):

    # Creates numeric arrays of variables
    x = df[predictor].astype(float).values
    y = df[dv].astype(float).values

    # Conducts Pearson correlation and saves the r and p values
    r, p = pearsonr(x, y)

    return x, y, float(r), float(p)

# Conducts transformation on numeric column in case none-linear relation between the covariate and the dependent variable
# Offset is a constant thats added to data before taking the logarithm to handle zero 
# or negative values, ensuring all inputs are positive
def log_transform(df,column,new_column=None,offset="auto"):

    df_out = df.copy()

    x = df_out[column].astype(float) #Creates an array of the column that was selected


    # In case offset was not specified by user, the code calculates the minimal offset value required to ensure
    # all the values are positive
    if offset == "auto": 
        min_val = x.min()
        offset_used = abs(min_val) + 1 if min_val <= 0 else 0 #Takes minimal value's absolute value and adds one to it
    else:
        offset_used = float(offset) #specified offset value by user

    transformed = np.log(x + offset_used) #Creates log transformation of values in column and with the added offset

    #New column title in case it hasn't been specified by user
    if new_column is None: 
        new_column = f"log_{column}"

    df_out[new_column] = transformed

    return df_out, offset_used


#Extremely IMPORTANT assumption for ANCOVA test!!!!
# Function checks whether  the homogeneity of slopes asusmption is violated or note
# In turn measure whether there's an interaction between the independent and the covariate
def check_homogeneity_of_slopes(df,dv,iv,covariate):

    #Validation test 
    assert_required_columns(df, [dv,iv,covariate])

    
    #Fits model of interaction
    model = ols(f"{DV} ~ C({IV}) * {Covariate}",data=df).fit() 

    # Given that we're checking the interaction, we create an ANOVA table
    table = sm.stats.anova_lm(model, typ=3)  

    interaction_key = f"C({iv}):{covariate}"
    
    p_val = table.loc[interaction_key, "PR(>F)"]
    return p_val, table


# Levene ANCOVA
def levene_ancova(df, dv, iv, covariate, center='median'):
    """
    Levene's test for ANCOVA using model residuals.
    In order to check for homogeinity of groups, unlike the ANOVA test,
    we conduct it on the residuals.
    If p-value is smaller than 0.05, the assumption would be violated

    """

    # Validation test
    df_clean = validate_ancova_for_levene(df, dv, iv, covariate)

    # Fit ANCOVA model
    model = smf.ols(f"{dv} ~ C({iv}) + {covariate}",data=df_clean).fit()

    df_clean = df_clean.copy()

    # Creates a column with residuals of each row
    df_clean["_residuals"] = model.resid 

    #Raises error in case of NaN values
    if df_clean["_residuals"].isna().any(): 
        raise ValueError(
            "Residuals contain NaN values. "
            "Check model specification or input data."
        )

    # Levene on residuals
    # Groups the values of the residuals based on the levels of the iv, and creates an array
    # for each level
    groups = [
        df_clean.loc[df_clean[iv] == level, "_residuals"].values
        for level in df_clean[iv].unique()
    ]
    # Run the levene test using skipy library
    # The asterisk takes each item in the list and unpacks them seperately 
    stat, p = levene(*groups, center=center)
    return stat, p

# Levene_two_way_anova
def levene_two_way_anova(df, dv, iv, factor2, center='median'):

    """
    Levene test for two-way ANOVA to check homogeneity of variance
    Uses median instead of mean because extreme values of each level were 
    less likely to be affected by cooks distance function.
    In case the p-value was significant, we can understand that the variances
    in the groups aren't equal
    """ 
    # We ensure that the data is in fact suitable for our levene test using 
    # Our previous function
    df_clean = validate_two_way_anova_for_levene(
        df, dv, iv, factor2)

    # Converts the values of dv into a data frame, ignores the title, and groups each group by the factors,
    # Turns each group into an array and adds it to the list of groups
    groups = [
        sub_df[dv].values
        for _, sub_df in df_clean.groupby([iv, factor2])
    ]

    # Run the levene test using skipy library
    # The asterisk takes each item in the list and unpacks them seperately 
    stat, p = levene(*groups, center=center)

    return stat, p 

# Normality of residuals
def check_normality_of_residuals_visual(df,dv,iv,covariate):
    """
    To check for normality of residuals, we have to create a graph for all of the residuals
    and physically check if the residuals are distributed linearly.
    In this function, we create two graphs; Q-Q plot and Histogram, using matplotlib.pyplot library.

    """

    assert_required_columns(df, [dv,iv,covariate])
    model = ols(f"{dv} ~ C({iv}) * {covariate}", data=df).fit() #We fit the model of ANCOVA

    # We create an array of the residuals using the .resid function
    # Just in case some of the residuals had NaN values we drop them to ensure smoother output of graphs
    resid = model.resid.dropna() 

    # Histogram
    plt.figure()
    plt.hist(resid, bins=30) #It controls the granularity of the plot
    plt.title("Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Q-Q Plot
    plt.figure()
    sm.qqplot(resid, line="45") #Adds a 45-degree reference line for comparrison 
    plt.title("Q-Q Plot of Residuals")
    plt.show()



import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def check_residual_normality_visual(df,dv,iv,predictor,interaction=False,center=True):
    """
    Visual check of residual normality for linear models.
    Can be used for ANCOVA (no interaction) or moderated regression (with interaction).
    """
    data = df.copy()

    # Mean-center continuous predictor (recommended)
    if center:
        data[predictor + "_c"] = data[predictor] - data[predictor].mean()
        pred = predictor + "_c"
    else:
        pred = predictor

    # Build formula
    if interaction: #moderated regression
        formula = f"{dv} ~ C({iv}) * {pred}"
    else: #ANCOVA
        formula = f"{dv} ~ C({iv}) + {pred}"

    #Fit the model
    model = ols(formula, data=data).fit()
    
    #Calculate the residuals
    resid = model.resid.dropna()

    # Histogram
    plt.figure()
    plt.hist(resid, bins=30)
    plt.title("Residuals Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Q-Q Plot
    plt.figure()
    sm.qqplot(resid, line="45")
    plt.title("Q-Q Plot of Residuals")
    plt.show()



# Multicollinearity check
def check_vif(df, iv, covariate):
    #validation test
    assert_required_columns(df, [iv, covariate])

    # Add predictors into a lost
    predictors = [iv, covariate]

    # Build design matrix
    X_raw = df[predictors]
    # Using the get_dummies function, it converts the categorical variable into a dummy variable with numeric 
    # values of 1 and 0
    X_dummies = pd.get_dummies(X_raw, drop_first=True) 
    
    # Adds the intercept, representing the reference group's mean when the covariate is zero
    X = sm.add_constant(X_dummies)

    # Save the columns names into a list and also creates an array of all the values of X
    vifs_list = []
    cols = X.columns.tolist()
    X_vals = X.values.astype(float)

    # This loop calculates the Variance Inflation Factor (VIF) for every column in our model
    # using variance_inflation_factor and adds the values to a dictionary 
    for i, col in enumerate(cols):
        vif_val = variance_inflation_factor(X_vals, i)
        vifs_list.append({"feature": col, "vif": float(vif_val)})

    # Convert to DataFrame for easier handling
    vif_df = pd.DataFrame(vifs_list)

    # We locate high VIFs (excluding 'const') and add it its own column
    high_vif_mask = (vif_df['vif'] > 5) & (vif_df['feature'] != 'const')
    high_vif_df = vif_df[high_vif_mask]

    if not high_vif_df.empty:
        # We extract the names of the problematic features 
        problematic_features = high_vif_df['feature'].tolist()
        print(f"High Multicollinearity detected in: {problematic_features}")

        return True
    else:
        print("Multicollinearity check passed: All VIFs are within acceptable limits.")

        return False
    

def center_moderator(df, moderator, center=None):
    data = df.copy()
    if center == True:
        data[moderator + "_c"] = data[moderator] - data[moderator].mean()
        mod = moderator + "_c"

    else:
        mod = moderator

    return mod
    
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

def homoscedasticity_test_moderated_regression(df, dv, iv, moderator):
    """
    Fits a moderated regression model and tests homoscedasticity
    using the Breusch–Pagan test.
    Parameters
    ----------
    df : Dataframe containing variables
    dv : str - Dependent variable name
    iv : str - Independent variable name (can be categorical or continuous)
    moderator : str - Moderator variable name (continuous)
    center : bool - Whether to mean-center continuous predictors (recommended)

    Returns
    -------
    lm p- value : Since our sample is large, it's better to focus on the the lm p-value instead
                 of the f p-value
    """

    data = df.copy()

    # Build formula (IV can be categorical or continuous)
    formula = f"{dv} ~ C({iv}) * {mod}" 

    # Fit model
    model = smf.ols(formula, data=data).fit()

    # Breusch–Pagan test
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
        model.resid,
        model.model.exog
    )



    return lm_pvalue 



def ancova_assumptions_pipeline(df, dv, iv, cov):
    df_clean = df.copy()
    conduct_moderated_regression = False
    df_clean = ancova_validation_pipeline(df, dv, iv, cov)
    id_col = "Subject_ID"
    df_clean= drop_duplicate_subjects(df_clean, id_col, keep="first")
    linearity_x, linearity_y, linearity_r, linearity_p = check_linearity_predictor_dv(df_clean, dv, cov)
    linearity_graph_cov_dv(linearity_x,linearity_y,linearity_r, linearity_p, dv, cov, show_plot=False, kind="hexbin")
    if linearity_p >= 0.05:
        df_clean, log_offset = log_transform(df_clean,dv ,new_column=None,offset="auto")
        logger.debug(f"Conducted log transformation on {dv} with an offset of {log_offset}")
    homogeneity_of_slopes_p_val, homogeneity_of_slopes_table = check_homogeneity_of_slopes(df_clean,dv,iv,cov)

    if homogeneity_of_slopes_p_val >= 0.05:
        logger.debug(f"There's a significant interaction between {iv} and {cov}. Conducting moderated regression instead")
        conduct_moderated_regression = True
        return conduct_moderated_regression
    levene_ancova_stat, levene_ancova_p = levene_ancova(df_clean, dv, iv, cov, center='median')
    check_normality_of_residuals_visual(df_clean,dv,iv,cov)

    return conduct_moderated_regression,levene_ancova_p

def moderated_regression_assumptions_pipeline(df, dv, iv, moderator):
    df_clean = df.copy()
    df_clean = moderated_regression_validation_pipeline(df, dv, iv, moderator)
    id_col = "Subject_ID"
    df_clean= drop_duplicate_subjects(df_clean, id_col, keep="first")
    linearity_x, linearity_y, linearity_r, linearity_p = check_linearity_predictor_dv(df_clean, dv, cov)
    linearity_graph_cov_dv(linearity_x,linearity_y,linearity_r, linearity_p, dv, cov, show_plot=False, kind="hexbin")
    if linearity_p >= 0.05:
        df_clean, log_offset = log_transform(df_clean,dv ,new_column=None,offset="auto")
        logger.debug(f"Conducted log transformation on {dv} with an offset of {log_offset}")
    breusch_pagan_p = homoscedasticity_test_moderated_regression(df, dv, iv, moderator)
    multicolinearity_check = check_vif(df, iv, moderator)
    df_clean[moderator] = center_moderator(df, moderator, center=multicolinearity_check)

    return df_clean



    

