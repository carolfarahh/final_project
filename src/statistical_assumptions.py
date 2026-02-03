import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.api as sm



# Removes duplicated subject IDs as a measure to check the independence of variables
def drop_duplicate_subjects(df, id_col, keep="first"):
    if id_col not in df.columns: # Raises error in case column isn't found
        raise ValueError(f"Column '{id_col}' not found. If you don’t have IDs, independence is judged by study design.")

    return df.drop_duplicates(subset=[id_col], keep=keep) # Keep="first" keeps one row per subject in case of duplication

# Checks linearity between a covariate and dependent variable.
# 'kind' can be "scatter" or "hexbin" for high-density data.
def check_linearity_cov_dv(df, dv, cov, show_plot=True):

    # Creates numeric arrays of variables
    x = df[cov].astype(float).values
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
def check_homogeneity_of_slopes(df,DV,IV,Covariate):
    #Fits model of interaction
    model = ols(f"{DV} ~ C({IV}) * {Covariate}",data=df).fit() 

    # Given that we're checking the interaction, we create an ANOVA table
    table = sm.stats.anova_lm(model, typ=3)  
    return table #Return the ANOVA table

#Levene test for two-way ANOVA
#Uses median instead of mean because extreme values of each level were 
#less likely to be affected by cooks distance function
def levene_two_way_anova(df, dv, factor1, factor2, center='median'):
    groups = [
        sub_df[dv].dropna().values
        for _, sub_df in df.groupby([factor1, factor2])
        if len(sub_df) > 1
    ]

    stat, p = levene(*groups, center=center)
    return stat, p

import statsmodels.formula.api as smf
from scipy.stats import levene

# Validation / sanity-check function (reusable)

def validate_ancova_for_levene(df, dv, iv, covariate):
    """
    Validate data for Levene's test in ANCOVA.
    Raises ValueError if assumptions for the test are violated.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned dataframe (rows with NaNs dropped)
    """

    # ---- Column existence ----
    required_cols = {dv, iv, covariate}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---- Drop NaNs ----
    df_clean = df[[dv, iv, covariate]].dropna()

    if len(df_clean) < 3:
        raise ValueError("Not enough observations after dropping missing values.")

    # ---- IV checks ----
    n_levels = df_clean[iv].nunique()
    if n_levels < 2:
        raise ValueError(
            f"Levene's test requires at least 2 levels in '{iv}'. "
            f"Found {n_levels}."
        )

    group_sizes = df_clean[iv].value_counts()
    if (group_sizes < 2).any():
        bad_levels = group_sizes[group_sizes < 2].index.tolist()
        raise ValueError(
            f"Each level of '{iv}' must have at least 2 observations. "
            f"Problematic levels: {bad_levels}"
        )

    # ---- Covariate checks ----
    if df_clean[covariate].nunique() < 2:
        raise ValueError(
            f"Covariate '{covariate}' has no variability (constant). "
            "ANCOVA cannot be fitted."
        )

    return df_clean

# Clean levene_ancova using the validator
import statsmodels.formula.api as smf
from scipy.stats import levene

def levene_ancova(df, dv, iv, covariate, center='median'):
    """
    Levene's test for ANCOVA using model residuals.
    """

    # Validate input
    df_clean = validate_ancova_for_levene(df, dv, iv, covariate)

    # Fit ANCOVA model
    model = smf.ols(
        f"{dv} ~ C({iv}) + {covariate}",
        data=df_clean
    ).fit()

    df_clean = df_clean.copy()
    df_clean["_residuals"] = model.resid

    if df_clean["_residuals"].isna().any():
        raise ValueError(
            "Residuals contain NaN values. "
            "Check model specification or input data."
        )

    # Levene on residuals
    groups = [
        df_clean.loc[df_clean[iv] == level, "_residuals"].values
        for level in df_clean[iv].unique()
    ]

    stat, p = levene(*groups, center=center)
    return stat, p
# Validation function for Two-Way ANOVA Levene

def validate_two_way_anova_for_levene(df, dv, factor1, factor2):
    """
    Validate data for Levene's test in two-way ANOVA.
    Raises ValueError if assumptions for the test are violated.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned dataframe (rows with NaNs dropped)
    """

    # ---- Column existence ----
    required_cols = {dv, factor1, factor2}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---- Drop NaNs ----
    df_clean = df[[dv, factor1, factor2]].dropna()

    if len(df_clean) < 4:
        raise ValueError(
            "Not enough observations after dropping missing values "
            "for two-way ANOVA."
        )

    # ---- Factor level checks ----
    for factor in (factor1, factor2):
        n_levels = df_clean[factor].nunique()
        if n_levels < 2:
            raise ValueError(
                f"Factor '{factor}' must have at least 2 levels. "
                f"Found {n_levels}."
            )

    # ---- Cell size checks (factor1 × factor2) ----
    cell_sizes = df_clean.groupby([factor1, factor2]).size()

    if (cell_sizes < 2).any():
        bad_cells = cell_sizes[cell_sizes < 2].index.tolist()
        raise ValueError(
            "Each factor1 × factor2 cell must contain at least "
            "2 observations for Levene's test.\n"
            f"Problematic cells: {bad_cells}"
        )

    return df_clean

# Clean levene_two_way_anova using the validator
from scipy.stats import levene

def levene_two_way_anova(df, dv, factor1, factor2, center='median'):
    """
    Levene's test for two-way ANOVA.
    Tests equality of variances across all factor1 × factor2 cells.
    """

    # Validate input
    df_clean = validate_two_way_anova_for_levene(
        df, dv, factor1, factor2
    )

    # Levene across all cells
    groups = [
        sub_df[dv].values
        for _, sub_df in df_clean.groupby([factor1, factor2])
    ]

    stat, p = levene(*groups, center=center)
    return stat, p


def check_normality_of_residuals_visual(df,DV,IV,Covariate):
    model = ols(
        f"{DV} ~ C({IV}) * {Covariate}",
        data=df
    ).fit()

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

    return {"n_resid": int(resid.shape[0])}


def square_column(df, col, inplace=False):
    """
    Squares the values of a column.
    If inplace=False, returns a new DataFrame.
    """
    if not inplace:
        df = df.copy()
        
    df[col] = df[col] ** 2
    return df


def check_vif(df):  #checks multicollinearity, means that two or more predictors in the ANCOVA model are highly correlated with each other. 
    # Build design matrix like the model would 
    X = pd.get_dummies(df[["disease_stage", "age", "gender"]], drop_first=True) #convert CV into dummy variables so they can be used in regression.

        #We calculate variance inflation factor(VIF) for each predictor.
    X = sm.add_constant(X)

    vifs = []
    cols = X.columns.tolist()
    X_vals = X.values.astype(float)

    for i, col in enumerate(cols):
        vif_val = variance_inflation_factor(X_vals, i)
        vifs.append({"feature": col, "vif": float(vif_val)})

    return pd.DataFrame(vifs)


