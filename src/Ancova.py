#before we start with the ancova test we should check if the assumption for it .....
#load_data,normalize and filter the somatic expansion group.
import pandas as pd

def load_and_filter_somatic(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = ["brain_volume_loss", "disease_stage", "age", "gender", "somatic_expansion"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter somatic expansion rows (adjust values list if needed)
    som = df[df["somatic expansion"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])].copy()
    if som.empty:
        raise ValueError("Somatic expansion filter returned 0 rows. Check column values in 'somatic expansion'.")

    # Clean key variables
    som = som.dropna(subset=["brain_volume_loss", "disease_stage", "age", "gender"]).copy()

    return som

# 1) independence of observation 
def check_independence_duplicates(df, id_col="participant_id"): # This step checks the ANCOVA assumption that each observation is independent (each row = one unique participant).
    id_col = id_col.strip().lower()
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. If you don’t have IDs, independence is judged by study design.")
    dup_ids = df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)
    return dup_ids  # the function goes to the id_column and check if we have duplicates and add the to list, if the list came out empty that means we do not have duplicates .

# 2) linearity between covariate and DV (WITH VISUALIZATION)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def check_linearity_age_dv(df, dv="brain_volume_loss", cov="age", show_plot=True):     #this step check the ANCOVA assumption that the relationship between the covariate and the DV is linear.
    # numeric arrays
    x = df[cov].astype(float).values
    y = df[dv].astype(float).values

    # Pearson correlation (numeric check)
    r, p = pearsonr(x, y)  #calculate the pearson correlation coefficient (r) and its p-value, to give statistical indication of linear relationship. 

    # Visualization (scatter + best-fit line)
    # we use scatter plot to show age in x-axis and Brain Volume Loss in the Y-axis if the dots in the scatter plots are roughly following a straight-line pattern, the linearity assumption is valid.
    if show_plot:
        plt.figure()
        plt.scatter(x, y, alpha=0.7)
        #best fit line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        plt.plot(x_line, m * x_line + b)

        plt.xlabel(cov)
        plt.ylabel(dv)
        plt.title(f"Linearity check: {cov} vs {dv} (r={r:.2f}, p={p:.3g})")
        plt.show()

    return {"pearson_r": float(r), "p_value": float(p)}

#we will do this function only if the function above is not valid.
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_ancova_with_age_squared(df, DV, IV, Covariate, CV=None):
    # ANCOVA + quadratic term for the covariate (handles non-linearity)
    if CV is None:
        formula = f"{DV} ~ C({IV}) + {Covariate} + I({Covariate}**2)"
    else:
        formula = f"{DV} ~ C({IV}) + {Covariate} + I({Covariate}**2) + C({CV})"

    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return table, model



# 3) homogeneity of regression slopes.
import statsmodels.api as sm
from statsmodels.formula.api import ols

def check_homogeneity_of_slopes(df,DV,IV,Covariate,):   #checks the ANCOVA assumption that the relationship between the covariate and the DV is the same across all levels of IV. 
    model = ols(
        f"{DV}) ~ C(({IV}) * {Covariate}",   # model that includes an interaction term between  the covariate and the group factor(stage*age)
        data=df
    ).fit()

    table = sm.stats.anova_lm(model, typ=2) #creates  an ANOVA from the fitted regression modle
    
    return table

# 4) homogeneity of variances.
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




# 5)  normality of residuals. 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def check_normality_of_residuals_visual(df,DV,IV,Covariate,):   #checks if the residuals(prediction errors) are approximately normally distributed.
    model = ols(
        f"({DV}) ~ C(({IV})) * ({Covariate}))",
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
# We used Histogram and Q-Q plot because we have a large data set so we cant use normal methods (shapiro-wilk). 
    return {"n_resid": int(resid.shape[0])}


import numpy as np

def log_transform(                  
    df,
    column,
    new_column=None,
    offset="auto"
):                      #applies a natural logarithm transformation to a numeric column.
    """
    Log-transform a column safely.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column to transform
    new_column : str or None
        Name of transformed column (default: log_<column>)
    offset : "auto" or float
        Value added to make data positive before log

    Returns
    -------
    df_out : pandas.DataFrame
    offset_used : float
    """
    df_out = df.copy()

    x = df_out[column].astype(float)

    if offset == "auto":
        min_val = x.min()
        offset_used = abs(min_val) + 1 if min_val <= 0 else 0      # because log cannot take 0 or negative values the function automatically adds an offset if needed.
    else:
        offset_used = float(offset)

    transformed = np.log(x + offset_used)

    if new_column is None:
        new_column = f"log_{column}"

    df_out[new_column] = transformed

    return df_out, offset_used  #log transformation used  when the DV is right-skewed(many low values and few large values) it compresses large values to make the distribution more normal.

import numpy as np


# 7) outliers points. 
from statsmodels.formula.api import ols

def remove_influential_by_cooks(df, DV, IV, Covariate, CV=None):
  

    # Build formula
    if CV is None:
        formula = f"{DV} ~ C({IV}) + {Covariate}"
    else:
        formula = f"{DV} ~ C({IV}) + {Covariate} + C({CV})"

    # Fit model
    model = ols(formula, data=df).fit()

    # Cook's distance
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]

    out = df.copy()
    out["cooks_distance"] = cooks

    # Threshold
    threshold = 4 / len(out)

    # Influential rows (to remove)
    influential_rows = out[out["cooks_distance"] > threshold].copy()

    # Cleaned dataset (keep only non-influential)
    cleaned_df = out[out["cooks_distance"] <= threshold].drop(columns=["cooks_distance"]).copy()

    return cleaned_df, influential_rows, float(threshold)




    