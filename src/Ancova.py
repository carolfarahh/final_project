#before we start with the ancova test we should check if the assumption for it .....
#load_data,normalize and filter the somatic expansion group.
import pandas as pd

def load_and_filter_somatic(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = ["brain-volume-loss", "disease stage", "age", "gender", "somatic expansion"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Filter somatic expansion rows (adjust values list if needed)
    som = df[df["somatic expansion"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])].copy()
    if som.empty:
        raise ValueError("Somatic expansion filter returned 0 rows. Check column values in 'somatic expansion'.")

    # Clean key variables
    som = som.dropna(subset=["brain-volume-loss", "disease stage", "age", "gender"]).copy()

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

def check_linearity_age_dv(df, dv="brain-volume-loss", cov="age", show_plot=True):     #this step check the ANCOVA assumption that the relationship between the covariate and the DV is linear.
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


# 3) homogeneity of regression slopes.
import statsmodels.api as sm
from statsmodels.formula.api import ols

def check_homogeneity_of_slopes(df,DV,IV,Covariate,):   #checks the ANCOVA assumption that the relationship between the covariate and the DV is the same across all levels of IV. 
    model = ols(
        f"Q({'DV'}) ~ C(Q({'IV'})) * Q({'Covariate'})))",   # model that includes an interaction term between  the covariate and the group factor(stage*age)
        data=df
    ).fit()

    table = sm.stats.anova_lm(model, typ=2) #creates  an ANOVA from the fitted regression modle
    
    return table

# 4) homogeneity of variances.
def levene_test(df, dependent_variable, group_variable, center="mean", dropna=True, min_group_size=2):    # checks that the variance of DV is similar across all groups (disease stages).
    if dependent_variable not in df.columns:
        raise KeyError(f"Missing column: {dependent_variable}")
    if group_variable not in df.columns:
        raise KeyError(f"Missing column: {group_variable}")

    if center not in {"median", "mean", "trimmed"}:
        raise ValueError("center must be one of: 'median', 'mean', 'trimmed'")

    y = df[dependent_variable]
    g = df[group_variable]

    if dropna:
        mask = y.notna() & g.notna()
        y = y[mask]
        g = g[mask]

    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        raise ValueError("dependent_variable contains non-numeric values after conversion")

    group_sizes = g.value_counts()
    if group_sizes.shape[0] < 2:
        raise ValueError("group_variable must have at least 2 groups")

    too_small = group_sizes[group_sizes < min_group_size]
    if not too_small.empty:
        raise ValueError(f"Some groups have fewer than {min_group_size} observations: {too_small.to_dict()}")

    grouped = [y_num[g == level].to_numpy() for level in group_sizes.index]
    stat, pval = levene(*grouped, center=center)

    return pd.DataFrame(
        [{
            "test": "levene",
            "center": center,
            "stat": float(stat),
            "pval": float(pval),
            "n_groups": int(group_sizes.shape[0]),
            "group_sizes": group_sizes.to_dict(),
        }]
    )



# 5)  normality of residuals. 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

def check_normality_of_residuals_visual(df,DV,IV,Covariate,):   #checks if the residuals(prediction errors) are approximately normally distributed.
    model = ols(
        f"({'DV'}) ~ C(({'IV'})) * ({'Covariate'}))",
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

def sqrt_transform(
    df,
    column,
    new_column=None,
    offset="auto"
):          #applies square-root transformation to a numeric column.
    """
    Square-root transform a column safely.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column to transform
    new_column : str or None
        Name of transformed column (default: sqrt_<column>)
    offset : "auto" or float
        Value added to make data non-negative

    Returns
    -------
    df_out : pandas.DataFrame
    offset_used : float
    """
    df_out = df.copy()

    x = df_out[column].astype(float)

    if offset == "auto":
        min_val = x.min()
        offset_used = abs(min_val) if min_val < 0 else 0        #square-root dose not take negative values so if values are negative the function adds an offset.
    else:
        offset_used = float(offset)

    transformed = np.sqrt(x + offset_used)

    if new_column is None:
        new_column = f"sqrt_{column}"

    df_out[new_column] = transformed

    return df_out, offset_used          #this transformation is useful for moderately skewed data and reduces the effect of large values(less aggressively than log transformation)


#6) multicollinearity VIF between predictors.
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df):  #checks multicollinearity, means that two or more predictors in the ANCOVA model are highly correlated with each other. 
    # Build design matrix like the model would 
    X = pd.get_dummies(df[["disease stage", "age", "gender"]], drop_first=True) #convert CV into dummy variables so they can be used in regression.

        #We calculate variance inflation factor(VIF) for each predictor.
    X = sm.add_constant(X)

    vifs = []
    cols = X.columns.tolist()
    X_vals = X.values.astype(float)

    for i, col in enumerate(cols):
        vif_val = variance_inflation_factor(X_vals, i)
        vifs.append({"feature": col, "vif": float(vif_val)})

    return pd.DataFrame(vifs)

# 7) outliers points. 
from statsmodels.formula.api import ols

def check_influence_cooks_distance(df, DV, IV, Covariate, CV=None): # this step checks whether there are influential participants that have strong effect on the ANCOVA results.
    if CV is None:
        formula = f"{DV} ~ C({IV}) + {Covariate}"
    else:
        formula = f"{DV} ~ C({IV}) + {Covariate} + C({CV})"
        #builds an ANCOVA-style regression model using the dv and predictors and fits the model to the dat. 
    model = ols(formula, data=df).fit()

    infl = model.get_influence()    # calculates cook's distance for every participant.
    cooks = infl.cooks_distance[0]  # adds cook's distance as a new column to the dataset.

    out = df.copy()
    out["cooks_distance"] = cooks

    threshold = 4 / len(out)
    influential = out[out["cooks_distance"] > threshold].sort_values("cooks_distance", ascending=False) # identifies influential participants by selecting rows where cook's distance is greater than the threshold and then sorts the flagged participants from the highest distance to the lowest.

    return {"threshold": float(threshold), "influential_rows": influential}


# ANCOVA TEST.
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def load_and_filter_somatic(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = ["brain-volume-loss", "disease stage", "age", "gender", "somatic expansion"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only somatic expansion patients (edit values if needed)
    somatic_df = df[df["somatic expansion"].astype(str).str.strip().str.lower().isin(
        ["1", "true", "yes", "y"]
    )].copy()

    if somatic_df.empty:
        raise ValueError("No rows found for somatic expansion group. Check values in 'somatic expansion'.")

    # Drop missing values for the ANCOVA variables
    somatic_df = somatic_df.dropna(subset=["brain-volume-loss", "disease stage", "age", "gender"]).copy()

    return somatic_df


def run_ancova(df,DV,IV,Covariate,CV):
    # ANCOVA model
    model = ols(
      f"Q({'DV'}) ~ C(Q({'IV'})) * Q({'Covariate'}) + C(Q({'CV'}))",
        data=df
    ).fit()

    # ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)

    