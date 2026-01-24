def check_independence_duplicates(df, id_col):
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. If you don’t have IDs, independence is judged by study design.")
    dup_ids = df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)
    return dup_ids  # empty = good sign (no repeats)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.formula.api import ols
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def check_independence_duplicates(df, id_col):
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. If you don’t have IDs, independence is judged by study design.")
    dup_ids = df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)
    return dup_ids  # empty = good sign (no repeats)

#New function
def drop_duplicate_subjects(df, id_col, keep="first"):
    """
    Removes duplicated subject IDs.
    keep="first" keeps one row per subject.
    keep=False removes all repeated subjects entirely.
    """
    return df.drop_duplicates(subset=[id_col], keep=keep)

def check_linearity_age_dv(df, dv="Brain_Volume_Loss", cov="Age", show_plot=True, kind="scatter"):
    """
    Checks linearity between a covariate and dependent variable.
    'kind' can be "scatter" or "hexbin" for high-density data.
    """
    # numeric arrays
    x = df[cov].astype(float).values
    y = df[dv].astype(float).values

    # Pearson correlation
    r, p = pearsonr(x, y)

    if show_plot:
        plt.figure(figsize=(8, 5))
        
        if kind == "scatter":
            # s=1 makes dots tiny; alpha adds transparency to show density
            plt.scatter(x, y, s=1, alpha=0.3, edgecolors='none', color='steelblue')
        elif kind == "hexbin":
            # Great for 10k+ points to see where the bulk of data lies
            hb = plt.hexbin(x, y, gridsize=50, cmap='BuPu', mincnt=1)
            plt.colorbar(hb, label='Count')

        # best-fit line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        plt.plot(x_line, m * x_line + b, color='darkorange', linewidth=2, label='Best Fit')

        plt.xlabel(cov)
        plt.ylabel(dv)
        plt.title(f"Linearity: {cov} vs {dv}\n(r={r:.2f}, p={p:.3g})")
        plt.legend()
        plt.show()

    return {"pearson_r": float(r), "p_value": float(p)}

def run_quadratic_ancova(df, dv, iv, covariate):
    """
    Fits an ANCOVA model with a quadratic covariate:
    DV ~ IV + covariate + covariate^2
    """

    df = df.copy()

    # Center the covariate (VERY important for stability)
    cov_c = f"{covariate}_c"
    cov_c_sq = f"{covariate}_c_sq"

    df[cov_c] = df[covariate] - df[covariate].mean()
    df[cov_c_sq] = df[cov_c] ** 2

    formula = f"{dv} ~ C({iv}) + {cov_c} + {cov_c_sq}"

    model = ols(formula, data=df).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    return model, anova_table


import statsmodels.api as sm


def check_homogeneity_of_slopes(df,DV,IV,Covariate):
    model = ols(
        f"{DV} ~ C({IV}) * {Covariate}",
        data=df
    ).fit()

    table = sm.stats.anova_lm(model, typ=2)
    # Key row to check: C(Q('disease stage')):Q('age')
    return table

# def levene_test(df, dependent_variable, group_variable, center="mean", dropna=True, min_group_size=2):

#     if dependent_variable not in df.columns:
#         raise KeyError(f"Missing column: {dependent_variable}")
#     if group_variable not in df.columns:
#         raise KeyError(f"Missing column: {group_variable}")

#     if center not in {"median", "mean", "trimmed"}:
#         raise ValueError("center must be one of: 'median', 'mean', 'trimmed'")

#     y = df[dependent_variable]
#     g = df[group_variable]

#     if dropna:
#         mask = y.notna() & g.notna()
#         y = y[mask]
#         g = g[mask]

#     y_num = pd.to_numeric(y, errors="coerce")
#     if y_num.isna().any():
#         raise ValueError("dependent_variable contains non-numeric values after conversion")

#     group_sizes = g.value_counts()
#     if group_sizes.shape[0] < 2:
#         raise ValueError("group_variable must have at least 2 groups")

#     too_small = group_sizes[group_sizes < min_group_size]
#     if not too_small.empty:
#         raise ValueError(f"Some groups have fewer than {min_group_size} observations: {too_small.to_dict()}")

#     grouped = [y_num[g == level].to_numpy() for level in group_sizes.index]
#     stat, pval = levene(*grouped, center=center)

#     return pd.DataFrame(
#         [{
#             "test": "levene",
#             "center": center,
#             "stat": float(stat),
#             "pval": float(pval),
#             "n_groups": int(group_sizes.shape[0]),
#             "group_sizes": group_sizes.to_dict(),
#         }]
#     )

# def check_homogeneity_of_variance_levene(df, dv, iv):
#     iv = []
#     for _, g in df.groupby(iv):
#         vals = g[dv].dropna().astype(float).values
#         if len(vals) > 0:
#             groups.append(vals)

#     if len(groups) < 2:
#         raise ValueError("Need at least 2 groups with data for Levene’s test.")

#     stat, p = levene(*iv, center="median")
#     return {"levene_stat": float(stat), "p_value": float(p)}

# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.formula.api import ols

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

import numpy as np

def log_transform(
    df,
    column,
    new_column=None,
    offset="auto"
):
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
        offset_used = abs(min_val) + 1 if min_val <= 0 else 0
    else:
        offset_used = float(offset)

    transformed = np.log(x + offset_used)

    if new_column is None:
        new_column = f"log_{column}"

    df_out[new_column] = transformed

    return df_out, offset_used

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


