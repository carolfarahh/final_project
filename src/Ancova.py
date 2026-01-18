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
def check_independence_duplicates(df, id_col="participant_id"):
    id_col = id_col.strip().lower()
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found. If you don’t have IDs, independence is judged by study design.")
    dup_ids = df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)
    return dup_ids  # empty = good sign (no repeats)

# 2) linearity between covariate and DV. 
import numpy as np
from scipy.stats import pearsonr

def check_linearity_age_dv(df, dv="brain-volume-loss", cov="age"):
    x = df[cov].astype(float).values
    y = df[dv].astype(float).values
    r, p = pearsonr(x, y)
    return {"pearson_r": float(r), "p_value": float(p)}

# 3) homogeneity of regression slopes.
import statsmodels.api as sm
from statsmodels.formula.api import ols

def check_homogeneity_of_slopes(df):
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) * Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    table = sm.stats.anova_lm(model, typ=2)
    # Key row to check: C(Q('disease stage')):Q('age')
    return table

# 4) homogeneity of variances.
from scipy.stats import levene

def check_homogeneity_of_variance_levene(df, dv="brain-volume-loss", group="disease stage"):
    groups = []
    for _, g in df.groupby(group):
        vals = g[dv].dropna().astype(float).values
        if len(vals) > 0:
            groups.append(vals)

    if len(groups) < 2:
        raise ValueError("Need at least 2 groups with data for Levene’s test.")

    stat, p = levene(*groups, center="median")
    return {"levene_stat": float(stat), "p_value": float(p)}

# 5)  normality of residuals. 
from scipy.stats import shapiro

def check_normality_of_residuals(df):
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) + Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    resid = model.resid.dropna()
    stat, p = shapiro(resid)

    return {"shapiro_stat": float(stat), "p_value": float(p), "n_resid": int(resid.shape[0])}

#6) multicollinearity VIF between predictors.
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df):
    # Build design matrix like the model would (no DV here)
    X = pd.get_dummies(df[["disease stage", "age", "gender"]], drop_first=True)

    # Add intercept
    X = sm.add_constant(X)

    vifs = []
    cols = X.columns.tolist()
    X_vals = X.values.astype(float)

    for i, col in enumerate(cols):
        vif_val = variance_inflation_factor(X_vals, i)
        vifs.append({"feature": col, "vif": float(vif_val)})

    return pd.DataFrame(vifs)

# 7) outliers points. 
def check_influence_cooks_distance(df):
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) + Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    infl = model.get_influence()
    cooks = infl.cooks_distance[0]  # array

    out = df.copy()
    out["cooks_distance"] = cooks

    # Common rule-of-thumb threshold
    threshold = 4 / len(out)
    influential = out[out["cooks_distance"] > threshold].sort_values("cooks_distance", ascending=False)

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


def run_ancova(df):
    # ANCOVA model
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) + Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    # ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)

    return ancova_table, model

