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

# 2) linearity between covariate and DV (WITH VISUALIZATION)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def check_linearity_age_dv(df, dv="brain-volume-loss", cov="age", show_plot=True):
    # numeric arrays
    x = df[cov].astype(float).values
    y = df[dv].astype(float).values

    # Pearson correlation (numeric check)
    r, p = pearsonr(x, y)

    # Visualization (scatter + best-fit line)
    if show_plot:
        plt.figure()
        plt.scatter(x, y, alpha=0.7)

        # best-fit line
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

def check_homogeneity_of_slopes(df,DV,IV,Covariate,):
    model = ols(
        f"Q({'DV'}) ~ C(Q({'IV'})) * Q({'Covariate'})))",
        data=df
    ).fit()

    table = sm.stats.anova_lm(model, typ=2)
    # Key row to check: C(Q('disease stage')):Q('age')
    return table

# 4) homogeneity of variances.
def levene_test(df, dependent_variable, group_variable, center="mean", dropna=True, min_group_size=2):

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

def check_normality_of_residuals_visual(df,DV,IV,Covariate,):
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

    return {"n_resid": int(resid.shape[0])}


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
from statsmodels.formula.api import ols

def check_influence_cooks_distance(df, DV, IV, Covariate, CV=None):
    if CV is None:
        formula = f"{DV} ~ C({IV}) + {Covariate}"
    else:
        formula = f"{DV} ~ C({IV}) + {Covariate} + C({CV})"

    model = ols(formula, data=df).fit()

    infl = model.get_influence()
    cooks = infl.cooks_distance[0]

    out = df.copy()
    out["cooks_distance"] = cooks

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


def run_ancova(df,DV,IV,Covariate,CV):
    # ANCOVA model
    model = ols(
      f"Q({'DV'}) ~ C(Q({'IV'})) * Q({'Covariate'}) + C(Q({'CV'}))",
        data=df
    ).fit()

    # ANCOVA table
    ancova_table = sm.stats.anova_lm(model, typ=2)

    