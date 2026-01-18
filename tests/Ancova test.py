# 1) test for Independence of observations
import pandas as pd

def check_independence_duplicates(df, id_col="participant_id"):
    """
    Checks for repeated participants (duplicate IDs).
    Returns a DataFrame of duplicated rows (empty = good).
    """

    id_col = id_col.strip().lower()

    # normalize column names (safe)
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    if id_col not in df.columns:
        raise ValueError(
            f"Column '{id_col}' not found. Independence must be justified by study design "
            "or add a participant ID column."
        )

    duplicated_rows = df[df.duplicated(subset=[id_col], keep=False)].sort_values(id_col)
    return duplicated_rows

# 2) linearity between the covariate (Age) and DV (brain-volume-loss)
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
    """
    Tests the homogeneity of regression slopes assumption by
    fitting the interaction model: stage * age.

    Returns an ANOVA table (DataFrame).
    """
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) * Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    table = sm.stats.anova_lm(model, typ=2)
    return table

# 4) homogeneity of variances.
from scipy.stats import levene

def check_homogeneity_of_variance_levene(df, dv="brain-volume-loss", group="disease stage"):
    """
    Levene's test checks if DV variance is equal across groups.
    Returns dict with levene_stat and p_value.
    """
    groups = []
    for _, g in df.groupby(group):
        vals = g[dv].dropna().astype(float).values
        if len(vals) > 0:
            groups.append(vals)

    if len(groups) < 2:
        raise ValueError("Need at least 2 groups with data for Levene’s test.")

    stat, p = levene(*groups, center="median")
    return {"levene_stat": float(stat), "p_value": float(p)}

# 5) normality of residuals.
from scipy.stats import shapiro
from statsmodels.formula.api import ols

def check_normality_of_residuals(df):
    """
    Fits ANCOVA model and runs Shapiro-Wilk normality test on residuals.
    Returns dict with shapiro_stat, p_value, and number of residuals.
    """
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) + Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    resid = model.resid.dropna()

    stat, p = shapiro(resid)

    return {"shapiro_stat": float(stat), "p_value": float(p), "n_resid": int(resid.shape[0])}

# 6) multicollinearity VIF between predictors.
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_vif(df):
    """
    Computes VIF for predictors in ANCOVA:
    disease stage, age, gender
    Returns a DataFrame with columns: feature, vif
    """

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

def check_influence_cooks_distance(df):
    """
    Fits ANCOVA model and returns influential rows based on Cook's distance.
    Rule of thumb threshold: 4/n
    """
    model = ols(
        "Q('brain-volume-loss') ~ C(Q('disease stage')) + Q('age') + C(Q('gender'))",
        data=df
    ).fit()

    infl = model.get_influence()
    cooks = infl.cooks_distance[0]

    out = df.copy()
    out["cooks_distance"] = cooks

    threshold = 4 / len(out)
    influential = out[out["cooks_distance"] > threshold].sort_values("cooks_distance", ascending=False)

    return {"threshold": float(threshold), "influential_rows": influential}

#  8) test for Ancova test.
import numpy as np
import pandas as pd
import pytest

from Ancova import load_and_filter_somatic, run_ancova


def make_df_basic(n=60, seed=0):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "brain-volume-loss": rng.normal(2.0, 0.3, n),
        "disease stage": rng.choice(["1", "2", "3"], size=n),
        "age": rng.normal(50, 10, n),
        "gender": rng.choice(["M", "F"], size=n),
        "somatic expansion": rng.choice(["yes", "no"], size=n, p=[0.7, 0.3])
    })
    return df


def test_load_and_filter_somatic_filters_yes_only(tmp_path):
    df = make_df_basic()

    csv_path = tmp_path / "fake_hd.csv"
    df.to_csv(csv_path, index=False)

    som = load_and_filter_somatic(str(csv_path))

    # should contain only "yes/true/1/y" group (here: only "yes" exists)
    assert not som.empty
    assert som["somatic expansion"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"]).all()

    # required columns still exist
    for col in ["brain-volume-loss", "disease stage", "age", "gender"]:
        assert col in som.columns


def test_load_and_filter_somatic_raises_if_missing_required_column(tmp_path):
    df = make_df_basic()

    # drop a required column
    df = df.drop(columns=["gender"])

    csv_path = tmp_path / "missing_col.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_and_filter_somatic(str(csv_path))


def test_load_and_filter_somatic_raises_if_filter_returns_empty(tmp_path):
    df = make_df_basic()
    df["somatic expansion"] = "no"   # ensure filter finds nothing

    csv_path = tmp_path / "no_somatic.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_and_filter_somatic(str(csv_path))


def test_run_ancova_returns_table_and_model():
    # build already-filtered somatic df
    df = make_df_basic()
    som = df[df["somatic expansion"].str.lower() == "yes"].copy()

    ancova_table, model = run_ancova(som)

    # ancova_table should be a DataFrame-like table
    assert hasattr(ancova_table, "shape")
    assert ancova_table.shape[0] >= 1

    # should contain the disease stage term in some form
    assert "C(Q('disease stage'))" in ancova_table.index

    # model should be fitted and have params
    assert hasattr(model, "params")
    assert len(model.params) > 0
