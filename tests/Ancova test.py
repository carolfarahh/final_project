# tests/test_ancova.py
# Run with: pytest -q

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # prevents GUI popping up during tests


# =========================
# IMPORT YOUR FUNCTIONS HERE
# =========================
# Example:
# from Ancova import (
#     load_and_filter_somatic,
#     check_independence_duplicates,
#     check_linearity_age_dv,
#     run_ancova_with_age_squared,
#     check_homogeneity_of_slopes,
#     validate_ancova_for_levene,
#     levene_ancova,
#     check_normality_of_residuals_visual,
#     log_transform,
#     remove_influential_by_cooks,
# )

# ------------------------------------------
# QUICK FIXES YOUR CODE NEEDS TO PASS TESTS:
# ------------------------------------------
# 1) load_and_filter_somatic: you require "somatic_expansion" but filter "somatic expansion"
#    -> use df["somatic_expansion"] everywhere (since you said you use underscores).
#
# 2) check_homogeneity_of_slopes: formula has extra parentheses and wrong Q usage.
#    -> should look like: f"{DV} ~ C({IV}) * {Covariate}" (+ optional C(CV))
#
# 3) check_normality_of_residuals_visual: formula has extra parentheses
#    -> should be: f"{DV} ~ C({IV}) + {Covariate}" (or + C(CV)) not "* ... ))"


# =========================
# FIXTURES (small fake data)
# =========================
@pytest.fixture
def small_df():
    # a minimal dataset with 3 stages, numeric DV, numeric age, and gender
    return pd.DataFrame({
        "brain_volume_loss": [1.0, 1.2, 0.9, 2.5, 2.6, 2.4, 3.0, 3.2, 2.9],
        "disease_stage":     [1,   1,   1,   2,   2,   2,   3,   3,   3],
        "age":               [30,  32,  31,  45,  47,  44,  55,  56,  54],
        "gender":            ["M", "F", "M", "F", "F", "M", "M", "F", "M"],
        "somatic_expansion": ["yes","yes","yes","yes","yes","yes","yes","yes","yes"],
        "participant_id":    ["p1","p2","p3","p4","p5","p6","p7","p8","p9"]
    })


# =========================
# 0) load_and_filter_somatic
# =========================
def test_load_and_filter_somatic_success(tmp_path, small_df):
    # write temp csv
    p = tmp_path / "tmp.csv"
    small_df.to_csv(p, index=False)

    df_out = load_and_filter_somatic(str(p))
    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) == len(small_df)  # all are somatic_expansion yes
    assert set(["brain_volume_loss", "disease_stage", "age", "gender"]).issubset(df_out.columns)


def test_load_and_filter_somatic_missing_columns_raises(tmp_path, small_df):
    p = tmp_path / "tmp.csv"
    bad = small_df.drop(columns=["age"])
    bad.to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_and_filter_somatic(str(p))


def test_load_and_filter_somatic_empty_filter_raises(tmp_path, small_df):
    p = tmp_path / "tmp.csv"
    bad = small_df.copy()
    bad["somatic_expansion"] = "no"
    bad.to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_and_filter_somatic(str(p))


# ==================================
# 1) check_independence_duplicates
# ==================================
def test_check_independence_duplicates_returns_empty_when_unique(small_df):
    dup = check_independence_duplicates(small_df, id_col="participant_id")
    assert isinstance(dup, pd.DataFrame)
    assert dup.empty


def test_check_independence_duplicates_finds_duplicates(small_df):
    df = small_df.copy()
    df.loc[1, "participant_id"] = df.loc[0, "participant_id"]  # duplicate
    dup = check_independence_duplicates(df, id_col="participant_id")
    assert not dup.empty
    assert (dup["participant_id"] == df.loc[0, "participant_id"]).all()


def test_check_independence_duplicates_raises_if_no_id(small_df):
    df = small_df.drop(columns=["participant_id"])
    with pytest.raises(ValueError):
        check_independence_duplicates(df, id_col="participant_id")


# ==================================
# 2) check_linearity_age_dv
# ==================================
def test_check_linearity_age_dv_returns_dict(small_df):
    res = check_linearity_age_dv(small_df, dv="brain_volume_loss", cov="age", show_plot=False)
    assert isinstance(res, dict)
    assert "pearson_r" in res and "p_value" in res
    assert isinstance(res["pearson_r"], float)
    assert isinstance(res["p_value"], float)


# ==================================
# (optional) ANCOVA with age^2
# ==================================
def test_run_ancova_with_age_squared_runs(small_df):
    table, model = run_ancova_with_age_squared(
        small_df,
        DV="brain_volume_loss",
        IV="disease_stage",
        Covariate="age",
        CV="gender"
    )
    # table is an ANOVA table, model is statsmodels RegressionResults
    assert hasattr(table, "shape")
    assert hasattr(model, "params")


# ==================================
# 3) check_homogeneity_of_slopes
# ==================================
def test_check_homogeneity_of_slopes_has_interaction_row(small_df):
    # this test assumes your function builds a model with IV*Covariate interaction
    table = check_homogeneity_of_slopes(small_df, DV="brain_volume_loss", IV="disease_stage", Covariate="age")
    assert hasattr(table, "index")
    # interaction term should appear
    # depending on formula, it might be: "C(disease_stage):age" or similar
    interaction_found = any(":" in str(idx) for idx in table.index)
    assert interaction_found


# ==================================
# 4) validate_ancova_for_levene + levene_ancova
# ==================================
def test_validate_ancova_for_levene_success(small_df):
    df_clean = validate_ancova_for_levene(small_df, "brain_volume_loss", "disease_stage", "age")
    assert isinstance(df_clean, pd.DataFrame)
    assert not df_clean.empty


def test_validate_ancova_for_levene_raises_missing_col(small_df):
    df = small_df.drop(columns=["age"])
    with pytest.raises(ValueError):
        validate_ancova_for_levene(df, "brain_volume_loss", "disease_stage", "age")


def test_validate_ancova_for_levene_raises_single_group(small_df):
    df = small_df.copy()
    df["disease_stage"] = 1
    with pytest.raises(ValueError):
        validate_ancova_for_levene(df, "brain_volume_loss", "disease_stage", "age")


def test_validate_ancova_for_levene_raises_group_too_small(small_df):
    df = small_df.copy()
    # make stage 3 only 1 row
    df = df[df["disease_stage"] != 3].copy()
    df = pd.concat([df, small_df[small_df["disease_stage"] == 3].head(1)], ignore_index=True)
    with pytest.raises(ValueError):
        validate_ancova_for_levene(df, "brain_volume_loss", "disease_stage", "age")


def test_validate_ancova_for_levene_raises_constant_covariate(small_df):
    df = small_df.copy()
    df["age"] = 50
    with pytest.raises(ValueError):
        validate_ancova_for_levene(df, "brain_volume_loss", "disease_stage", "age")


def test_levene_ancova_returns_stat_p(small_df):
    stat, p = levene_ancova(small_df, "brain_volume_loss", "disease_stage", "age", center="median")
    assert isinstance(stat, float)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


# ==================================
# 5) check_normality_of_residuals_visual
# ==================================
def test_check_normality_of_residuals_visual_returns_n_resid(monkeypatch, small_df):
    # prevent plt.show() from blocking
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: None)

    out = check_normality_of_residuals_visual(
        small_df,
        DV="brain_volume_loss",
        IV="disease_stage",
        Covariate="age"
    )
    assert isinstance(out, dict)
    assert "n_resid" in out
    assert out["n_resid"] > 0


# ==================================
# log_transform
# ==================================
def test_log_transform_adds_column_and_offset(small_df):
    df = small_df.copy()
    df2, offset = log_transform(df, "brain_volume_loss", new_column="log_bvl", offset="auto")
    assert "log_bvl" in df2.columns
    assert isinstance(offset, float)
    assert np.isfinite(df2["log_bvl"]).all()


def test_log_transform_handles_nonpositive():
    df = pd.DataFrame({"x": [-2.0, 0.0, 2.0]})
    df2, offset = log_transform(df, "x", new_column="log_x", offset="auto")
    assert offset > 0
    assert np.isfinite(df2["log_x"]).all()


# ==================================
# 7) remove_influential_by_cooks
# ==================================
def test_remove_influential_by_cooks_returns_clean_and_influential(small_df):
    clean_df, influential_rows, threshold = remove_influential_by_cooks(
        small_df,
        DV="brain_volume_loss",
        IV="disease_stage",
        Covariate="age",
        CV="gender"
    )
    assert isinstance(clean_df, pd.DataFrame)
    assert isinstance(influential_rows, pd.DataFrame)
    assert isinstance(threshold, float)
    # clean_df should never be bigger than original
    assert len(clean_df) <= len(small_df)
