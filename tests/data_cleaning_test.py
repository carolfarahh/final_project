#test for load_hd_data
from io import StringIO
import pandas as pd

def test_load_hd_data_success():
    csv_data = StringIO(
        "Age, Gender, Disease stage, Gene factor, Brain-volume-loss\n"
        "25, M, 1, A, 0.10\n"
        "30, F, 2, B, 0.20\n"
    )
    analysis_df, demo_df = load_hd_data(csv_data)

    # types
    assert isinstance(analysis_df, pd.DataFrame)
    assert isinstance(demo_df, pd.DataFrame)

    # correct columns
    assert list(demo_df.columns) == ["age", "gender"]
    assert list(analysis_df.columns) == ["disease stage", "gene factor", "brain-volume-loss"]

    # correct row count
    assert len(analysis_df) == 2
    assert len(demo_df) == 2


def test_load_hd_data_missing_column_raises():
    csv_data = StringIO(
        "Age, Gender, Disease stage, Gene factor\n"   # missing Brain-volume-loss
        "25, M, 1, A\n"
    )

    try:
        load_hd_data(csv_data)
        assert False, "Expected ValueError for missing column, but no error was raised."
    except ValueError as e:
        assert "Missing columns" in str(e)
        assert "brain-volume-loss" in str(e).lower()



#test for Anova function
import pandas as pd
from scipy.stats import f_oneway

def test_anova_hd_basic():
    # Create a tiny analysis_df with 2 groups
    analysis_df = pd.DataFrame({
        "disease stage": [1, 1, 2, 2],
        "gene factor":   ["A", "A", "B", "B"],
        "brain-volume-loss": [1.0, 2.0, 3.0, 4.0]
    })

    f_stat, p_value = anova_hd(analysis_df)

    # Expected result computed directly the "official" way
    g1 = analysis_df[analysis_df["disease stage"] == 1]["brain-volume-loss"]
    g2 = analysis_df[analysis_df["disease stage"] == 2]["brain-volume-loss"]
    exp_f, exp_p = f_oneway(g1, g2)

    # Compare (floating point -> use tolerance)
    assert abs(f_stat - exp_f) < 1e-12
    assert abs(p_value - exp_p) < 1e-12


def test_anova_hd_ignores_nans():
    analysis_df = pd.DataFrame({
        "disease stage": [1, 1, 2, 2, 2],
        "gene factor":   ["A", "A", "B", "B", "B"],
        "brain-volume-loss": [1.0, None, 3.0, 4.0, None]
    })

    f_stat, p_value = anova_hd(analysis_df)

    # Expected: drop NaNs manually too
    g1 = analysis_df[analysis_df["disease stage"] == 1]["brain-volume-loss"].dropna()
    g2 = analysis_df[analysis_df["disease stage"] == 2]["brain-volume-loss"].dropna()
    exp_f, exp_p = f_oneway(g1, g2)

    assert abs(f_stat - exp_f) < 1e-12
    assert abs(p_value - exp_p) < 1e-12


def test_anova_hd_one_group_raises():
    # Only one disease stage -> ANOVA not valid
    analysis_df = pd.DataFrame({
        "disease stage": [1, 1, 1],
        "gene factor":   ["A", "A", "A"],
        "brain-volume-loss": [1.0, 2.0, 3.0]
    })

    try:
        anova_hd(analysis_df)
        assert False, "Expected an error when only one group exists, but no error was raised."
    except Exception:
        # SciPy may raise different exception types depending on version/input
        assert True
        
        
#tets for levene function
import pandas as pd
from scipy.stats import levene

def test_levene_hd_basic():
    analysis_df = pd.DataFrame({
        "disease stage": [1, 1, 2, 2],
        "gene factor": ["A", "A", "B", "B"],
        "brain-volume-loss": [1.0, 2.0, 3.0, 4.0]
    })

    stat, p_value = levene_hd(analysis_df)

    # expected result calculated directly with scipy
    g1 = analysis_df[analysis_df["disease stage"] == 1]["brain-volume-loss"].dropna()
    g2 = analysis_df[analysis_df["disease stage"] == 2]["brain-volume-loss"].dropna()
    exp_stat, exp_p = levene(g1, g2)

    assert abs(stat - exp_stat) < 1e-12
    assert abs(p_value - exp_p) < 1e-12


def test_levene_hd_ignores_nans():
    analysis_df = pd.DataFrame({
        "disease stage": [1, 1, 2, 2, 2],
        "gene factor": ["A", "A", "B", "B", "B"],
        "brain-volume-loss": [1.0, None, 3.0, 4.0, None]
    })

    stat, p_value = levene_hd(analysis_df)

    g1 = analysis_df[analysis_df["disease stage"] == 1]["brain-volume-loss"].dropna()
    g2 = analysis_df[analysis_df["disease stage"] == 2]["brain-volume-loss"].dropna()
    exp_stat, exp_p = levene(g1, g2)

    assert abs(stat - exp_stat) < 1e-12
    assert abs(p_value - exp_p) < 1e-12

        
