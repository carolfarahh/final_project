import pandas as pd
from src.statistical_analysis import welch_anova


def test_welch_anova():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "y":     [1,   2,   1,   10,  11,  10,  5,   6,   5]
    })

    res = welch_anova(df, dependent_variable="y", independent_variable="group")

    assert res is not None
    assert isinstance(res, pd.DataFrame)
    assert len(res) >= 1

from src.statistical_analysis import tukey

def create_df_for_test():
        df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "y":     [1,   2,   1,   10,  11,  10,  5,   6,   5]
        })
        return df

def test_tukey():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "y":     [1,   2,   1,   10,  11,  10,  5,   6,   5]
    })

    res = tukey(df, "y", "group")

    assert res is not None
    assert isinstance(res, pd.DataFrame)
    assert len(res) >= 1

    
from src.statistical_analysis import levene_test


def test_levene_test():
    df = pd.DataFrame({
        "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "y":     [1,   2,   1,   10,  11,  10,  5,   6,   5]
    })

    res = levene_test(df, "y", "group")

    assert res is not None
    assert isinstance(res, pd.DataFrame)
    assert len(res) >= 1
