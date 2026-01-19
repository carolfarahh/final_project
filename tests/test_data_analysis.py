import pandas as pd
import pytest

from src.statistical_analysis import levene_test


def test_levene_test_returns_dataframe_with_expected_columns():
    df = pd.DataFrame(
        {
            "Brain_Volume_Loss": [1.0, 1.2, 0.9, 2.0, 2.1, 1.9],
            "Disease_Stage": ["Early", "Early", "Early", "Late", "Late", "Late"],
        }
    )

    out = levene_test(df, "Brain_Volume_Loss", "Disease_Stage", center="mean")
    assert list(out.columns) == ["test", "center", "stat", "pval", "n_groups", "group_sizes"]
    assert out.loc[0, "test"] == "levene"
    assert out.loc[0, "center"] == "mean"
    assert out.loc[0, "n_groups"] == 2
    assert out.loc[0, "group_sizes"] == {"Early": 3, "Late": 3}
    assert 0.0 <= out.loc[0, "pval"] <= 1.0


def test_levene_test_missing_column_raises_keyerror():
    df = pd.DataFrame({"Brain_Volume_Loss": [1.0, 2.0]})
    with pytest.raises(KeyError):
        levene_test(df, "Brain_Volume_Loss", "Disease_Stage", center="mean")


def test_levene_test_invalid_center_raises_valueerror():
    df = pd.DataFrame(
        {
            "Brain_Volume_Loss": [1.0, 2.0, 3.0, 4.0],
            "Disease_Stage": ["A", "A", "B", "B"],
        }
    )
    with pytest.raises(ValueError):
        levene_test(df, "Brain_Volume_Loss", "Disease_Stage", center="mode")


def test_levene_test_one_group_raises_valueerror():
    df = pd.DataFrame(
        {
            "Brain_Volume_Loss": [1.0, 2.0, 3.0],
            "Disease_Stage": ["Only", "Only", "Only"],
        }
    )
    with pytest.raises(ValueError):
        levene_test(df, "Brain_Volume_Loss", "Disease_Stage", center="mean")


def test_levene_test_group_too_small_raises_valueerror():
    df = pd.DataFrame(
        {
            "Brain_Volume_Loss": [1.0, 2.0, 3.0],
            "Disease_Stage": ["A", "A", "B"],
        }
    )
    with pytest.raises(ValueError):
        levene_test(df, "Brain_Volume_Loss", "Disease_Stage", center="mean", min_group_size=2)


def test_levene_test_non_numeric_values_raise_valueerror():
    df = pd.DataFrame(
        {
            "Brain_Volume_Loss": [1.0, "bad", 2.0, 3.0],
            "Disease_Stage": ["A", "A", "B", "B"],
        }
    )
    with pytest.raises(ValueError):
        levene_test(df, "Brain_Volume_Loss", "Disease_Stage", center="mean")
