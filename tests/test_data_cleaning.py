import pandas as pd
import pytest

from src.data_cleaning import (
    select_columns,
    strip_spaces_columns,
    normalize_case_columns,
    gene_filter,
    convert_numeric_columns,
    drop_missing_required,
)



def test_select_columns_returns_only_requested_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    out = select_columns(df, ["A", "C"])
    assert list(out.columns) == ["A", "C"]
    assert out.shape == (2, 2)


def test_select_columns_missing_column_raises_keyerror():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError):
        select_columns(df, ["A", "B"])



def test_strip_spaces_columns_strips_whitespace():
    df = pd.DataFrame({"col1": ["  X  "], "col2": ["   Y"]})
    out = strip_spaces_columns(df, ["col1", "col2"])
    assert out.loc[0, "col1"] == "X"
    assert out.loc[0, "col2"] == "Y"


def test_strip_spaces_columns_missing_column_raises_keyerror():
    df = pd.DataFrame({"col1": ["X"]})
    with pytest.raises(KeyError):
        strip_spaces_columns(df, ["col1", "col2"])



def test_normalize_case_columns_lower_converts_to_lowercase():
    df = pd.DataFrame({"Sex": ["mAlE", "FEMale"]})
    out = normalize_case_columns(df, ["Sex"], method="lower")
    assert out["Sex"].tolist() == ["male", "female"]


def test_normalize_case_columns_invalid_method_raises_valueerror():
    df = pd.DataFrame({"Sex": ["mAlE"]})
    with pytest.raises(ValueError):
        normalize_case_columns(df, ["Sex"], method="title")



def test_gene_filter_keeps_only_requested_values():
    df = pd.DataFrame({"Gene/Factor": ["MLH1", "MSH3", "OTHER"]})
    out = gene_filter(df, "Gene/Factor", ["MLH1", "MSH3"])
    assert out["Gene/Factor"].tolist() == ["MLH1", "MSH3"]


def test_gene_filter_strips_df_values_before_matching():
    df = pd.DataFrame({"Gene/Factor": [" MLH1 ", "MSH3", "OTHER"]})
    out = gene_filter(df, "Gene/Factor", ["MLH1", "MSH3"])
    assert out["Gene/Factor"].tolist() == [" MLH1 ", "MSH3"]



def test_convert_numeric_columns_converts_numeric_strings_to_numbers():
    df = pd.DataFrame({"Age": ["20", "30"], "DV": ["0.1", "0.2"]})
    out = convert_numeric_columns(df, ["Age", "DV"])
    assert out["Age"].tolist() == [20, 30]
    assert out["DV"].tolist() == [0.1, 0.2]


def test_convert_numeric_columns_coerces_invalid_values_to_nan():
    df = pd.DataFrame({"Age": ["20", "bad", None]})
    out = convert_numeric_columns(df, ["Age"])
    assert out["Age"].tolist()[0] == 20
    assert pd.isna(out["Age"].tolist()[1])
    assert pd.isna(out["Age"].tolist()[2])



def test_drop_missing_required_drops_rows_missing_required_columns():
    df = pd.DataFrame(
        {
            "Age": [20, None, 30],
            "DV": [0.1, 0.2, None],
            "Extra": [1, 2, 3],
        }
    )
    out = drop_missing_required(df, ["Age", "DV"])
    assert out.shape[0] == 1
    assert out["Age"].tolist() == [20]
    assert out["DV"].tolist() == [0.1]


def test_drop_missing_required_missing_column_raises_keyerror():
    df = pd.DataFrame({"Age": [20]})
    with pytest.raises(KeyError):
        drop_missing_required(df, ["Age", "DV"])


