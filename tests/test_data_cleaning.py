import pandas as pd
import pytest
from src.data_cleaning import select_columns


def test_select_columns_keeps_only_requested_columns():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    out = select_columns(df, ["A", "C"])
    assert list(out.columns) == ["A", "C"]
    assert out.shape == (2, 2)


def test_select_columns_raises_keyerror_when_missing():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError):
        select_columns(df, ["A", "B"])





from src.data_cleaning import strip_spaces_columns


def test_strip_spaces_columns_removes_spaces():
    df = pd.DataFrame({"a": ["  X  "], "b": ["   Y"]})
    out = strip_spaces_columns(df, ["a", "b"])
    assert out.loc[0, "a"] == "X"
    assert out.loc[0, "b"] == "Y"


def test_strip_spaces_columns_keeps_missing_as_missing():
    df = pd.DataFrame({"a": ["  X  "], "b": [None]})
    out = strip_spaces_columns(df, ["a", "b"])
    assert pd.isna(out.loc[0, "b"])




from src.data_cleaning import gene_filter


def test_gene_filter_keeps_only_requested_values():
    df = pd.DataFrame({
        "Gene/Factor": ["MLH1", "MSH3", "HTT (Somatic Expansion)", "OTHER"]
    })
    out = gene_filter(df, "Gene/Factor", ["MLH1", "MSH3", "HTT (Somatic Expansion)"])
    assert out["Gene/Factor"].tolist() == ["MLH1", "MSH3", "HTT (Somatic Expansion)"]



def test_gene_filter_returns_empty_when_no_match():
    df = pd.DataFrame({"Gene/Factor": ["OTHER"]})
    out = gene_filter(df, "Gene/Factor", ["MLH1"])
    assert out.shape[0] == 0


import pandas as pd
from src.data_cleaning import drop_missing_required


def test_drop_missing_required_drops_rows_with_missing_in_required_columns():
    df = pd.DataFrame({
        "Age": [20, None, 30],
        "Sex": ["F", "M", "F"],
        "Brain Volume Loss": [0.1, 0.2, None],
    })
    out = drop_missing_required(df, ["Age", "Brain Volume Loss"])
    assert out.shape[0] == 1
    assert out["Age"].tolist() == [20]
    assert out["Brain Volume Loss"].tolist() == [0.1]


def test_drop_missing_required_does_not_drop_when_missing_is_in_non_required_column():
    df = pd.DataFrame({
        "Age": [20, 25],
        "Sex": ["F", "M"],
        "Brain Volume Loss": [0.1, 0.2],
        "Extra": [None, "ok"],
    })
    out = drop_missing_required(df, ["Age", "Sex", "Brain Volume Loss"])
    assert out.shape[0] == 2



import pandas as pd
from src.data_cleaning import convert_numeric_columns


def test_convert_numeric_columns_converts_numeric_strings_to_numbers():
    df = pd.DataFrame({"Age": ["20", "30"], "Brain Volume Loss": ["0.1", "0.2"]})
    out = convert_numeric_columns(df, ["Age", "Brain Volume Loss"])

    assert out["Age"].tolist() == [20, 30]
    assert out["Brain Volume Loss"].tolist() == [0.1, 0.2]


def test_convert_numeric_columns_coerces_invalid_values_to_nan():
    df = pd.DataFrame({"Age": ["20", "bad"]})
    out = convert_numeric_columns(df, ["Age"])

    assert out["Age"].tolist()[0] == 20
    assert pd.isna(out["Age"].tolist()[1])



from src.data_cleaning import normalize_case_columns


def test_normalize_case_columns_lower_fixes_mixed_case():
    df = pd.DataFrame({"Sex": ["mAlE", "FEMale"]})
    out = normalize_case_columns(df, ["Sex"], method="lower")

    assert out["Sex"].tolist() == ["male", "female"]


def test_normalize_case_columns_upper_fixes_mixed_case_and_preserves_nan():
    df = pd.DataFrame({"Sex": ["mAlE", None]})
    out = normalize_case_columns(df, ["Sex"], method="upper")

    assert out["Sex"].tolist()[0] == "MALE"
    assert pd.isna(out["Sex"].tolist()[1])


