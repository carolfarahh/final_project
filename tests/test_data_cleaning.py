import pandas as pd
from src.data_cleaning import stage_filter, gene_filter


import pandas as pd
from src.data_cleaning import stage_filter


# =============================================================================
# Tests: stage_filter
#
# This section tests stage_filter(df, column, value).
#
# What these tests check:
#   1) Normal behavior: returns only rows that match the requested stage.
#   2) Common messy data: handles leading/trailing spaces in the column values.
#   3) No side effects: does not change the original input DataFrame.
#   4) Invalid input: raises KeyError when the column does not exist.
#   5) Empty input: returns an empty result for an empty DataFrame.
# =============================================================================


def test_stage_filter_returns_only_requested_stage():
    """Normal case: should return only rows matching the requested stage."""
    df_original = pd.DataFrame(
        {"Disease_Stage": ["Pre-Symptomatic", "Early", "Pre-Symptomatic"]}
    )

    df_filtered = stage_filter(df_original, "Disease_Stage", "Pre-Symptomatic")

    assert len(df_filtered) == 2
    assert df_filtered["Disease_Stage"].eq("Pre-Symptomatic").all()


def test_stage_filter_handles_spaces_in_column_values():
    """
    Common data issue:
    String values may include extra spaces at the beginning or end.
    The function should trim spaces in the column before filtering.
    """
    df_original = pd.DataFrame(
        {"Disease_Stage": [" Pre-Symptomatic  ", "Early", "Pre-Symptomatic "]}
    )

    df_filtered = stage_filter(df_original, "Disease_Stage", "Pre-Symptomatic")

    assert len(df_filtered) == 2
    assert df_filtered["Disease_Stage"].eq("Pre-Symptomatic").all()


def test_stage_filter_does_not_modify_input_dataframe():
    """
    Side-effect check:
    The function should not change the original DataFrame.
    This matters because other project steps may reuse the original data.
    """
    df_original = pd.DataFrame({"Disease_Stage": ["Pre-Symptomatic ", "Early"]})

    _ = stage_filter(df_original, "Disease_Stage", "Pre-Symptomatic")

    # The original value should remain unchanged (still has the trailing space)
    assert df_original.loc[0, "Disease_Stage"] == "Pre-Symptomatic "


def test_stage_filter_missing_column_raises_keyerror():
    """Invalid input: if the column is missing, pandas should raise KeyError."""
    df_original = pd.DataFrame({"Some_Other_Column": [1, 2]})

    try:
        stage_filter(df_original, "Disease_Stage", "Pre-Symptomatic")
        assert False  # If we reach this line, no error was raised (unexpected)
    except KeyError:
        assert True


def test_stage_filter_empty_dataframe_returns_empty_result():
    """Empty input: an empty DataFrame should return an empty filtered result."""
    df_original = pd.DataFrame({"Disease_Stage": []})

    df_filtered = stage_filter(df_original, "Disease_Stage", "Pre-Symptomatic")

    assert len(df_filtered) == 0


import pandas as pd
from src.data_cleaning import gene_filter


# =============================================================================
# Tests: gene_filter
#
# What these tests check:
#   1) Normal behavior: returns only rows whose value is in values_list.
#   2) Handles extra spaces in the DataFrame column values.
#   3) Handles extra spaces in values_list (input list cleaning).
#   4) No side effects: does not change the original input DataFrame.
#   5) Invalid input: raises KeyError when the column does not exist.
# =============================================================================


def test_gene_filter_normal_case():
    """Normal case: should return only rows where the gene is in the list."""
    df_original = pd.DataFrame({"Gene": ["MLH1", "MSH3", "FAN1", "MLH1"]})

    df_filtered = gene_filter(df_original, "Gene", ["MLH1", "MSH3"])

    assert len(df_filtered) == 3
    assert df_filtered["Gene"].isin(["MLH1", "MSH3"]).all()


def test_gene_filter_handles_spaces_in_column_values():
    """Column values may contain extra spaces; the function should trim them."""
    df_original = pd.DataFrame({"Gene": [" MLH1  ", "MSH3", " FAN1 ", "MLH1 "]})

    df_filtered = gene_filter(df_original, "Gene", ["MLH1", "MSH3"])

    assert len(df_filtered) == 3
    assert df_filtered["Gene"].isin(["MLH1", "MSH3"]).all()


def test_gene_filter_handles_spaces_in_values_list():
    """values_list may contain extra spaces; the function should trim them."""
    df_original = pd.DataFrame({"Gene": ["MLH1", "MSH3", "FAN1"]})

    df_filtered = gene_filter(df_original, "Gene", [" MLH1 ", " MSH3 "])

    assert len(df_filtered) == 2
    assert df_filtered["Gene"].isin(["MLH1", "MSH3"]).all()


def test_gene_filter_does_not_modify_input_dataframe():
    """The function should not change the original DataFrame (no side effects)."""
    df_original = pd.DataFrame({"Gene": [" MLH1  ", "FAN1"]})

    _ = gene_filter(df_original, "Gene", ["MLH1"])

    # Original value should remain unchanged (still includes spaces)
    assert df_original.loc[0, "Gene"] == " MLH1  "


def test_gene_filter_missing_column_raises_keyerror():
    """If the column is missing, pandas should raise KeyError."""
    df_original = pd.DataFrame({"X": ["MLH1", "MSH3"]})

    try:
        gene_filter(df_original, "Gene", ["MLH1"])
        assert False
    except KeyError:
        assert True
