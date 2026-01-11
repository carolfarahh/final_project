import pandas as pd
from src.data_cleaning import stage_filter, gene_filter


def test_stage_filter():
    df = pd.DataFrame({"Disease_Stage": ["Pre-Symptomatic ", "Early", "Pre-Symptomatic"]})

    result = stage_filter(df, "Disease_Stage", "Pre-Symptomatic")

    assert len(result) == 2
    assert (result["Disease_Stage"] == "Pre-Symptomatic").all()
    assert df.loc[0, "Disease_Stage"] == "Pre-Symptomatic "


def test_gene_filter():
    df = pd.DataFrame({"Gene/Factor": ["MLH1", "OTHER", "MSH3", "MLH1"]})

    result = gene_filter(df, "Gene/Factor", ["MLH1", "MSH3"])

    assert len(result) == 3
    assert set(result["Gene/Factor"].unique()) == {"MLH1", "MSH3"}
