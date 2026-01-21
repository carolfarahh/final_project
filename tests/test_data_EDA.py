import pandas as pd
import math

from src.EDA import missing_summary, group_counts, numeric_summary, descriptives_by_groups


def test_missing_summary_counts_and_percent():
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, 2]})
    out = missing_summary(df, ["A", "B"])

    a = out.loc[out["column"] == "A"].iloc[0]
    b = out.loc[out["column"] == "B"].iloc[0]

    assert int(a["missing_count"]) == 1
    assert math.isclose(float(a["missing_percent"]), (1 / 3) * 100)

    assert int(b["missing_count"]) == 2
    assert math.isclose(float(b["missing_percent"]), (2 / 3) * 100)


def test_group_counts_stage_gene():
    df = pd.DataFrame(
        {
            "stage": ["Pre", "Pre", "Early", "Pre"],
            "gene": ["MSH3", "MLH1", "MSH3", "MSH3"],
        }
    )
    out = group_counts(df, ["stage", "gene"])

    counts = {(r["stage"], r["gene"]): int(r["count"]) for _, r in out.iterrows()}
    assert counts[("Pre", "MSH3")] == 2
    assert counts[("Pre", "MLH1")] == 1
    assert counts[("Early", "MSH3")] == 1


def test_numeric_summary_basic_stats():
    df = pd.DataFrame({"x": [1, 2, 3, None]})
    out = numeric_summary(df, ["x"]).iloc[0]

    assert out["column"] == "x"
    assert int(out["n"]) == 3
    assert math.isclose(float(out["mean"]), 2.0)
    assert float(out["min"]) == 1.0
    assert float(out["max"]) == 3.0


def test_descriptives_by_groups_stage_gene_with_age_mean():
    df = pd.DataFrame(
        {
            "dv": [1, 2, 3, 4],
            "stage": ["Pre", "Pre", "Early", "Early"],
            "gene": ["MSH3", "MSH3", "MSH3", "MLH1"],
            "age": [10, 20, 30, 40],
        }
    )
    out = descriptives_by_groups(df, "dv", ["stage", "gene"], extra_means=["age"])

    row = out.loc[(out["stage"] == "Pre") & (out["gene"] == "MSH3")].iloc[0]
    assert int(row["n"]) == 2
    assert math.isclose(float(row["mean"]), 1.5)
    assert math.isclose(float(row["mean_age"]), 15.0)
