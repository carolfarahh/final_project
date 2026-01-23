from __future__ import annotations
from typing import Dict, Optional, Sequence

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EDA: Tables after cleaning (use your existing helper functions)
# ============================================================
def eda_tables_after_cleaning(
    sub_def: pd.DataFrame,
    *,
    stage_col: str = "Disease_Stage",
    outcome_col: str = "Brain_Volume_Loss",
    age_col: str = "Age",
    sex_col: str = "Sex",
    cag_col: str = "HTT_CAG_Repeat_Length",
) -> Dict[str, pd.DataFrame]:
    """
    Post-cleaning EDA tables for the research question:
    Disease_Stage vs Brain_Volume_Loss (describe) + Age/Sex balance.

    Returns dict of tables ready to display in the notebook.
    Requires these functions to exist (from your src.eda):
      - missingness_table
      - group_descriptives
      - sex_by_stage_table
      - age_by_stage_summary
    """
    tables: Dict[str, pd.DataFrame] = {}

    # 1) basic shape
    tables["shape"] = pd.DataFrame(
        {"n_rows": [int(len(sub_def))], "n_cols": [int(sub_def.shape[1])]}
    )

    # 2) missingness
    tables["missingness"] = missingness_table(sub_def)

    # 3) stage balance (counts + %)
    stage_counts = sub_def[stage_col].value_counts(dropna=False)
    stage_pct = (stage_counts / stage_counts.sum()) * 100 if stage_counts.sum() else stage_counts * 0
    tables["stage_balance"] = pd.DataFrame({"n": stage_counts, "pct": stage_pct})

    # 4) outcome descriptives by stage
    tables["brain_loss_by_stage"] = group_descriptives(sub_def, stage_col, outcome_col)

    # 5) age descriptives by stage
    if age_col in sub_def.columns:
        tables["age_by_stage"] = age_by_stage_summary(sub_def, stage_col=stage_col, age_col=age_col)

    # 6) sex distribution by stage (counts + row %)
    if sex_col in sub_def.columns:
        ct = sex_by_stage_table(sub_def, stage_col=stage_col, sex_col=sex_col)
        tables["sex_by_stage_counts"] = ct
        tables["sex_by_stage_pct"] = ct.div(ct.sum(axis=1), axis=0) * 100

    # 7) CAG descriptives by stage (optional)
    if cag_col in sub_def.columns:
        tables["cag_by_stage"] = group_descriptives(sub_def, stage_col, cag_col)

    # 8) overall numeric summary (optional, but useful)
    numeric_cols = [c for c in [outcome_col, age_col, cag_col] if c in sub_def.columns]
    if numeric_cols:
        tables["overall_numeric_describe"] = sub_def[numeric_cols].apply(pd.to_numeric, errors="coerce").describe()

    return tables


# ============================================================
# EDA: Sampling dataset for plots (fast on large data)
# ============================================================
def eda_plot_sample(
    sub_def: pd.DataFrame,
    *,
    cols: Sequence[str] = ("Disease_Stage", "Brain_Volume_Loss", "Age", "Sex"),
    n: int = 3000,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Returns a sampled dataframe for plotting.
    Requires your function: sample_for_plot
    """
    keep = [c for c in cols if c in sub_def.columns]
    plot_df = sub_def[keep].copy()
    plot_df = plot_df.dropna(subset=[c for c in ["Disease_Stage", "Brain_Volume_Loss"] if c in plot_df.columns])
    return sample_for_plot(plot_df, n=n, random_state=random_state)


# ============================================================
# EDA: Plots (Matplotlib) for the notebook
# ============================================================
def plot_stage_counts(
    sub_def: pd.DataFrame,
    *,
    stage_col: str = "Disease_Stage",
) -> None:
    counts = sub_def[stage_col].value_counts(dropna=False)
    ax = counts.plot(kind="bar")
    ax.set_title("Counts per Disease Stage")
    ax.set_xlabel(stage_col)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_brain_loss_box_by_stage(
    sub_def: pd.DataFrame,
    *,
    stage_col: str = "Disease_Stage",
    outcome_col: str = "Brain_Volume_Loss",
    n: int = 5000,
    random_state: Optional[int] = 0,
) -> None:
    """
    Boxplot of Brain_Volume_Loss by Disease_Stage using a sample (for speed).
    """
    plot_df = eda_plot_sample(
        sub_def,
        cols=(stage_col, outcome_col),
        n=n,
        random_state=random_state,
    )
    plot_df[outcome_col] = pd.to_numeric(plot_df[outcome_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[outcome_col])

    stages = list(plot_df[stage_col].dropna().unique())
    data = [plot_df.loc[plot_df[stage_col] == s, outcome_col].values for s in stages]

    plt.figure()
    plt.boxplot(data, labels=stages, showfliers=False)
    plt.title("Brain Volume Loss by Disease Stage (sampled)")
    plt.xlabel(stage_col)
    plt.ylabel(outcome_col)
    plt.tight_layout()
    plt.show()


def plot_brain_loss_vs_age_scatter_by_stage(
    sub_def: pd.DataFrame,
    *,
    stage_col: str = "Disease_Stage",
    outcome_col: str = "Brain_Volume_Loss",
    age_col: str = "Age",
    n: int = 6000,
    random_state: Optional[int] = 0,
) -> None:
    """
    Scatter: Age vs Brain_Volume_Loss, colored by stage (sampled).
    """
    plot_df = eda_plot_sample(
        sub_def,
        cols=(stage_col, outcome_col, age_col),
        n=n,
        random_state=random_state,
    )
    plot_df[outcome_col] = pd.to_numeric(plot_df[outcome_col], errors="coerce")
    plot_df[age_col] = pd.to_numeric(plot_df[age_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[outcome_col, age_col])

    plt.figure()
    for stage, grp in plot_df.groupby(stage_col):
        plt.scatter(grp[age_col], grp[outcome_col], s=10, alpha=0.5, label=str(stage))
    plt.title("Brain Volume Loss vs Age by Disease Stage (sampled)")
    plt.xlabel(age_col)
    plt.ylabel(outcome_col)
    plt.legend(title=stage_col)
    plt.tight_layout()
    plt.show()


tables = eda_tables_after_cleaning(sub_def)

tables["shape"]
tables["missingness"].head(10)
tables["stage_balance"]
tables["brain_loss_by_stage"]
tables.get("age_by_stage")
tables.get("sex_by_stage_counts")
tables.get("sex_by_stage_pct")
tables.get("cag_by_stage")
tables.get("overall_numeric_describe")

plot_stage_counts(sub_def)
plot_brain_loss_box_by_stage(sub_def)
plot_brain_loss_vs_age_scatter_by_stage(sub_def)
