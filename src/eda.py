from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd


def row_count_log() -> List[Dict[str, Any]]:
    """Create an empty log to track row counts across notebook steps."""
    return []


def log_step(log: List[Dict[str, Any]], step: str, df: pd.DataFrame) -> None:
    """Append step name and current row count."""
    log.append({"step": step, "rows": int(len(df))})


def log_to_df(log: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert log into DataFrame and add percent remaining."""
    out = pd.DataFrame(log)
    if out.empty:
        return out
    first = out.loc[0, "rows"]
    out["pct_remaining"] = (out["rows"] / first * 100) if first else 0.0
    return out


import pandas as pd


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missingness per column: count + percent.
    Sorted by missing_pct descending.
    """
    total = len(df)
    missing_count = df.isna().sum()

    out = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": (missing_count / total * 100) if total else 0.0,
        }
    ).sort_values("missing_pct", ascending=False)

    return out


def sample_for_plot(df: pd.DataFrame, n: int = 20000, random_state: int = 0) -> pd.DataFrame:
    """
    Sample up to n rows for plotting to keep notebooks fast.
    Returns a copy. If df <= n, returns df.copy().
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    if len(df) <= n:
        return df.copy()

    return df.sample(n=n, random_state=random_state).copy()


import pandas as pd


def group_descriptives(df: pd.DataFrame, group_col: str, dv_col: str) -> pd.DataFrame:
    """
    Descriptive stats of dv_col by group_col.
    Returns: n, mean, median, std, min, max, q1, q3, iqr
    """
    g = df.groupby(group_col, dropna=False)[dv_col]

    out = g.agg(
        n="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
    )

    out["iqr"] = out["q3"] - out["q1"]
    return out


def sex_by_stage_table(
    df: pd.DataFrame,
    stage_col: str = "Disease_Stage",
    sex_col: str = "Sex",
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Crosstab of Sex by Disease_Stage.
    If normalize=True, returns row-wise proportions.
    """
    if normalize:
        return pd.crosstab(df[stage_col], df[sex_col], normalize="index")
    return pd.crosstab(df[stage_col], df[sex_col])


def age_by_stage_summary(
    df: pd.DataFrame,
    stage_col: str = "Disease_Stage",
    age_col: str = "Age",
) -> pd.DataFrame:
    """
    Descriptive stats of Age by Disease_Stage.
    """
    return group_descriptives(df, stage_col, age_col)


from typing import Sequence
import pandas as pd


def assert_filtered_gene_values(
    df: pd.DataFrame,
    gene_col: str,
    allowed_values: Sequence[str],
) -> None:
    """
    Sanity check: ensure df[gene_col] contains only allowed_values.
    Intended to be used once after filtering (no plotting/grouping).
    """
    present = pd.Series(df[gene_col].dropna().unique()).astype(str).tolist()
    present_set = set(present)
    allowed_set = set([str(x) for x in allowed_values])

    extras = sorted(list(present_set - allowed_set))
    if extras:
        raise ValueError(f"Unexpected values in {gene_col}: {extras}")
