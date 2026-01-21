from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# Row-count logging utilities (useful to document filtering steps)
# ============================================================
def row_count_log() -> List[Dict[str, Any]]:
    """Create an empty log container to track row counts across steps."""
    return []


def log_step(log: List[Dict[str, Any]], step: str, df: pd.DataFrame) -> None:
    """
    Append one step to the log.

    Stores:
      - step: step name
      - rows: number of rows in df
      - pct_remaining: % of rows relative to the FIRST step in the log
    """
    if not isinstance(log, list):
        raise TypeError("log must be a list returned by row_count_log()")

    rows = int(len(df))

    # First step becomes the baseline (100%)
    if len(log) == 0:
        base_rows = rows
        log.append(
            {"step": step, "rows": rows, "pct_remaining": 100.0, "_base_rows": base_rows}
        )
        return

    base_rows = int(log[0].get("_base_rows", log[0].get("rows", 0)))
    pct_remaining = 0.0 if base_rows == 0 else (rows / base_rows) * 100.0

    # Keep the baseline stored in each record to avoid relying on external state
    log.append(
        {"step": step, "rows": rows, "pct_remaining": pct_remaining, "_base_rows": base_rows}
    )


def log_to_df(log: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the log list into a clean DataFrame.

    We remove any internal helper keys (like _base_rows).
    """
    out = pd.DataFrame(log)
    if out.empty:
        return pd.DataFrame(columns=["step", "rows", "pct_remaining"])

    if "_base_rows" in out.columns:
        out = out.drop(columns=["_base_rows"])

    return out[["step", "rows", "pct_remaining"]]


# ============================================================
# Missingness + sampling helpers (EDA-friendly)
# ============================================================
def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return missingness per column.

    Output:
      index: column names
      columns: missing_count, missing_pct
    """
    total = len(df)
    missing = df.isna().sum()
    missing_pct = (missing / total) * 100 if total > 0 else missing * 0

    out = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    out.index.name = "column"
    return out.sort_values("missing_count", ascending=False)


def sample_for_plot(
    df: pd.DataFrame, n: int = 2000, random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Sample up to n rows for plotting (keeps plots responsive on large data).

    Rules:
      - If len(df) <= n: return df.copy()
      - If n <= 0: raise ValueError
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if len(df) <= n:
        return df.copy()

    return df.sample(n=n, random_state=random_state).copy()


# ============================================================
# Group descriptives (core EDA summaries)
# ============================================================
def group_descriptives(
    df: pd.DataFrame, group_col: str, value_col: str, dropna: bool = True
) -> pd.DataFrame:
    """
    Compute descriptive stats of a numeric column per group.

    Output index = group values.
    Output columns: n, mean, median, iqr
    """
    sub = df[[group_col, value_col]].copy()

    if dropna:
        sub = sub.dropna(subset=[group_col, value_col])

    # Force numeric conversion for the value column
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")

    if dropna:
        sub = sub.dropna(subset=[value_col])

    def _iqr(s: pd.Series) -> float:
        # Interquartile range = Q3 - Q1
        return float(s.quantile(0.75) - s.quantile(0.25))

    g = sub.groupby(group_col)[value_col]

    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "iqr": g.apply(_iqr),
        }
    )
    return out


def sex_by_stage_table(
    df: pd.DataFrame, stage_col: str = "Disease_Stage", sex_col: str = "Sex"
) -> pd.DataFrame:
    """
    Crosstab of Sex by Disease Stage (counts).

    Output:
      rows = disease stages
      columns = sex categories
    """
    sub = df[[stage_col, sex_col]].copy()
    sub = sub.dropna(subset=[stage_col, sex_col])
    return pd.crosstab(sub[stage_col], sub[sex_col])


def age_by_stage_summary(
    df: pd.DataFrame, stage_col: str = "Disease_Stage", age_col: str = "Age"
) -> pd.DataFrame:
    """Convenience wrapper: descriptive stats of Age by Disease Stage."""
    return group_descriptives(df, stage_col, age_col)


# ============================================================
# Sanity check helper (post-filter validation)
# ============================================================
def assert_filtered_gene_values(
    df: pd.DataFrame, gene_col: str, allowed_values: List[Any]
) -> None:
    """
    Sanity check: ensure df[gene_col] contains ONLY allowed values.

    Raises:
      ValueError if any unexpected values are present.
    """
    present = pd.Series(df[gene_col].dropna().unique()).astype(str).tolist()
    present_set = set(present)
    allowed_set = set([str(x) for x in allowed_values])

    extras = sorted(list(present_set - allowed_set))
    if extras:
        raise ValueError(f"Unexpected values in {gene_col}: {extras}")
