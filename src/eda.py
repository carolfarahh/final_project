from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import pandas as pd


# ------------------------------------------------------------
# This helper enforces that required columns exist before any EDA step.
# It fails fast with a clear error, so analysis does not run on wrong schemas.
# ------------------------------------------------------------
def assert_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# ------------------------------------------------------------
# This helper ensures a column contains only a whitelist of allowed values.
# It is useful after filtering, to confirm no unexpected categories remain.
# ------------------------------------------------------------
def assert_allowed_values(
    df: pd.DataFrame,
    col: str,
    allowed_values: Sequence[Any],
    dropna: bool = True,
) -> None:
    assert_required_columns(df, [col])

    s = df[col]
    if dropna:
        s = s.dropna()

    present = set(s.astype(str).unique().tolist())
    allowed = set(pd.Series(list(allowed_values)).astype(str).unique().tolist())

    extras = sorted(list(present - allowed))
    if extras:
        raise ValueError(f"Unexpected values in '{col}': {extras}")


# ------------------------------------------------------------
# This function returns a quick "bird's-eye view" of the dataset:
# shape, column dtypes, and number of unique values per column.
# ------------------------------------------------------------
def basic_overview(df: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "dtypes": df.dtypes.copy(),
        "n_unique": df.nunique(dropna=False).copy(),
    }


# ------------------------------------------------------------
# This function quantifies missing data per column (count + percent).
# It is a standard first step in any EDA to assess data completeness.
# ------------------------------------------------------------
def missingness_table(df: pd.DataFrame, sort: bool = True) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    total = int(len(df))
    missing = df.isna().sum()
    missing_pct = (missing / total) * 100 if total > 0 else missing * 0

    out = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
    out.index.name = "column"

    if sort:
        out = out.sort_values("missing_count", ascending=False)

    return out


# ------------------------------------------------------------
# This function reports full-row duplicates (exact repeated rows).
# Duplicates can inflate results, so we report count and percentage.
# ------------------------------------------------------------
def duplicates_info(df: pd.DataFrame) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    n = int(len(df))
    dup = int(df.duplicated().sum())
    pct = (dup / n) * 100 if n > 0 else 0.0

    return {"n_duplicate_rows": float(dup), "duplicate_pct": float(pct)}


# ------------------------------------------------------------
# This function summarizes numeric columns using standard descriptive stats.
# If cols is None, it automatically selects numeric columns.
# ------------------------------------------------------------
def numeric_summary(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if cols is None:
        num_df = df.select_dtypes(include=["number"])
        cols = list(num_df.columns)
    else:
        assert_required_columns(df, cols)
        # Strict: require these columns to be numeric dtype
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise TypeError(f"Non-numeric columns passed to numeric_summary: {non_numeric}")

    if len(cols) == 0:
        return pd.DataFrame()

    return df[list(cols)].describe().T


# ------------------------------------------------------------
# This function summarizes categorical columns by frequency and percentage.
# It returns one small table per column (all categories by default).
# ------------------------------------------------------------
def categorical_summary(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    dropna: bool = False,
) -> Dict[str, pd.DataFrame]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if cols is None:
        # "categorical" here means non-numeric columns (object/category/bool, etc.)
        cols = list(df.select_dtypes(exclude=["number"]).columns)
    else:
        assert_required_columns(df, cols)

    out: Dict[str, pd.DataFrame] = {}

    for c in cols:
        vc = df[c].value_counts(dropna=dropna)
        total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
        pct = (vc / total) * 100

        out[c] = pd.DataFrame({"count": vc, "pct": pct})

    return out


# ------------------------------------------------------------
# This function produces group-wise descriptives for ONE numeric column:
# n, mean, median, and IQR for each group.
# It is a clean EDA step before any statistical modeling.
# ------------------------------------------------------------
def group_descriptives(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    dropna: bool = True,
) -> pd.DataFrame:
    assert_required_columns(df, [group_col, value_col])

    sub = df[[group_col, value_col]].copy()

    if dropna:
        sub = sub.dropna(subset=[group_col, value_col])

    # Strict numeric conversion: fail if any non-numeric values exist
    try:
        sub[value_col] = pd.to_numeric(sub[value_col], errors="raise")
    except Exception as e:
        raise ValueError(f"'{value_col}' must be numeric and convertible to float") from e

    g = sub.groupby(group_col, observed=True)[value_col]

    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)

    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "iqr": (q3 - q1).astype(float),
        }
    )

    return out


# ------------------------------------------------------------
# This function creates a contingency table (counts) between two categorical columns.
# It is useful to check imbalance across groups (e.g., Sex by Stage).
# ------------------------------------------------------------
def crosstab_counts(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    dropna: bool = True,
) -> pd.DataFrame:
    assert_required_columns(df, [row_col, col_col])

    sub = df[[row_col, col_col]].copy()
    if dropna:
        sub = sub.dropna(subset=[row_col, col_col])

    return pd.crosstab(sub[row_col], sub[col_col])
