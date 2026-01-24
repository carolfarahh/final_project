# Data handling and numeric conversion
import pandas as pd

# Levene’s test for equality of variances across groups
from scipy.stats import levene


def levene_test(
    df,
    dependent_variable,
    group_variable,
    center="mean",
    dropna=True,
    min_group_size=2
):
    # Validate required columns
    if dependent_variable not in df.columns:
        raise KeyError(f"Missing column: {dependent_variable}")
    if group_variable not in df.columns:
        raise KeyError(f"Missing column: {group_variable}")

    # Validate Levene center option
    if center not in {"median", "mean", "trimmed"}:
        raise ValueError("center must be one of: 'median', 'mean', 'trimmed'")

    # Extract variables
    y = df[dependent_variable]
    g = df[group_variable]

    # Optionally drop rows with missing values in y or g
    if dropna:
        mask = y.notna() & g.notna()
        y = y[mask]
        g = g[mask]

    # Ensure dependent variable is numeric
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        raise ValueError("dependent_variable contains non-numeric values after conversion")

    # Check group counts and minimum group size
    group_sizes = g.value_counts()
    if group_sizes.shape[0] < 2:
        raise ValueError("group_variable must have at least 2 groups")

    too_small = group_sizes[group_sizes < min_group_size]
    if not too_small.empty:
        raise ValueError(
            f"Some groups have fewer than {min_group_size} observations: {too_small.to_dict()}"
        )

    # Build per-group arrays for Levene test
    grouped = [y_num[g == level].to_numpy() for level in group_sizes.index]

    # Run Levene test
    stat, pval = levene(*grouped, center=center)

    # Return results as a one-row DataFrame
    return pd.DataFrame(
        [{
            "test": "levene",
            "center": center,
            "stat": float(stat),
            "pval": float(pval),
            "n_groups": int(group_sizes.shape[0]),
            "group_sizes": group_sizes.to_dict(),
        }]
    )
