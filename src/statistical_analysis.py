# Import pandas for data handling and numeric conversion
import pandas as pd

# Import Levene’s test function from SciPy (used to test equality of variances across groups)
from scipy.stats import levene


def levene_test(
    df,
    dependent_variable,
    group_variable,
    center="mean",
    dropna=True,
    min_group_size=2
):
    # Check that the dependent variable column exists in the DataFrame
    if dependent_variable not in df.columns:
        # Raise KeyError because the column name is not found
        raise KeyError(f"Missing column: {dependent_variable}")

    # Check that the grouping (factor) column exists in the DataFrame
    if group_variable not in df.columns:
        # Raise KeyError because the column name is not found
        raise KeyError(f"Missing column: {group_variable}")

    # Validate the 'center' parameter (Levene test allows only these center options)
    if center not in {"median", "mean", "trimmed"}:
        # Raise ValueError because the user provided an invalid option
        raise ValueError("center must be one of: 'median', 'mean', 'trimmed'")

    # Extract the dependent variable values (what we measure)
    y = df[dependent_variable]

    # Extract the group labels (how we split the data into groups)
    g = df[group_variable]

    # Optionally remove rows where either y or g is missing (NaN)
    if dropna:
        # Build a mask that keeps only rows where both y and g are not missing
        mask = y.notna() & g.notna()
        # Apply the mask to y
        y = y[mask]
        # Apply the mask to g (same rows must be kept)
        g = g[mask]

    # Convert y into numeric values (invalid values become NaN because errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")

    # If conversion created any NaN values, it means there were non-numeric values in y
    if y_num.isna().any():
        # Stop the analysis because Levene test requires numeric dependent variable
        raise ValueError("dependent_variable contains non-numeric values after conversion")

    # Count how many observations exist in each group level
    group_sizes = g.value_counts()

    # Levene test needs at least two groups to compare variances
    if group_sizes.shape[0] < 2:
        # Stop because one group cannot be compared to anything
        raise ValueError("group_variable must have at least 2 groups")

    # Identify groups that have fewer observations than the minimum allowed
    too_small = group_sizes[group_sizes < min_group_size]

    # If any group is too small, stop and report which groups fail the condition
    if not too_small.empty:
        # Convert the small groups counts to a dictionary for a clear error message
        raise ValueError(
            f"Some groups have fewer than {min_group_size} observations: {too_small.to_dict()}"
        )

    # Split the numeric dependent values into separate arrays, one array per group level
    grouped = [y_num[g == level].to_numpy() for level in group_sizes.index]

    # Run Levene’s test on all group arrays (compares variances across groups)
    stat, pval = levene(*grouped, center=center)

    # Return results in a one-row DataFrame (easy to log, print, or save later)
    return pd.DataFrame(
        [{
            # Name of the test we ran
            "test": "levene",
            # Which center method was used (mean/median/trimmed)
            "center": center,
            # Test statistic value
            "stat": float(stat),
            # P-value from the test
            "pval": float(pval),
            # Number of groups included in the test
            "n_groups": int(group_sizes.shape[0]),
            # Group sizes as a dictionary (group label -> number of observations)
            "group_sizes": group_sizes.to_dict(),
        }]
    )
