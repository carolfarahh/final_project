from typing import Any, Sequence
import pandas as pd

####################### Sanity Checks #######################
# After filtering, we conduct these sanity checks to ensure that 
# The values in our new clean dataframe go by the criterias of our research



# This helper enforces that required columns exist before any EDA step
# It fails fast with a clear error, so analysis does not run on wrong schema
# Also hints that it shouldn't be assigned to a variable

def assert_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# This helper ensures a column contains only a whitelist of allowed values.
# It is useful after filtering, to confirm no unexpected categories remain.

def assert_allowed_values(
    df: pd.DataFrame,
    col: str,
    allowed_values: Sequence[Any],
    dropna: bool = True,
) -> None:
    assert_required_columns(df, [col]) #Uses the previous function to ensure that the column mentioned exists

    s = df[col] #s for series
    if dropna:
        s = s.dropna() #Drops the missing values from our new series

    # Presents the unique values of our series in our original dataframe while also standerdizing the values
    present = set(s.astype(str).unique().tolist()) 
    
    # Presents the unique values of our column in our original dataframe while also standerdizing the values
    allowed = set(pd.Series(list(allowed_values)).astype(str).unique().tolist())

    #Creates a list with all the unexpected values that aren't relevant to our research 
    extras = sorted(list(present - allowed))

    if extras: #In case there are extras it returns an error with the column name and the irrelevant values
        raise ValueError(f"Unexpected values in '{col}': {extras}")
    

# Validation / sanity-check function (reusable)
def validate_ancova_for_levene(df, dv, iv, covariate):
    """
    Validate data for Levene's test in ANCOVA.
    Raises ValueError if assumptions for the test are violated.
    Returns cleaned dataframe (rows with NaNs dropped)

    """

    # Check whether variable columns are available in the df
    assert_required_columns(df, [dv, iv, covariate])

    # Drop NaNs for sanity check
    df_clean = df[[dv, iv, covariate]].dropna()

    # Levene ANCOVA requires at least four observations 
    # After dropping NaN, it raises error if there are less than 4 observations
    if len(df_clean) < 3:
        raise ValueError("Not enough observations after dropping missing values.")

    # The number of levels in the iv must be at least 2 to conduct a levene test
    # The number of levels of iv are accessed through .nunique, a ValueError is raised if the assumption is violated
    n_levels = df_clean[iv].nunique()
    if n_levels < 2:
        raise ValueError(
            f"Levene's test requires at least 2 levels in '{iv}'. "
            f"Found {n_levels}."
        )
    # Each group in the independent variable in ancova must have at least 2 observations
    # these lines of code check the number of values in the iv column and raises ValueError if
    # assumption is violated
    group_sizes = df_clean[iv].value_counts()
    if (group_sizes < 2).any():
        bad_levels = group_sizes[group_sizes < 2].index.tolist()
        raise ValueError(
            f"Each level of '{iv}' must have at least 2 observations. "
            f"Problematic levels: {bad_levels}"
        )

    # The covariate's values in an ANCOVA test must not be a fixed constant. 
    # These lines of codes measures the number of unique values in the column and 
    # raises an error in case of violation of assumption
    if df_clean[covariate].nunique() < 2:
        raise ValueError(
            f"Covariate '{covariate}' has no variability (constant). "
            "ANCOVA cannot be fitted."
        )

    return df_clean 

# Validation function for Two-Way ANOVA Levene
def validate_two_way_anova_for_levene(df, dv, iv, factor2):
    """
    Validate data for Levene's test in two-way ANOVA.
    Raises ValueError if assumptions for the test are violated.

    Returns clean df that can be used in levene test function

    """

    ## Check whether variable columns are available in the df
    assert_required_columns(df, [dv, iv, factor2])

    # For sanity check again, we dropna in case the code didn't capture it before
    df_clean = df[[dv, iv, factor2]].dropna()

    # Factor level checks 
    for factor in (iv, factor2):
        n_levels = df_clean[factor].nunique()
        if n_levels < 2:
            raise ValueError(
                f"Factor '{factor}' must have at least 2 levels. "
                f"Found {n_levels}."
            )

    # Group size checks factor1(iv) × factor2
    # Calculate the size of every unique combination of IV and Factor2 and raises ValueError
    # if their size doesn't meet the criteria
    group_sizes = df_clean.groupby([iv, factor2]).size()

    if (group_sizes_sizes < 2).any():
        bad_cells = group_sizes[group_sizes < 2].index.tolist()
        raise ValueError(
            "Each factor1 × factor2 cell must contain at least "
            "2 observations for Levene's test.\n"
            f"Problematic cells: {bad_cells}")

    return df_clean

