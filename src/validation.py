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
