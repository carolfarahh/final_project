from typing import Any, Dict, Optional, Sequence
from src.validation import assert_required_columns
from src.app_logger import logger

import pandas as pd


####################### Exploratory Data Analysis #######################

# This function reports full-row duplicates (exact repeated rows).
# Duplicates can inflate results, so we report count and percentage.
def duplicates_info(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame") 

    n = int(len(df)) #Counts the number of rows in data frame
    dup = int(df.duplicated().sum()) #Sums the duplicated rows
    pct = (dup / n) * 100 if n > 0 else 0.0 # To prevent a ZeroDivisionError we assign the percentile variable 0.0
    logger.debug("Gathering duplicates info")
    return {"n_duplicate_rows": float(dup), "duplicate_pct": float(pct)}


# This function summarizes numeric columns using standard descriptive stats.
# If cols is None, it automatically selects numeric columns.
def numeric_summary(df, cols) :
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if cols is None: # In case user didn't enter specific columns
        num_df = df.select_dtypes(include=["number"]) # The columns with numeric values are selected
        cols = list(num_df.columns) #A list with the columns with numeric values is created
    else:
        assert_required_columns(df, cols) #
        # Require these columns to be numeric dtype
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric: #Raises error if certain columns didn't have numeric values
            raise TypeError(f"Non-numeric columns passed to numeric_summary: {non_numeric}") 

    if len(cols) == 0: #In case there are no columns, an empty df is returned
        return pd.DataFrame()


    # In case all the columns stand by the criterias (numeric)
    # Descriptive statistics are returned
    logger.debug("Creating df with descriptive statistics")
    return df[list(cols)].describe().T 


# This function summarizes categorical columns by frequency and percentage.
# It returns one small table per column (all categories by default).

def categorical_summary(df,cols):
# Validate input
    assert_required_columns(df, cols)
    
    # Check if any numeric columns were accidentally passed with the same logic as the previous function
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        raise TypeError(
            f"Numeric columns passed to categorical_summary: {numeric_cols}. "
            "Use numeric_summary() for these instead.")
                    
    # A variable as an empty dictionary containing the final putput
    out = {} 
    # Loops in all of the columns specifed
    for c in cols:
        vc = df[c].value_counts()

        # Saves total as 1 in case it's zero to prevent ZeroDivisionError
        total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0 

        pct = (vc / total) * 100

        out[c] = pd.DataFrame({"count": vc, "pct": pct}) #Converts the dictionary to a dataframe for easier use
    logger.debug("Conducting categorical summary")
    return out #Returns a frequency table as a DataFrame


# This function produces group-wise descriptives for ONE numeric column:
# n, mean, median, and IQR for each group.
# It is a clean EDA step before any statistical modeling.
def group_descriptives(df,group_col,value_col) :
    
    assert_required_columns(df, [group_col, value_col]) #Sanity check

    sub = df[[group_col, value_col]].copy() # Creates a sub dataframe for conducting groupby descriptives

    # Strict numeric conversion: fail if any non-numeric values exist
    try:
        # Tries to convert values in columns to numeric values
        sub[value_col] = pd.to_numeric(sub[value_col], errors="raise") 
    except Exception as e: 
        raise ValueError(f"'{value_col}' must be numeric and convertible to float") from e

    g = sub.groupby(group_col, observed=True)[value_col] #Conducts groupby

    # Calculates quadratic quantiles
    q1 = g.quantile(0.25)
    q3 = g.quantile(0.75)

    # Converts a dictionary of descriptives into a df
    out = pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "iqr": (q3 - q1).astype(float),
        }
    )
    logger.debug("Creating descriptives per group")
    return out




