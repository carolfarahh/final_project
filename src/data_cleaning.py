import pandas as pd
from statsmodels.formula.api import ols

# Returns the unavailable columns. 
# This function is used in other functions which depend on the input of the required columns
def _require_columns(df, columns): 
    missing = [c for c in columns if c not in df.columns] #searches for unavailable columns and adds them to a list
    if missing:
        raise KeyError(f"Missing columns: {missing}") #returns error with the missing columns

# Selects the required columns
def select_columns(df, columns):
    _require_columns(df, columns)
    return df.loc[:, columns].copy() #Retruns a copy of the df with the selected columns

# Strips the spaces in the columns to ensure standardization
def strip_spaces_columns(df, columns):
    df = df.copy()
    _require_columns(df, columns)
    for col in columns: #Loops over the columns in the list and strips them of all spaces
        df[col] = df[col].astype("string").str.strip()
    return df

# Standerdizes the case of all values in the columns according to the choice of the user
def normalize_case_columns(df, columns, method="lower"):
    df = df.copy()
    if method not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'") #Returns error if the users choice was unidentefiable

    _require_columns(df, columns)
    for col in columns: #Loops over the columns in the df and coverts all string to either upper or lower case
        series = df[col].astype("string")
        if method == "lower":
            df[col] = series.str.lower()
        else:
            df[col] = series.str.upper()
    return df #Returns the standerdized df

# Filters df according to specific list of values
def gene_filter(df, column, values_list, method="lower"): 
    df = df.copy()
    _require_columns(df, [column]) #Returns error in case the selected column doesn't exist

    # Converts values in values list to string and removes all spaces to ensure successful filtering
    if method not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'")
    
    if method == "lower":
        cleaned_list = [str(x).lower().strip() for x in values_list]
    else:
        cleaned_list = [str(x).upper().strip() for x in values_list]

    # Since the DF is already standardized, we just grab the column as strings
    series = df[column].astype("string")

    return df[series.isin(cleaned_values_list)].copy() #Returns a copy of the filtered df


# Ensures the consinuous variables' values are numeric
def convert_numeric_columns(df, columns):
    df = df.copy()
    _require_columns(df, columns)

# Goes over the columns and converts values to numeric values, and in case of error, it replaces the value with missing datum
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") 
    return df #Returns the new updated df

# Drops missing values in the required columns
def drop_missing_required(df, required_columns):
    df = df.copy()
    _require_columns(df, required_columns)
    return df.dropna(subset=required_columns).copy() 


# Runs the full cleaning pipeline to return an analysis-ready DataFrame.
def clean_pipeline(
    df,
    *,
    selected_columns,
    text_columns_strip,
    text_columns_case,
    case_method,
    gene_column,
    genes_keep,
    numeric_columns,
    required_columns,
):
    df1 = select_columns(df, selected_columns)
    df2 = strip_spaces_columns(df1, text_columns_strip)
    df3 = normalize_case_columns(df2, text_columns_case, method=case_method)
    df4 = gene_filter(df3, gene_column, genes_keep)
    df5 = convert_numeric_columns(df4, numeric_columns)
    df6 = drop_missing_required(df5, required_columns)
    return df6



def remove_influential_by_cooks(df, DV, IV, statistical_test, covariate = None, factor2 = None , check_interaction= None):
    # Build formula depending on statistical test
    if statistical_test == "ANOVA":
        if check_interaction == "True":
            formula = f"{DV} ~ C({IV}) + C({factor2}) + C({IV}):C({factor2})"
        else:
            formula = f"{DV} ~ C({IV}) + C({factor2})"

    elif statistical_test == "ANCOVA":
        formula = f"{DV} ~ C({IV}) + {covariate}"

    elif statistical_test == "Moderated Regression":
        formula = f"{DV} ~ C({IV}) * {covariate}"

    else:
        raise ValueError ("Values must either be 'ANOVA', 'ANCOVA' or 'Moderated Regression'")
    # Fit model
    model = ols(formula, data=df).fit()

    # Cook's distance
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]

    out = df.copy()
    out["cooks_distance"] = cooks

    # Threshold
    threshold = 4 / len(out)

    # Influential rows (to remove)
    influential_rows = out[out["cooks_distance"] > threshold].copy()

    # Cleaned dataset (keep only non-influential)
    cleaned_df = out[out["cooks_distance"] <= threshold].drop(columns=["cooks_distance"]).copy()

    return cleaned_df, influential_rows, float(threshold)


