import pandas as pd


def _require_columns(df, columns):
    # Build a list of column names that are required but NOT found in the DataFrame
    missing = [c for c in columns if c not in df.columns]
    # If at least one required column is missing, stop the code and show a clear error
    if missing:
        # Raise KeyError because the DataFrame structure is not what we expect
        raise KeyError(f"Missing columns: {missing}")


def select_columns(df, columns):
    # Make sure all requested columns exist before selecting them
    _require_columns(df, columns)
    # Select only these columns (all rows) and return a new independent copy
    return df.loc[:, columns].copy()


def strip_spaces_columns(df, columns):
    # Create a copy so we do not change the original DataFrame6
    df = df.copy()
    # Make sure the columns we want to clean actually exist
    _require_columns(df, columns)
    # Loop over each column we want to clean
    for col in columns:
        # Convert to pandas "string" dtype, then remove leading/trailing spaces from each value
        df[col] = df[col].astype("string").str.strip()
    # Return the cleaned DataFrame copy
    return df


def normalize_case_columns(df, columns, method="lower"):
    # Create a copy so the function is safe and does not modify the original input
    df = df.copy()
    # Validate the method input: we only allow "lower" or "upper"
    if method not in {"lower", "upper"}:
        # Raise ValueError because the user provided an invalid option
        raise ValueError("method must be 'lower' or 'upper'")

    # Make sure the columns we want to normalize exist
    _require_columns(df, columns)
    # Loop over each column we want to standardize
    for col in columns:
        # Convert the column to pandas "string" dtype to safely apply string operations
        series = df[col].astype("string")
        # If method is "lower", convert all text to lowercase
        if method == "lower":
            df[col] = series.str.lower()
        # Otherwise, convert all text to uppercase
        else:
            df[col] = series.str.upper()
    # Return the standardized DataFrame copy
    return df


def gene_filter(df, column, values_list):
    # Create a copy so filtering does not affect the original DataFrame
    df = df.copy()
    # Make sure the filter column exists in the DataFrame
    _require_columns(df, [column])

    # Clean the allowed values list by converting to string and stripping spaces
    cleaned_values_list = [str(x).strip() for x in values_list]
    # Convert the column to string and strip spaces to match the cleaned list
    series = df[column].astype("string").str.strip()
    # Keep only rows where the cleaned column values are in the allowed list, and return a copy
    return df[series.isin(cleaned_values_list)].copy()


def convert_numeric_columns(df, columns):
    # Create a copy so we do not overwrite the original DataFrame
    df = df.copy()
    # Make sure all numeric columns exist before converting
    _require_columns(df, columns)
    # Loop over each column that should become numeric
    for col in columns:
        # Convert values to numbers; invalid values become NaN (because errors="coerce")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Return the converted DataFrame copy
    return df


def drop_missing_required(df, required_columns):
    # Create a copy so the original DataFrame is not modified
    df = df.copy()
    # Make sure all required columns exist (so dropna will work correctly)
    _require_columns(df, required_columns)
    # Drop rows that have NaN in ANY of the required columns, then return a new copy
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
