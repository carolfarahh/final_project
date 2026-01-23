def _require_columns(df, columns):
    # Create a list of the required columns that are NOT present in the DataFrame
    missing = [c for c in columns if c not in df.columns]
    # If any required columns are missing, stop and raise an error
    if missing:
        # Raise KeyError because the DataFrame does not contain the expected structure
        raise KeyError(f"Missing columns: {missing}")


def unique_values(df, column):
    # Make sure the requested column exists before we use it
    _require_columns(df, [column])
    # Drop missing values (NaN) so they do not appear as a "unique value"
    # Convert values to pandas "string" type for consistent handling
    # Get unique values, convert them to a Python list, then sort them alphabetically
    return sorted(df[column].dropna().astype("string").unique().tolist())


def value_counts_report(df, column):
    # Make sure the requested column exists before counting values
    _require_columns(df, [column])
    # Count how many times each value appears, including NaN (dropna=False)
    return df[column].value_counts(dropna=False)

