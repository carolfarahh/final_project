import pandas as pd


def _require_columns(df, columns):
    # Raise an error if any required columns are missing
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def select_columns(df, columns):
    # Return a copy with only the requested columns
    _require_columns(df, columns)
    return df.loc[:, columns].copy()


def strip_spaces_columns(df, columns):
    # Strip leading/trailing spaces in the given text columns
    df = df.copy()
    _require_columns(df, columns)
    for col in columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def normalize_case_columns(df, columns, method="lower"):
    # Normalize text case in the given columns (lower/upper)
    df = df.copy()

    # Validate the method argument
    if method not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'")

    _require_columns(df, columns)
    for col in columns:
        s = df[col].astype("string")
        df[col] = s.str.lower() if method == "lower" else s.str.upper()
    return df


def gene_filter(df, column, values_list):
    # Keep rows where df[column] matches allowed values (after stripping spaces)
    df = df.copy()
    _require_columns(df, [column])

    # Clean allowed values and compare against cleaned column values
    allowed = [str(x).strip() for x in values_list]
    s = df[column].astype("string").str.strip()
    return df[s.isin(allowed)].copy()


def convert_numeric_columns(df, columns):
    # Convert selected columns to numeric (invalid values become NaN)
    df = df.copy()
    _require_columns(df, columns)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_missing_required(df, required_columns):
    # Drop rows with missing values in any required column
    df = df.copy()
    _require_columns(df, required_columns)
    return df.dropna(subset=required_columns).copy()




