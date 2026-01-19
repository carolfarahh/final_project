import pandas as pd


def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def select_columns(df, columns):
    _require_columns(df, columns)
    return df.loc[:, columns].copy()


def strip_spaces_columns(df, columns):
    df = df.copy()
    _require_columns(df, columns)
    for col in columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def normalize_case_columns(df, columns, method="lower"):
    df = df.copy()
    if method not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'")

    _require_columns(df, columns)
    for col in columns:
        series = df[col].astype("string")
        if method == "lower":
            df[col] = series.str.lower()
        else:
            df[col] = series.str.upper()
    return df


def gene_filter(df, column, values_list):
    df = df.copy()
    _require_columns(df, [column])

    cleaned_values_list = [str(x).strip() for x in values_list]
    series = df[column].astype("string").str.strip()
    return df[series.isin(cleaned_values_list)].copy()


def convert_numeric_columns(df, columns):
    df = df.copy()
    _require_columns(df, columns)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


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
