def select_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df.loc[:, columns].copy()


def strip_spaces_columns(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("string").str.strip()
    return df


def gene_filter(df, column, values_list):
    df = df.copy()
    cleaned_values_list = [str(x).strip() for x in values_list]
    return df[df[column].isin(cleaned_values_list)].copy()


def drop_missing_required(df, required_columns):
    df = df.copy()
    return df.dropna(subset=required_columns).copy()


import pandas as pd

def convert_numeric_columns(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

