def _require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def unique_values(df, column):
    _require_columns(df, [column])
    return sorted(df[column].dropna().astype("string").unique().tolist())


def value_counts_report(df, column):
    _require_columns(df, [column])
    return df[column].value_counts(dropna=False)
