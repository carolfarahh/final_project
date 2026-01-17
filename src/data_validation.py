def unique_values(df, column):
    return sorted(df[column].dropna().astype("string").unique().tolist())


def value_counts_report(df, column):
    return df[column].value_counts(dropna=False)
