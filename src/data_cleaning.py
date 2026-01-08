def stage_filter(df, column, value):
    df[column] = df[column].astype(str).str.strip()
    return df[df[column].eq(value)].copy()


def gene_filter(df, column, values_list):
    df[column] = df[column].astype(str).str.strip()
    return df[df[column].isin(values_list)].copy()
