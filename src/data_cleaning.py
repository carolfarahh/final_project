def stage_filter(df, column, value):
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    return df[df[column].eq(value)].copy()

def gene_filter(df, column, values_list):
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()

    cleaned_values_list = [str(x).strip() for x in values_list]

    return df[df[column].isin(cleaned_values_list)].copy()
