import pandas as pd

def stage_filter(df, column, value):
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    return df[df[column].eq(value)].copy()


def gene_filter(df, column, values_list):
    df = df.copy()
    df[column] = df[column].astype(str).str.strip()
    return df[df[column].isin(values_list)].copy()


def transform_to_float(df, columns):
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    return df




def remove_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
