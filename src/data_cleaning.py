import pandas as pd

def transform_to_float(df, column):
    # 'coerce' handles errors by setting them to NaN (float64)
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df[column]


def fill_NA_mean(df, column):
    column_mean = df[column].mean()
    df[column] = df[column].fillna(column_mean)
    return df[column]


def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers from a DataFrame column using the IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
