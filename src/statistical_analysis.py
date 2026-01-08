import pingouin as pg
import numpy as np

def data_describe(df):
    data_stats = df.describe()
    return data_stats


def gameshowell(df, dependent_variable, independent_variable):
    results = pg.pairwise_gameshowell(data=df, dv= dependent_variable, between= independent_variable)
    return results
