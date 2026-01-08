import pandas as pd
from src import data_cleaning

def remove_outliers_iqr_test(df, column, multiplier=1.5):
    try:
        remove_outliers_iqr(df, column, multiplier=1.5)
    except ValueError:
        print("Values in columns are not suitable")
    else:
        print("Outliers removed successfuly")

#Floating point error?


