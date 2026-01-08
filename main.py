from src.data_import import load_data

from tests.test_data_import import load_data_test
import pandas as pd
df = load_data("Huntington_Disease_Dataset.CSV")
load_data_test(df)
# print(df["Effect"].unique())