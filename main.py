from src.data_import import load_data
from src.statistical_analysis import gameshowell, one_way_anova
import warnings


from tests.test_data_analysis import games_howell_test, anova_test
from tests.test_data_import import load_data_test

import pandas as pd
import numpy as np
import pingouin as pg


#Load data test
load_data_test_result = load_data_test("Huntington_Disease_Dataset.csv")

df = load_data("Huntington_Disease_Dataset.csv")

list1 = [1,4]
list2 = [3,5]
list3 = [12,13]
# list1 = [1,1]

list_array = np.array([list1, list2, list3])
df_huhu = pd.DataFrame(list_array)
df_huhu["dv"] = df_huhu[0]
df_huhu["factors"] = df_huhu[1]

# one_way_anova(df_huhu, "dv", "factors")

anova_test(df_huhu, "dv", "factors")
