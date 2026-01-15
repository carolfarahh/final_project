from src.data_import import load_data_c
# from src.statistical_analysis import gameshowell, data_describe
import warnings


# from tests.test_data_analysis import games_howell_test, anova_test
# from tests.test_data_import import load_data_test

import pandas as pd
import numpy as np
import pingouin as pg


#Load data test
required_columns = ["Age", "Sex", "Disease_Stage", "Gene/Factor", "Brain_Volume_Loss"]

df = load_data_c("Huntington_Disease_Dataset.csv", required_columns)

data_desscribe = df.groupby(['Gene/Factor', 'Sex'])['Brain_Volume_Loss'].describe().reset_index()
print(data_desscribe)