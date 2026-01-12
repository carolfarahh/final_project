#TODO: import relevant libraries and modules
import pandas as pd
import pingouin as pg
import numpy as np

from src.data_import import load_data_c
from src.data_cleaning import stage_filter, gene_filter, transform_to_float, fill_NA_mean, remove_outliers_iqr
from src.statistical_analysis import data_describe, levene_test, anova, welch_anova, tukey, gameshowell
#TODO: load data

required_columns = ["Age", "Sex", "Disease_Stage", "Gene/Factor", "Brain_Volume_Loss"]
df = load_data_c("Huntington_Disease_Dataset.csv", required_columns)

#TODO: Replace NAN with mean in columns with continuous variables

coninouous_variables = ["Brain_Volume_Loss"]

df = transform_to_float(df, coninouous_variables)

#TODO: Filter data: based on values

gene_list= ["HTT (Somatic Expansion)", "MLH1", "MSH3"]
df = gene_filter(df, "Gene/Factor", gene_list)
df= stage_filter(df, "Disease_Stage", "Pre-Symptomatic")

#TODO: Filter data: remove outliers
# new_df = remove_outliers_iqr(df, "Brain_Volume_Loss", multiplier=1.5) 

# multiplier=1.5
# Q1 = df["Brain_Volume_Loss"].quantile(0.005)
# Q3 = df["Brain_Volume_Loss"].quantile(0.995)
# IQR = Q3 - Q1

# lower_bound = Q1 - multiplier * IQR
# upper_bound = Q3 + multiplier * IQR

# outliers = df[(df["Brain_Volume_Loss"] < lower_bound) | (df["Brain_Volume_Loss"] > upper_bound)]
# print(len(outliers))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

print(df.shape)
print(df.head())



# z_scores = np.abs(stats.zscore(df))
# outliers_z = np.where(z_scores > 3)

# print("Outlier positions (row, col):")
# print(list(zip(outliers_z[0][:10], outliers_z[1][:10])))

# print(new_df)
#TODO: Descriptive statistics: provide general information regarding the data frame
#TODO: Conduct levene test to check for homogeinity of variance
#TODO: Conduct relevant statistical test to check for mean differences among groups and report results
#TODO: In case of significance, conduct post-hoc test to find out where the difference occured 



