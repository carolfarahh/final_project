#TODO: import relevant libraries and modules
import pandas as pd
import pingouin as pg
import numpy as np
from src.data_import import load_data
from src.data_cleaning import stage_filter, gene_filter, to_numeric, fill_NA_mean, remove_outliers_iqr
from src.statistical_analysis import data_describe, levene_test, anova, welch_anova, tukey, gameshowell
#TODO: load data

df = load_data("Huntington_Disease_Dataset.csv")
#TODO: Select relevant columns
#TODO: Replace NAN with mean in columns with continuous variables
#TODO: Filter data: based on values
#TODO: Filter data: remove outliers
#TODO: Descriptive statistics: provide general information regarding the data frame
#TODO: Conduct levene test to check for homogeinity of variance
#TODO: Conduct relevant statistical test to check for mean differences among groups and report results
#TODO: In case of significance, conduct post-hoc test to find out where the difference occured 



