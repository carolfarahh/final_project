#TODO: import relevant libraries and modules
import pandas as pd
import pingouin as pg
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import f_oneway



from src.data_import import load_data_c
from src.data_cleaning import stage_filter, gene_filter, transform_to_float, remove_outlier_isolate_forest
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

df=remove_outlier_isolate_forest(df,"Brain_Volume_Loss")

#TODO: Descriptive statistics: provide general information regarding the data frame
print(data_describe(df))

grouped_data = df.groupby('Gene/Factor')['Brain_Volume_Loss'].mean()
frequency_table = df["Gene/Factor"].value_counts()
print(grouped_data)

#TODO: Conduct levene test to check for homogeinity of variance
#TODO: Conduct relevant statistical test to check for mean differences among groups and report results
#TODO: In case of significance, conduct post-hoc test to find out where the difference occured 

levene_results = levene_test(df, "Brain_Volume_Loss", "Gene/Factor")
is_equal_var = levene_results['equal_var'].item()
if is_equal_var== True:
    f_stat, p_val = anova("Brain_Volume_Loss", df, "Gene/Factor")
    print(f"--- Results: F = {f_stat:.4f}, p = {p_val:.4f} ---")
    if p_val<0.05:
       post_hoc_result = tukey(df, "Brain_Volume_Loss", "Gene/Factor")
       print(post_hoc_result)
    else:
        print("The results were insignificant")
else:
    welch_anova_f_val, welch_anova_p_val = welch_anova(df, "Brain_Volume_Loss", "Gene/Factor")  
    if welch_anova_p_val < 0.05:
        post_hoc_result = gameshowell(df, "Brain_Volume_Loss", "Gene/Factor")
        print(post_hoc_result['pval'])





