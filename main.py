import pandas as pd
from pathlib import Path
from src.data_import import load_data
from src.data_cleaning import select_columns, strip_spaces_columns, normalize_case_columns, gene_filter, convert_numeric_columns, drop_missing_required, check_influence_cooks_distance
from src.statistical_analysis import factor_categorical

from src.app_logger import logger


#Import Data from CSV file
df = load_data("/Users/carolfarah/final_projext_yas/final_project/Huntington_Disease_Dataset.csv")

#Clean Data

#List of relevant columns in our new dataset
columns_list = ["Patient_ID", "Gene/Factor", "Disease_Stage", "Brain_Volume_Loss", "Age", "Sex"]

sub_df = select_columns(df, columns_list) #Columns from original df were selected
sub_df = strip_spaces_columns(sub_df, columns=["Gene/Factor", "Disease_Stage", "Sex"])
sub_df = normalize_case_columns(sub_df, columns= ["Gene/Factor", "Disease_Stage", "Sex"])
sub_df = gene_filter(sub_df, "Gene/Factor", values_list= ["mlh1", "msh3", "htt (somatic expansion)"])
sub_df = drop_missing_required(sub_df, columns_list)
clean_df, removed_rows, threshold= check_influence_cooks_distance(sub_df, "Brain_Volume_Loss", "Age", "Sex")
sub_df = factor_categorical(sub_df, "Disease_Stage", "Sex")

#Add logging


#EDA 
from src.eda import basic_overview, missingness_table, duplicates_info

#basic EDA quality checks on sub_df Here we check: (1) a quick overview, (2) missing values per column, and (3) full-row duplicates.

overview = basic_overview(sub_df)
print("Rows:", overview["n_rows"])
print("Cols:", overview["n_cols"])
print("\nDtypes:")
display(overview["dtypes"])

print("\nMissingness:")
display(missingness_table(sub_df))

print("\nDuplicates:")
print(duplicates_info(sub_df))

from src.eda import numeric_summary, categorical_summary
#Descriptive summaries Now we summarize the distribution of numeric variables (Age, Brain_Volume_Loss) 
#and the frequency of categorical variables (Gene/Factor, Disease_Stage, Sex).
print("Numeric summary:")
display(numeric_summary(sub_df))

print("\nCategorical summary:")
cat = categorical_summary(sub_df, cols=["Gene/Factor", "Disease_Stage", "Sex"])
for col, table in cat.items():
    print(f"\n{col}")
    display(table)

# Group summaries for the research question We summarize Brain_Volume_Loss across Disease_Stage 
# (main factor), and then we check how Age and Sex are distributed across stages (because we will adjust for them later).

from src.eda import group_descriptives, crosstab_counts

print("Brain_Volume_Loss by Disease_Stage:")
display(group_descriptives(sub_df, group_col="Disease_Stage", value_col="Brain_Volume_Loss"))

print("\nAge by Disease_Stage:")
display(group_descriptives(sub_df, group_col="Disease_Stage", value_col="Age"))

print("\nSex by Disease_Stage (counts):")
display(crosstab_counts(sub_df, row_col="Disease_Stage", col_col="Sex"))

#Final sanity checks before moving to modeling We confirm that the 
#dataset contains only the expected Gene/Factor values and that the key analysis columns exist.

from src.eda import assert_required_columns, assert_allowed_values

assert_required_columns(
    sub_df,
    ["Patient_ID", "Gene/Factor", "Disease_Stage", "Brain_Volume_Loss", "Age", "Sex"]
)

assert_allowed_values(
    sub_df,
    col="Gene/Factor",
    allowed_values=["mlh1", "msh3", "htt (somatic expansion)"]
)

assert_allowed_values(
    sub_df,
    col="Sex",
    allowed_values=["male", "female"]
)

assert_allowed_values(
    sub_df,
    col="Disease_Stage",
    allowed_values=["early", "middle", "late", "pre-symptomatic"]
)

print("Sanity checks passed.")

#Boxplot of Brain Volume Loss across Disease Stages This plot helps visualize differences in 
# the distribution of Brain_Volume_Loss between disease stages (median, spread, and potential outliers).

import matplotlib.pyplot as plt

plt.figure()
sub_df.boxplot(column="Brain_Volume_Loss", by="Disease_Stage")
plt.title("Brain_Volume_Loss by Disease_Stage")
plt.suptitle("")  # removes the automatic pandas subtitle
plt.xlabel("Disease_Stage")
plt.ylabel("Brain_Volume_Loss")
plt.xticks(rotation=20)
plt.show()

#Histogram of Brain Volume Loss This plot shows 
#the overall distribution of Brain_Volume_Loss to check skewness and potential outliers.

plt.figure()
plt.hist(sub_df["Brain_Volume_Loss"], bins=30)
plt.title("Distribution of Brain_Volume_Loss")
plt.xlabel("Brain_Volume_Loss")
plt.ylabel("Count")
plt.show()

#Scatter plot (Age vs Brain Volume Loss) with transparency This scatter plot visualizes 
# the relationship between Age and Brain_Volume_Loss. Because the dataset is large, many 
# points overlap; therefore, we use transparency (alpha) to reduce overplotting and make the density of points easier to interpret.

plt.figure()
plt.scatter(sub_df["Age"], sub_df["Brain_Volume_Loss"], s=8, alpha=0.2)
plt.title("Age vs Brain_Volume_Loss (alpha=0.2)")
plt.xlabel("Age")
plt.ylabel("Brain_Volume_Loss")
plt.show()

#Boxplot of Brain Volume Loss by Sex This plot compares the distribution of Brain_Volume_Loss between sexes 
#(median and spread), which supports the “adjust for Sex” part of the research question.

plt.figure()
sub_df.boxplot(column="Brain_Volume_Loss", by="Sex")
plt.title("Brain_Volume_Loss by Sex")
plt.suptitle("")
plt.xlabel("Sex")
plt.ylabel("Brain_Volume_Loss")
plt.show()

#Assumptions
from src.statistical_assumptions import check_independence_duplicates, plot_ancova_linearity, drop_duplicate_subjects, levene_test
# Independence of variables assumption

independence_test = check_independence_duplicates(sub_df, "Patient_ID")

if independence_test.empty:
    print("No duplicates, observations are all independent")
else:
    print(f"Duplicates detected! There are {len(independence_test)/2}")
    sub_df = drop_duplicate_subjects(sub_df, "Patient_ID", keep= "first")
    print(f"{len(independence_test)/2} rows removed!")

#Linearity
linearity_check = check_linearity_age_dv(df, dv="Brain_Volume_Loss", cov="Age", show_plot=True)
linearity_check_p_value = linearity_check["p_value"]

# Homogeneity of Variance (Homoscedasticity)

ancova_levene_stat, ancova_levene_p = levene_ancova(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Age", center='median')

# Linearity of residuals dataset assumption

ancova_linearity_graphs = plot_ancova_linearity(sub_df, dv="Brain_Volume_Loss", iv="Disease_Stage", cov="Age")

while True:
    transform_dataset = input("Does dataset require transformation? (yes/no): ").strip().lower()
    
    if transform_dataset in {"yes", "no"}:
        break
    else:
        print("Please enter 'yes' or 'no'.")

if transform_dataset == "yes":
    clean_df = log_transform(clean_df, "Brain_Volume_Loss", new_column = "Brain_Volume_Loss") #replaces values with new values after transformation
    clean_df = log_transform(clean_df, "Age", new_column = "Age")
    ancova_linearity_graphs = plot_ancova_linearity(sub_df, dv="Brain_Volume_Loss", iv="Disease_Stage", cov="Age")
else:
    print("Proceeding with assumptions analysis")

# Homogeneity of slopes assumption

homogeneity_of_slopes_table = check_homogeneity_of_slopes(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Age")
p_val_homogeneity_of_slopes= anova_table.loc["C(Disease_Stage):Age", "PR(>F)"]

p_iv  = ancova_table.loc["C(Disease_Stage)", "PR(>F)"]
p_cov = ancova_table.loc["Age", "PR(>F)"]

iv_sig  = p_iv  < 0.05
cov_sig = p_cov < 0.05

from src.statistical_analysis import run_ancova, run_ancova_with_statsmodels_posthoc, run_moderated_regression

if p_val_homogeneity_of_slopes > 0.05:
    print("The effect of the covariate Age are the same on the level of IV Disease_Stage.\n Conducting ANCOVA")
    ancova_test_model, ancova_test_table = run_ancova(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Age", ancova_levene_p, linearity_check_p_value, alpha=0.05)
    if not iv_sig and not cov_sig:
        print("An ANCOVA revealed no significant effect of disease stage on brain volume loss, controlling for age," \
        "nor was age significantly associated with brain volume loss.\n")

    elif iv_sig and not cov_sig:
        print("An ANCOVA revealed a significant effect of disease stage on brain volume loss, " \
        f"controlling for age p = {p_iv}, ηp² = {ancova_table_table.iloc[0]["partial_eta_sq"]}. Age was not significantly associated with brain volume loss.")
        print("\nPost-hoc pairwise comparisons of adjusted means were conducted using Bonferroni correction.")
        run_posthoc = True

    elif not iv_sig and cov_sig:
        print("An ANCOVA revealed no significant effect of disease stage on brain volume loss after controlling for age. " \
        f"Age was significantly associated with brain volume loss, p = {ancova_table_table.iloc[1]["PR(>F)"]}.")

    else:
        print("An ANCOVA revealed a significant effect of disease stage on brain volume loss, " \
        f"controlling for age, F(df₁, df₂) = X.XX, p = {p_iv}, ηp² = {ancova_table_table.iloc[0]["partial_eta_sq"]}." \ 
        f"Age was also significantly associated with brain volume loss, F(1, df₂) = X.XX, p = {p_cov}.")
        print("\nPost-hoc pairwise comparisons of adjusted means were conducted using Bonferroni correction.")
        run_posthoc = True

    if run_posthoc = True:
        print("\nPost-hoc pairwise comparisons were conducted using Bonferroni correction.")
        ancova_post_hoc = run_ancova_with_statsmodels_posthoc(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Age", alpha=0.05)
    
    print("The results of the post-hoc are as follows:\n" + ancova_post_hoc)

 
else: #Moderated regression
    print("The effect of the covariate Age differs depending on the level of IV Disease_Stage.\n Conducting moderated regression instead")
    moderated_regression_results = run_moderated_regression(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Age")

    if "IV:Covariate" in moderated_regression_results.index and moderated_regression_results.loc["IV:Covariate", "P>|t|"] < 0.05:
    print("Interaction significant \nRunning spotlight/simple slopes at ±1 SD of Covariate")
    spotlight_analysis_results = run_spotlight_analysis(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Age")

    print("Spotlight Analysis (Simple Slopes)\n")
    print(spotlight_analysis_results)

    elif "IV" in moderated_regression_results.index and moderated_regression_results.loc["IV", "P>|t|"] < 0.05:
    if df[iv].nunique() > 2:
        print("IV main effect significant\n running pairwise post-hoc comparisons between IV levels")
    else:
        print("IV main effect significant\n post-hoc needed (only 2 levels)")


#2 way ANOVA
from src.statistical_assumptions import levene_two_way_anova
from src.statistical_analysis import anova_model, simple_effects_tukey, posthoc_main_effect

anova_levene_stat, anova_levene_p = levene_two_way_anova(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Sex", center="median")


#Running ANOVA test

two_way_anova_results = anova_model(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Sex", anova_levene_p, check_interaction= True, alpha= 0.05 )
p_interaction = two_way_anova_results.loc['C(Sex):C(Disease_Stage)', 'PR(>F)']

if p_interaction < 0.05:
    simple_effects_and_tukey = simple_effects_tukey(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Sex", alpha=0.05, anova_levene_p)
    print(simple_effects_and_tukey)
else:
    additive_anova_results = anova_model(sub_df, "Brain_Volume_Loss", "Disease_Stage", "Sex", anova_levene_p, check_interaction= False, alpha= 0.05 )

    for i in additive_anova_results ['PR(>F)']:
        main_effect_p = ['PR(>F)'][i]
        if main_effect_p < 0.05:
            posthoc_main_effect_results = (sub_df,"Brain_Volume_Loss", "Disease_Stage",factor,main_effect_p,levene_test,alpha=0.05)



def posthoc_main_effect(df,dv,factor,main_effect_p,levene_test,alpha=0.05):




                              







