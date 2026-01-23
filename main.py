import pandas as pd
from pathlib import Path
from src.data_import import load_data
from src.data_cleaning import select_columns, strip_spaces_columns, normalize_case_columns, gene_filter, convert_numeric_columns, drop_missing_required, check_influence_cooks_distance


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

 #Add logging


#EDA 


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

levene_test_ANCOVA = 


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
if p_val_homogeneity_of_slopes > 0.05:
    print("The effect of the covariate Age are the same on the level of IV Disease_Stage.\n Conducting ANCOVA")

else:
    print("The effect of the covariate Age differs depending on the level of IV Disease_Stage.\n Conducting moderated regression instead")




