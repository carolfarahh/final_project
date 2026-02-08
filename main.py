#Load data
from src.data_import import load_data
from src.statistical_assumptions import anova_assumptions_pipeline
from src.data_visualization import eda_plots
from src.data_cleaning import clean_pipeline
from src.EDA import duplicates_info, numeric_summary, categorical_summary, group_descriptives
from src.statistical_analysis import anova_pipeline, ancova_test_pipeline, moderated_regression_pipeline


def project_pipeline():
    df = load_data("/Users/carolfarah/final_projext_yas/final_project/Huntington_Disease_Dataset.csv")
    # Clean_data
    columns_list = ["Patient_ID", "Gene/Factor", "Disease_Stage", "Brain_Volume_Loss", "Age", "Sex"]
    gene_list = ["mlh1", "msh3", "htt (somatic expansion)"]
    numeric_columns = ["Brain_Volume_Loss", "Age"]
    factor_list = ["Disease_Stage", "Sex"]
    df_clean = clean_pipeline(df,selected_columns=columns_list, case_method ="lower", gene_column= "Gene/Factor",genes_keep=gene_list,numeric_columns = numeric_columns, categorical_columns=factor_list)
    #EDA descriptives
    duplicates_summary = duplicates_info(df_clean)
    numeric_summary_df = numeric_summary(df_clean, numeric_columns)
    categorical_summary_df = categorical_summary(df_clean, cols=["Gene/Factor", "Disease_Stage","Sex"])
    group_descriptives_df = group_descriptives(df_clean,"Disease_Stage","Brain_Volume_Loss")

    # #EDA plots
    eda_plots(df_clean)
    
    #ANOVA
    anova_pipeline(df_clean, "Brain_Volume_Loss", "Disease_Stage", "Sex", levene_p = None, alpha=0.05)

    #ANCOVA
    run_moderated_r_boolean = ancova_test_pipeline(df_clean, "Brain_Volume_Loss", "Disease_Stage", "Age", alpha=0.05)

    #Moderate regression
    
    moderated_regression_pipeline(df_clean, "Brain_Volume_Loss", "Disease_Stage", "Age", conduct_moderated_regression = run_moderated_r_boolean, alpha=0.05)
    return 

project_pipeline()
