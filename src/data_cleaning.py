import pandas as pd
from statsmodels.formula.api import ols
from src.app_logger import logger

# Returns the unavailable columns. 
# This function is used in other functions which depend on the input of the required columns
def _require_columns(df, columns): 
    missing = [c for c in columns if c not in df.columns] #searches for unavailable columns and adds them to a list
    if missing:
        raise KeyError(f"Missing columns: {missing}") #returns error with the missing columns
    

# Selects the required columns
def select_columns(df, columns):
    _require_columns(df, columns)
    logger.debug("Selecting columns from original dataset")
    return df.loc[:, columns].copy() #Retruns a copy of the df with the selected columns

# Strips the spaces in the columns to ensure standardization
def strip_spaces_columns(df, columns):
    df = df.copy()
    _require_columns(df, columns)
    for col in columns: #Loops over the columns in the list and strips them of all spaces
        df[col] = df[col].astype("string").str.strip()

    logger.debug("Stripping spaces from columns")
    return df

# Standerdizes the case of all values in the columns according to the choice of the user
def normalize_case_columns(df, columns, method="lower"):
    df = df.copy()
    if method not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'") #Returns error if the users choice was unidentefiable

    _require_columns(df, columns)
    for col in columns: #Loops over the columns in the df and coverts all string to either upper or lower case
        series = df[col].astype("string")
        if method == "lower":
            df[col] = series.str.lower()
        else:
            df[col] = series.str.upper()
    logger.debug("Normalising cases in all columns")
    return df #Returns the standerdized df

# Filters df according to specific list of values
def gene_filter(df, column, values_list, method="lower"): 
    df = df.copy()
    _require_columns(df, [column]) #Returns error in case the selected column doesn't exist

    # Converts values in values list to string and removes all spaces to ensure successful filtering
    if method not in {"lower", "upper"}:
        raise ValueError("method must be 'lower' or 'upper'")
    
    if method == "lower":
        cleaned_list = [str(x).lower() for x in values_list]
    else:
        cleaned_list = [str(x).upper() for x in values_list]

    # Since the DF is already standardized, we just grab the column as strings
    series = df[column].astype("string")

    logger.debug(f"Selecting the {values_list} from {column}")

    return df[series.isin(cleaned_list)].copy() #Returns a copy of the filtered df




# Ensures the consinuous variables' values are numeric
def convert_numeric_columns(df, columns):
    df = df.copy()
    _require_columns(df, columns)

# Goes over the columns and converts values to numeric values, and in case of error, it replaces the value with missing datum
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") 
        logger.debug(f"Ensuring values are numeric in {columns}")
    return df #Returns the new updated df

# Drops missing values in the required columns
def drop_missing_required(df, required_columns):
    df = df.copy()
    _require_columns(df, required_columns)
    logger.debug("Dropping missing values")
    return df.dropna(subset=required_columns).copy() 

def factor_categorical(df, factor_list):
    df = df.copy()
    # Converts variables into categorical variables
    for factor in factor_list:
        df[factor] = df[factor].astype("category")
    logger.debug(f"Converting {factor_list} to categorical variables")
    
    return df


# Runs the full cleaning pipeline to return an analysis-ready DataFrame.
def clean_pipeline(df,selected_columns,case_method,gene_column,genes_keep,numeric_columns, categorical_columns):
    df1 = select_columns(df, selected_columns)
    df2 = strip_spaces_columns(df1, selected_columns)
    df3 = normalize_case_columns(df2, selected_columns, method=case_method)
    df4 = gene_filter(df3, gene_column, genes_keep)
    df5 = convert_numeric_columns(df4, numeric_columns)
    df6 = drop_missing_required(df5, selected_columns)
    df7 = factor_categorical(df6, categorical_columns)
    logger.debug("Data has been cleaned!")
    return df7


def remove_influential_by_cooks(df, DV, IV, statistical_test, covariate=None, factor2=None, check_interaction=None):
    # 1. Clean data for the columns we actually need
    cols_needed = [DV, IV]
    if covariate: cols_needed.append(covariate)
    if factor2: cols_needed.append(factor2)
    df_fit = df.dropna(subset=cols_needed).copy()
    # 2. Build protected formula names
    dv_f = f"Q('{DV}')"
    iv_f = f"C(Q('{IV}'))"
    cov_f = f"Q('{covariate}')" if covariate else None
    f2_f = f"C(Q('{factor2}'))" if factor2 else None
    # Formula Logic
    if statistical_test == "ANOVA":
        if factor2 is not None:
            if check_interaction is True:
                formula = f"{dv_f} ~ {iv_f} * {f2_f}" 
            else:
                formula = f"{dv_f} ~ {iv_f} + {f2_f}"
        else:
            formula = f"{dv_f} ~ {iv_f}"
    elif statistical_test == "ANCOVA":
        formula = f"{dv_f} ~ {iv_f} + {cov_f}"
    elif statistical_test == "Moderated Regression":
        formula = f"{dv_f} ~ {iv_f} * {cov_f}"
    else:
        raise ValueError("statistical_test must be 'ANOVA', 'ANCOVA' or 'Moderated Regression'")
    df_fit = df_fit.astype({col: 'object' for col in df_fit.select_dtypes(['string', 'category']).columns})
    # 4. Fit and Calculate
    model = ols(formula, data=df_fit).fit()
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    # Assign Cook's distance to the cleaned-subset dataframe
    df_fit["cooks_distance"] = cooks
    threshold = 4 / len(df_fit)
    # Filter
    cleaned_df = df_fit[df_fit["cooks_distance"] <= threshold].drop(columns=["cooks_distance"])
    
    return cleaned_df