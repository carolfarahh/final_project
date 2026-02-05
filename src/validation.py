from typing import Any, Sequence
import pandas as pd

####################### Sanity Checks #######################
# After filtering, we conduct these sanity checks to ensure that 
# The values in our new clean dataframe go by the criterias of our research



# This helper enforces that required columns exist before any EDA step
# It fails fast with a clear error, so analysis does not run on wrong schema
# Also hints that it shouldn't be assigned to a variable

def assert_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# This helper ensures a column contains only a whitelist of allowed values.
# It is useful after filtering, to confirm no unexpected categories remain.

def assert_allowed_values(
    df: pd.DataFrame,
    col: str,
    allowed_values: Sequence[Any],
    dropna: bool = True,
) -> None:
    assert_required_columns(df, [col]) #Uses the previous function to ensure that the column mentioned exists

    s = df[col] #s for series
    if dropna:
        s = s.dropna() #Drops the missing values from our new series

    # Presents the unique values of our series in our original dataframe while also standerdizing the values
    present = set(s.astype(str).unique().tolist()) 
    
    # Presents the unique values of our column in our original dataframe while also standerdizing the values
    allowed = set(pd.Series(list(allowed_values)).astype(str).unique().tolist())

    #Creates a list with all the unexpected values that aren't relevant to our research 
    extras = sorted(list(present - allowed))

    if extras: #In case there are extras it returns an error with the column name and the irrelevant values
        raise ValueError(f"Unexpected values in '{col}': {extras}")
    
def validate_missing_values(df, columns_list):
    df_clean = df.copy()
    
    # Check if there are any NaNs in the specific columns
    if df_clean[columns_list].isnull().values.any():
        initial_count = len(df_clean)
        
        # This removes the entire row if ANY of the columns in columns_list have a NaN
        df_clean = df_clean.dropna(subset=columns_list)
        
        dropped_count = initial_count - len(df_clean)
        logger.debug(f"Missing values detected. Dropped {dropped_count} rows for smoother analysis.")
    
    return df_clean

def validate_variance_in_dv(df, dv):
    if df[dv].nunique() <= 1:
        raise ValueError(f"Dependent variable '{dv}' has no variation; analysis is impossible.")
    return None

def validate_category_levels_n(df, factors_list, test)
    
    if test == "moderated_regression":
        minimal_k = 2
    else:
        minimal_k = 3
    
    for factor in factors_list:
        # Check for at least 2 unique levels
        num_categories = df[factor].nunique() 
        if num_categories <= minimal_k:
            raise ValueError(
                f"Validation Failed: Factor '{factor}' must have at least {minimal_k} categories "
                f"to perform {test}. Found: {num_categories}."
            )

    
def validate_group_size(df, iv, factor2, test):
    
    if test == anova_tukey:
        n_min = 5
    else:
        n_min = 2
    group_sizes = df.groupby([iv, factor2]).size()

    if (group_sizes < n_min).any():
        bad_cells = group_sizes[group_sizes < n_min].index.tolist()
        raise ValueError(
            f"No enough observations in cells. Can't conduct {test}.\n"
            f"Problematic cells: {bad_cells}")
    else:
        Return None

def validate_variable_type (df, categorical_list=None, numeric_list=None):
    df_clean = df.copy()
    if categorical_list is not None:
        # Loops over variables in categorical lists and converts them to categorical variable
        for categorical in categorical_list:
            if df_clean[categorical].dtype.name != 'category':
                df_clean[categorical]= df_clean[categorical].astype('category')

    if numeric_list is not None:
        for var in numeric_list:
            if not pd.api.types.is_numeric_dtype(df_clean[var]):
                # Raise error if the user accidentally passed a categorical column as a numeric one
                if pd.api.types.is_categorical_dtype(df_clean[var]):
                    raise ValueError(
                        f"Validation Failed: Variable '{var}' is categorical, but was expected to be numeric/continuous."
                    )
                
                # Otherwise, try to convert 
                df_clean[var] = pd.to_numeric(df_clean[var], errors='coerce')
                
                # Final check: if conversion resulted in all NaNs because it wasn't numeric
                if df_clean[var].isnull().all() and len(df_clean) > 0:
                    raise ValueError(
                        f"Validation Failed: Variable '{var}' could not be converted to numeric values."
                    )
    
    return df_clean

def validate_sample_size_moderated_regression(df, post_hoc = None):
    # At least 10-20 observations per predictor 
    # (IV, Mod, IV*Mod = 3 predictors) -> sample size idealy should at least be n=30
    if post_hoc not None:
        min_size = 30 
    else:
        min_size = 20
    if len(df) < min_size:
        logger.debug(f"WARNING: Sample size (n={len(df)}) is small for moderated regression/spotlight analysis. Power may be low.")

def validate_sample_size_moderated_regression(df):
    # At least 10-20 observations per predictor 
    # (IV, Mod, IV*Mod = 3 predictors) -> sample size idealy should at least be n=30
    min_size =  
    if len(df) < min_size:
        logger.debug(f"WARNING: Sample size (n={len(df)}) is small for moderated regression. Power may be low.")


def anova_validation_pipeline (df, dv, iv, factor2):
    df_copy = df.copy()
    assert_required_columns(df_copy, [dv, iv, factor2])
    df_copy = validate_missing_values(df_copy, [dv, iv, factor2])
    validate_variance_in_dv(df_copy, dv)
    validate_category_levels_n(df_copy, [iv, factor2], "anova")
    validate_group_size(df_copy, iv, factor2, "anova")
    df_copy = validate_variable_type(df_copy, [iv, factor2], [dv])
    return df_copy

def anova_tukey_pipeline (df, dv, iv, factor2):
    df_copy = df.copy()
    assert_required_columns(df_copy, [dv, iv, factor2])
    df_copy = validate_missing_values(df_copy, [dv, iv, factor2])
    validate_variance_in_dv(df_copy, dv)
    validate_category_levels_n(df_copy, [iv, factor2], "anova_tukey")
    validate_group_size(df_copy, iv, factor2, "anova_tukey")
    df_copy = validate_variable_type(df_copy, [iv, factor2], [dv])
    return df_copy


def quadratic_model_adjustment_validation_pipeline (df, dv, iv,covariate):
    df_copy = df.copy()
    assert_required_columns(df_copy, [dv, iv, factor2])
    df_copy = validate_variable_type(df_copy, [iv], [dv, covariate])
    return df_copy

def ancova_validation_pipeline(df, dv, iv, covariate):
    df_copy = df.copy()
    assert_required_columns(df_copy, [dv, iv, covariate])
    df_copy = validate_missing_values(df_copy, [dv, iv, covariate])
    validate_variance_in_dv(df_copy, dv)
    validate_category_levels_n(df_copy, [iv], "ancova")
    validate_group_size(df_copy, iv, "ancova")
    df_copy = validate_variable_type(df_copy, [iv], [dv, covariate])
    return df_copy

def moderated_regression_validation_pipeline(df, dv, iv, moderator):
    df_copy = df.copy()
    assert_required_columns(df_copy, [dv, iv, moderator])
    df_copy = validate_missing_values(df_copy, [dv, iv, moderator])
    validate_variance_in_dv(df_copy, dv)
    validate_category_levels_n(df_copy, [iv], "moderated_regression")
    validate_sample_size_moderated_regression(df, post_hoc=None)
    validate_group_size(df_copy, iv, "moderated_regression")
    df_copy = validate_variable_type(df_copy, [iv], [dv, moderator])
    return df_copy

def moderated_regression_validation_pipeline(df, dv, iv, moderator):
    df_copy = df.copy()
    assert_required_columns(df_copy, [dv, iv, moderator])
    df_copy = validate_missing_values(df_copy, [dv, iv, moderator])
    validate_variance_in_dv(df_copy, dv)
    validate_category_levels_n(df_copy, [iv], "moderated_regression")
    validate_sample_size_moderated_regression(df, post_hoc=True)
    validate_group_size(df_copy, iv, "moderated_regression")
    df_copy = validate_variable_type(df_copy, [iv], [dv, moderator])
    return df_copy



