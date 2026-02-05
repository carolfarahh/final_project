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

def validate_category_levels_n(df, factors_list)
    for factor in factors_list:
        # Check for at least 2 unique levels
        num_categories = df[factor].nunique() 
        if num_categories < 2:
            raise ValueError(
                f"Validation Failed: Factor '{factor}' must have at least 2 categories "
                f"to perform an ANOVA. Found: {num_categories}."
            )

    
def validate_group_size(df, iv, factor2, test)
    
    if test == anova_tukey:
        n_min = 5
    else:
        n_min = 2
    group_sizes = df.groupby([iv, factor2]).size()

    if (group_sizes_sizes < n_min).any():
        bad_cells = group_sizes[group_sizes < n_min].index.tolist()
        raise ValueError(
            f"No enough observations in cells. Can't conduct {test}.\n"
            f"Problematic cells: {bad_cells}")
    else:
        Return None

# Validation / sanity-check function (reusable)
def validate_ancova_for_levene(df, dv, iv, covariate):
    """
    Validate data for Levene's test in ANCOVA.
    Raises ValueError if assumptions for the test are violated.
    Returns cleaned dataframe (rows with NaNs dropped)

    """

    # Check whether variable columns are available in the df
    assert_required_columns(df, [dv, iv, covariate])

    # Drop NaNs for sanity check
    df_clean = df[[dv, iv, covariate]].dropna()

    # Levene ANCOVA requires at least four observations 
    # After dropping NaN, it raises error if there are less than 4 observations
    if len(df_clean) < 3:
        raise ValueError("Not enough observations after dropping missing values.")

    # The number of levels in the iv must be at least 2 to conduct a levene test
    # The number of levels of iv are accessed through .nunique, a ValueError is raised if the assumption is violated
    n_levels = df_clean[iv].nunique()
    if n_levels < 2:
        raise ValueError(
            f"Levene's test requires at least 2 levels in '{iv}'. "
            f"Found {n_levels}."
        )
    # Each group in the independent variable in ancova must have at least 2 observations
    # these lines of code check the number of values in the iv column and raises ValueError if
    # assumption is violated
    group_sizes = df_clean[iv].value_counts()
        if (group_sizes < 2).any():
            bad_levels = group_sizes[group_sizes < 2].index.tolist()
            raise ValueError(
                f"Each level of '{iv}' must have at least 2 observations. "
                f"Problematic levels: {bad_levels}"
            )
    
    

    # The covariate's values in an ANCOVA test must not be a fixed constant. 
    # These lines of codes measures the number of unique values in the column and 
    # raises an error in case of violation of assumption
    if df_clean[covariate].nunique() < 2:
        raise ValueError(
            f"Covariate '{covariate}' has no variability (constant). "
            "ANCOVA cannot be fitted."
        )

    return df_clean 

# Validation function for Two-Way ANOVA Levene
def validate_two_way_anova_for_levene(df, dv, iv, factor2):
    """
    Validate data for Levene's test in two-way ANOVA.
    Raises ValueError if assumptions for the test are violated.

    Returns clean df that can be used in levene test function

    """

    ## Check whether variable columns are available in the df
    assert_required_columns(df, [dv, iv, factor2])

    # For sanity check again, we dropna in case the code didn't capture it before
    df_clean = df[[dv, iv, factor2]].dropna()

    # Factor level checks 
    for factor in (iv, factor2):
        n_levels = df_clean[factor].nunique()
        if n_levels < 2:
            raise ValueError(
                f"Factor '{factor}' must have at least 2 levels. "
                f"Found {n_levels}."
            )

    # Group size checks factor1(iv) × factor2
    # Calculate the size of every unique combination of IV and Factor2 and raises ValueError
    # if their size doesn't meet the criteria
    group_sizes = df_clean.groupby([iv, factor2]).size()

    if (group_sizes_sizes < 2).any():
        bad_cells = group_sizes[group_sizes < 2].index.tolist()
        raise ValueError(
            "Each factor1 × factor2 cell must contain at least "
            "2 observations for Levene's test.\n"
            f"Problematic cells: {bad_cells}")

    return df_clean


def validate_anova_inputs(df, dv, factors, check_small_effect):
    """
    Validates data for ANOVA/ANCOVA.
    factors: list of column names (e.g., [factor1, factor2] or [iv, covariate])
    """
    # Check if all columns exist
    col_list = factors
    col_list.append(dv)

    assert_required_columns(df, factors)

    # Check for missing values (NaN)
    if df[[dv] + factors].isnull().any().any():
        raise ValueError("Missing values detected in variables. Please drop or impute NaNs.")

    # Check for variance in the Dependent Variable

    if df[dv].nunique() <= 1:
        raise ValueError(f"Dependent variable '{dv}' has no variation; analysis is impossible.")

    # Check group sizes (Minimum n=2 per combination)
    # We only check this for categorical factors, not continuous covariates
    categorical_factors = [f for f in factors if df[f].dtype == 'object' or df[f].dtype.name == 'category']
    
    
    if len(categorical_factors) >= 2:
        group_counts = df.groupby(categorical_factors).size()
        if (group_counts < 2).any():
            bad_cells = group_counts[group_counts < 2].index.tolist()
            raise ValueError(f"Group sizes too small (n < 2) in cells: {bad_cells}")

    debug.logging("Data validation successful.")



def validate_anova_inputs(df, dv, factors):
    # 1. Check if columns exist
    for col in [dv] + factors:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    # 2. Categorical Validation & Conversion
    for factor in factors:
        # Check for at least 2 unique levels
        num_categories = df[factor].nunique() 
        
        if num_categories < 2:
            raise ValueError(
                f"Validation Failed: Factor '{factor}' must have at least 2 categories "
                f"to perform an ANOVA. Found: {num_categories}."
            )
        
        # Ensure the factor is actually a 'category' type
        # This prevents the .cat.categories error in your simple effects function
        if not pd.api.types.is_categorical_dtype(df[factor]):
            df[factor] = df[factor].astype('category')
            print(f"Note: Converted '{factor}' to categorical type.")

    # 3. Check for variance in the Dependent Variable (DV must be numeric)
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent Variable '{dv}' must be numeric.")
        
    if df[dv].nunique() <= 1:
        raise ValueError(f"Dependent variable '{dv}' has no variation.")


    def validate_ancova_inputs(df, dv, iv, covariate):
    # 1. Check if columns exist
    required_cols = [dv, iv, covariate]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

    # Create a copy to avoid modifying the original dataframe outside the function
    df = df.copy()

    # 2. Validate Dependent Variable (Numeric)
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"DV '{dv}' must be numeric.")
    if df[dv].nunique() <= 1:
        raise ValueError(f"DV '{dv}' has no variance.")

    # 3. Validate Covariate (Numeric)
    if not pd.api.types.is_numeric_dtype(df[covariate]):
        raise ValueError(f"Covariate '{covariate}' must be numeric for ANCOVA.")

    # 4. Validate Independent Variable (Categorical)
    if df[iv].nunique() < 2:
        raise ValueError(f"IV '{iv}' must have at least 2 levels.")
    
    # Ensure categorical type for statsmodels consistency
    if not isinstance(df[iv].dtype, pd.CategoricalDtype):
        df[iv] = df[iv].astype('category')

    # 5. Check for Missing Values
    nan_count = df[required_cols].isnull().any(axis=1).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} rows contain NaNs and will be dropped by statsmodels.")

    return df


from statsmodels.stats.outliers_influence import variance_inflation_factor

def validate_moderated_regression(df, dv, iv, moderator):
    """
    Validates data specifically for Moderated Regression (IV * Moderator).
    """
    cols = [dv, iv, moderator]
    
    # 1. Check for column existence
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    # 2. Handle Missing Values
    if df[cols].isnull().any().any():
        raise ValueError("Missing values detected. Please drop or impute NaNs before running regression.")

    # 3. Check for Numeric Data
    # Categorical data must be dummy-coded before calculating VIF or interaction terms
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numeric. Ensure categorical variables are dummy-coded.")

    # 4. Check for Zero Variance
    for col in cols:
        if df[col].nunique() <= 1:
            raise ValueError(f"Column '{col}' has no variance. Regression requires at least two distinct values.")

    # 5. Multicollinearity Check (IV vs Moderator)
    # We check the correlation between IV and Moderator before the interaction is created.
    correlation = df[iv].corr(df[moderator])
    if abs(correlation) > 0.80:
        print(f"WARNING: High correlation (r = {correlation:.2f}) between {iv} and {moderator}.")
        print("This may cause unstable coefficients. Consider mean-centering your predictors.")

    # 6. Sample Size Check
    # Rule of thumb: at least 10-20 observations per predictor (IV, Mod, IV*Mod = 3 predictors)
    min_size = 30 
    if len(df) < min_size:
        print(f"WARNING: Sample size (n={len(df)}) is small for moderated regression. Power may be low.")

    return True


def validate_spotlight_analysis(df, dv, iv, covariate):
    """
    Validates requirements for Spotlight (Slopes) Analysis.
    """
    cols = [dv, iv, covariate]

    # 1. Basic Existence & Missing Values
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
    
    if df[cols].isnull().any().any():
        raise ValueError("Missing values detected. Please drop or impute NaNs.")

    # 2. Covariate must be Continuous/Numeric
    if not pd.api.types.is_numeric_dtype(df[covariate]):
        raise TypeError(f"Covariate '{covariate}' must be numeric to calculate Mean and SD.")

    # 3. Variance Check for Covariate
    # If SD is 0, Low/Average/High spots will all be the same value
    sd_cov = df[covariate].std()
    if sd_cov == 0 or np.isnan(sd_cov):
        raise ValueError(f"Covariate '{covariate}' has no variance (SD=0). Spotlight analysis is impossible.")

    # 4. IV Type Check
    # If IV is categorical (object/category), ensure it's not a single level
    if df[iv].dtype == 'object' or df[iv].dtype.name == 'category':
        k = df[iv].nunique()
        if k < 2:
             raise ValueError(f"IV '{iv}' must have at least 2 levels for comparison. Found {k}.")
    
    # 5. Row Count Check
    # Interaction models with robust SEs (HC3) are unstable with very small N
    if len(df) < 20:
        raise ValueError(f"Sample size (n={len(df)}) is too small for reliable spotlight analysis.")

    return True

