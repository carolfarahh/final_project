import pandas as pd

df = pd.read_csv("HD_dataset.CSV")

print(df.head())
print(df.info())


#def clear_new_data()
import pandas as pd

def load_hd_data(csv_path):
    """
    Load HD dataset and return:
    - analysis_df: Disease stage, Gene factor, Brain-volume-loss
    - demo_df: Age, Gender (kept separate so they don't affect analysis)
    """

    df = pd.read_csv(csv_path)

    # Normalize column names (simple & readable)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "age",
        "gender",
        "disease stage",
        "gene factor",
        "brain-volume-loss"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    demo_df = df[["age", "gender"]].copy()
    analysis_df = df[
        ["disease stage", "gene factor", "brain-volume-loss"]
    ].copy()

    return analysis_df, demo_df


#statical analysis 
#leven's_test 
import pandas as pd
from scipy.stats import levene

groups = [
    g["brain-volume-loss"].dropna()
    for _, g in analysis_df.groupby("disease stage")
]

stat, p = levene(*groups)

print(f"Levene statistic = {stat:.3f}, p = {p:.4f}")

# one-way ANOVA
from scipy.stats import f_oneway

def anova_hd(analysis_df):
    groups = [
        g["brain-volume-loss"].dropna()
        for _, g in analysis_df.groupby("disease stage")
    ]

    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value



