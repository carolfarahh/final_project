import pandas as pd
import pingouin as pg
import numpy as np

def load_data(csv_path):
    """
    Load HD dataset and return:
    - analysis_df: Disease stage, Gene factor, Brain-volume-loss
    - demo_df: Age, Gender (kept separate so they don't affect analysis)
    """

    df = pd.read_csv(csv_path)

    # Normalize column names (simple & readable)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "Age",
        "Sex",
        "Disease_Stage",
        "Gene/Factor",
        "Brain_Volume_Loss"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    demo_df = df[["Age", "Gender"]].copy()
    analysis_df = df[
        ["Disease_Stage", "Gene/Factor", "Brain_Volume_Loss"]
    ].copy()

    return analysis_df, demo_df


def load_data_c(file_path, columns_list):
    data = pd.read_csv(file_path)
    return data[columns_list]