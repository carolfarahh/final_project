

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