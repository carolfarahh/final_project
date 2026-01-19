import pandas as pd
from pathlib import Path

def load_data(path):
    path_new = Path(path)
    if not path_new.exists():
        raise FileNotFoundError(f"CSV file not found: {path_new.resolve()}")
    return pd.read_csv(path)

