from pathlib import Path
import pandas as pd

def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")
    return pd.read_csv(path)
