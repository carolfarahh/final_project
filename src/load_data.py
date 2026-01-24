# Handle file paths in a safe, cross-platform way
from pathlib import Path

# Read CSV files into DataFrames
import pandas as pd


def load_data(path):
    # Load a CSV file into a pandas DataFrame
    path = Path(path)

    # Fail fast if the file path does not exist
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

    return pd.read_csv(path)
