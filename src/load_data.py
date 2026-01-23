# Import Path to handle file paths in a safe, cross-platform way (Windows/Mac/Linux)
from pathlib import Path

# Import pandas because we will use it to read the CSV file into a DataFrame
import pandas as pd


def load_data(path):
    # Convert the input (string or Path) into a Path object so we can use Path methods
    path = Path(path)

    # Check if the file actually exists before trying to read it
    if not path.exists():
        # If it does not exist, raise an error with the full resolved path to help debugging
        raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

    # Read the CSV file and return it as a pandas DataFrame
    return pd.read_csv(path)
