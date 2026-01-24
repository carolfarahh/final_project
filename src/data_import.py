import pandas as pd
from pathlib import Path
from src.app_logger import logger


def load_data(path):
    logger.debug("loading data")
    path_new = Path(path)
    if not path_new.exists():
        raise FileNotFoundError(f"CSV file not found: {path_new.resolve()}")
    return pd.read_csv(path)

