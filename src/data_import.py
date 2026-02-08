import pandas as pd
from pathlib import Path
from src.app_logger import logger
#Loading data
def load_data(path):
    """"
    Returns CSV file and raises FileNotFoundError in case the CSV file was not found

    """
    # logger.debug("loading data")
    path_new = Path(path)
    if not path_new.exists():
        raise FileNotFoundError(f"CSV file not found: {path_new.resolve()}")
    logger.debug("Importing Data!")
    return pd.read_csv(path)

