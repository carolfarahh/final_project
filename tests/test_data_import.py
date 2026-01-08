import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data_import
def load_data_test(file_path):
    try:
        load_data(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied when accessing the file: {file_path}")
    except exception as e:
        print("Failed to")