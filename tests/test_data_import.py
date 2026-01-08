import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_import import load_data
def load_data_test(file_path):
    try:
        load_data(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied when accessing the file: {file_path}")
    except Exception as e:
        print("Failed to")
    else:
        print("Data set loaded successfully")