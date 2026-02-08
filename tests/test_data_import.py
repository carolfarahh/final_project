import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import pytest
from pathlib import Path


from src.data_import import load_data  
def test_load_data_success(tmp_path):

    # Arrange: create a temporary CSV
    test_file = tmp_path / "test.csv"
    df_expected = pd.DataFrame({"a": [1, 2, 3],"b": [4, 5, 6]})
    df_expected.to_csv(test_file, index=False)
    df_loaded = load_data(test_file)
    # Assert
    pd.testing.assert_frame_equal(df_loaded, df_expected)

def test_load_data_file_not_found(): #In case file doesn't exist
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


        