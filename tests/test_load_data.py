import pandas as pd
import pytest
from src.load_data import load_data


def test_load_data_raises_filenotfounderror_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("Data/this_file_does_not_exist.csv")


def test_load_data_loads_small_file(tmp_path):
    data_path = tmp_path / "tiny.csv"
    data_path.write_text("A,B\n1,2\n3,4\n", encoding="utf-8")

    df = load_data(data_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["A", "B"]
