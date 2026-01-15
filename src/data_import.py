import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def load_data_c(file_path, columns_list):
    data = pd.read_csv(file_path)
    return data[columns_list]