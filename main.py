from src.load_data import load_data

df = load_data("Data/Huntington_Disease_Dataset.csv")
print(df.shape)
print(df.columns[:10].tolist())