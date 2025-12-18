import pandas as pd

try:
    df = pd.read_csv('data/authorData.csv', nrows=5)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head())
except Exception as e:
    print(e)
