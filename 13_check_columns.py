import pandas as pd


try:
    df = pd.read_csv('data/authorData.csv', nrows=0)
    print("Columns in authorData.csv:")
    print(df.columns.tolist())
    if any(c.startswith('V') for c in df.columns):
        print("FOUND V-COLUMNS!")
    else:
        print("NO V-COLUMNS found.")
except Exception as e:
    print(f"Error: {e}")
