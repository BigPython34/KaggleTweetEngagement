import pandas as pd
import numpy as np


train = pd.read_csv('data/train.csv')
print("Engagement Stats:")
print(train['engagement'].describe())
print(f"Skewness: {train['engagement'].skew()}")
print(f"99th percentile: {train['engagement'].quantile(0.99)}")
print(f"Max: {train['engagement'].max()}")
