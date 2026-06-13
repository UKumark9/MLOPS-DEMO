from sklearn.datasets import load_iris
import pandas as pd
import os

os.makedirs('data/raw', exist_ok=True)
Iris = load_iris(as_frame=True)
df = Iris.frame
df.to_csv('data/raw/iris.csv', index=False)
print(f"save iris data {df.shape[0]} to data/raw/iris.csv")
