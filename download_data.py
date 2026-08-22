import pandas as pd
import os

os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.to_csv("data/titanic.csv", index=False)

print(f"Downloaded {df.shape[0]} rows, {df.shape[1]} columns")
print("Saved to data/titanic.csv")