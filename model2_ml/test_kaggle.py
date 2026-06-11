import kagglehub
import pandas as pd
import os

# Download dataset
path = kagglehub.dataset_download(
    "mghobashy/drug-drug-interactions"
)

print("Dataset Path:")
print(path)

# Load CSV
csv_file = os.path.join(
    path,
    "db_drug_interactions.csv"
)

df = pd.read_csv(csv_file)

print("\n====================")
print("COLUMNS")
print("====================")
print(df.columns)

print("\n====================")
print("FIRST 5 ROWS")
print("====================")
print(df.head())

print("\n====================")
print("TOTAL ROWS")
print("====================")
print(len(df))