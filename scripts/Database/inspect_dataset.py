import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.getenv("DATASET_PATH")

print(f"Lecture de : {CSV_PATH}")

df = pd.read_csv(
    CSV_PATH,
    nrows=10
)

print("\n=== DIMENSIONS DE L'ÉCHANTILLON ===")
print(df.shape)

print("\n=== COLONNES ===")
for column in df.columns:
    print(column)

print("\n=== TYPES ===")
print(df.dtypes)

print("\n=== 5 PREMIÈRES LIGNES ===")
print(df.head().to_string())