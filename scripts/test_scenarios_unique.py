import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

path = os.getenv("DATASET_PATH")

cols = [
    "scenario_seed",
    "replication",
    "scheme",
    "mode",
    "N"
]

scenarios = set()

for chunk in pd.read_csv(path, usecols=cols, chunksize=100_000):

    for row in chunk.drop_duplicates().itertuples(index=False, name=None):
        scenarios.add(row)

print("Nombre de scénarios uniques :", len(scenarios))
print("\nExemples :")

for s in list(scenarios)[:10]:
    print(s)