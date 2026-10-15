import pandas as pd

DATASET_PATH = "data/processed/scenario_dataset.csv"

df = pd.read_csv(DATASET_PATH)

KEYS = [
    "N",
    "replication",
    "scenario_seed",
    "elevation_bin"
]

# Vérification : combien de modulations sont présentes
# pour chaque condition ?
coverage = (
    df.groupby(KEYS)["mode"]
    .nunique()
)

print("=== COUVERTURE DES MODULATIONS ===")
print(coverage.value_counts().sort_index())

complete_keys = coverage[coverage == 8].index

complete = (
    df.set_index(KEYS)
    .loc[complete_keys]
    .reset_index()
)

print(
    "\nConditions avec les 8 modulations :",
    complete.groupby(KEYS).ngroups
)

# Mode avec le PER minimal
idx = complete.groupby(KEYS)["per"].idxmin()

best = complete.loc[
    idx,
    KEYS + ["mode", "per"]
].copy()

best = best.rename(
    columns={
        "mode": "best_mode",
        "per": "best_per"
    }
)

print("\n=== MODULATION OPTIMALE ===")

counts = best["best_mode"].value_counts()

print(counts)

print("\n=== POURCENTAGES ===")

print(
    (
        counts /
        len(best) *
        100
    ).round(2)
)

best.to_csv(
    "results/best_modulation_by_condition.csv",
    index=False
)
