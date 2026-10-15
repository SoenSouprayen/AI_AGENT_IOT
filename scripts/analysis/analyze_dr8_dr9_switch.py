import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "data/processed/scenario_dataset.csv"
OUTPUT = "results/dr8_dr9_comparison.csv"

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv(INPUT)

# On ne garde que DR8 / DR9
df = df[df["mode"].isin(["DR8", "DR9"])].copy()

KEYS = [
    "N",
    "replication",
    "scenario_seed",
    "elevation_bin"
]

# -----------------------------------------
# PER DR8 vs DR9 sur chaque condition
# -----------------------------------------

comparison = (
    df.pivot_table(
        index=KEYS,
        columns="mode",
        values="per"
    )
    .dropna()
    .reset_index()
)

comparison["per_difference"] = (
    comparison["DR9"] - comparison["DR8"]
)

comparison["best_mode"] = comparison.apply(
    lambda row:
        "DR8"
        if row["DR8"] <= row["DR9"]
        else "DR9",
    axis=1
)

comparison.to_csv(OUTPUT, index=False)

print("=== COMPARAISON DR8 / DR9 ===")

print(
    comparison["best_mode"]
    .value_counts()
)

print("\n=== PAR CHARGE N ===")

table_n = pd.crosstab(
    comparison["N"],
    comparison["best_mode"],
    normalize="index"
) * 100

print(table_n.round(2))

print("\n=== PAR ELEVATION ===")

table_elevation = pd.crosstab(
    comparison["elevation_bin"],
    comparison["best_mode"],
    normalize="index"
) * 100

print(table_elevation.round(2))

# -----------------------------------------
# Graphique PER DR8 / DR9 vs elevation
# -----------------------------------------

mean_per = (
    df.groupby(
        ["elevation_bin", "mode"]
    )["per"]
    .mean()
    .unstack()
)

plt.figure(figsize=(9, 5))

for mode in ["DR8", "DR9"]:
    plt.plot(
        mean_per.index,
        mean_per[mode],
        marker="o",
        label=mode
    )

plt.xlabel("Élévation (°)")
plt.ylabel("PER moyen")
plt.title("Comparaison DR8 / DR9 selon l'élévation")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/figures/dr8_dr9_vs_elevation.png",
    dpi=300
)

plt.close()

# -----------------------------------------
# Différence DR9 - DR8
# -----------------------------------------

difference = (
    comparison
    .groupby("elevation_bin")["per_difference"]
    .mean()
)

plt.figure(figsize=(9, 5))

plt.plot(
    difference.index,
    difference.values,
    marker="o"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Élévation (°)")
plt.ylabel("PER(DR9) - PER(DR8)")
plt.title("Frontière de décision DR8 / DR9")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/figures/dr8_dr9_difference.png",
    dpi=300
)

plt.close()

print("\nFichier :", OUTPUT)
print("Figures générées dans results/figures/")
