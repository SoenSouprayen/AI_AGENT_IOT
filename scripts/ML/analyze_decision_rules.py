import pandas as pd
import numpy as np

PATH = "results/modulation_selection_results.csv"

df = pd.read_csv(PATH)

print("=" * 60)
print("DISTRIBUTION GLOBALE")
print("=" * 60)

print(
    df["actual_best_mode"]
    .value_counts()
)

# ============================================================
# PAR N
# ============================================================

print()
print("=" * 60)
print("MODE OPTIMAL PAR N")
print("=" * 60)

print(
    pd.crosstab(
        df["N"],
        df["actual_best_mode"],
        normalize="index"
    ).round(3)
)

print()
print("Comptages :")

print(
    pd.crosstab(
        df["N"],
        df["actual_best_mode"]
    )
)

# ============================================================
# PAR ELEVATION
# ============================================================

print()
print("=" * 60)
print("MODE OPTIMAL PAR ELEVATION")
print("=" * 60)

elevation_table = pd.crosstab(
    df["elevation_bin"],
    df["actual_best_mode"]
)

print(
    elevation_table
)

print()
print("Pourcentages :")

print(
    pd.crosstab(
        df["elevation_bin"],
        df["actual_best_mode"],
        normalize="index"
    ).round(3)
)

# ============================================================
# PAR N + ELEVATION
# ============================================================

print()
print("=" * 60)
print("MODE OPTIMAL PAR N + ELEVATION")
print("=" * 60)

table = pd.crosstab(
    [
        df["N"],
        df["elevation_bin"]
    ],
    df["actual_best_mode"]
)

print(
    table.to_string()
)

# ============================================================
# NOMBRE DE CHANGEMENTS DR8 / DR9
# ============================================================

print()
print("=" * 60)
print("CHANGEMENTS SELON ELEVATION")
print("=" * 60)

for N, group in df.groupby("N"):

    summary = (
        group.groupby(
            "elevation_bin"
        )["actual_best_mode"]
        .agg(
            lambda x:
            x.value_counts().index[0]
        )
    )

    print()
    print(f"N = {N}")

    for elevation, mode in summary.items():

        print(
            f"{elevation:>5}° -> {mode}"
        )

# ============================================================
# PURETE DES REGLES SIMPLES
# ============================================================

print()
print("=" * 60)
print("PURETE PAR N")
print("=" * 60)

for N, group in df.groupby("N"):

    counts = (
        group["actual_best_mode"]
        .value_counts()
    )

    majority = counts.index[0]

    accuracy = (
        group["actual_best_mode"]
        == majority
    ).mean()

    print(
        f"N={N} | "
        f"majoritaire={majority} | "
        f"accuracy={accuracy * 100:.2f}%"
    )