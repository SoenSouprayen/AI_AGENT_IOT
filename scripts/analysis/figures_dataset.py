import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "data/processed/scenario_dataset.csv"
FIGURES_PATH = "results/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("=== DATASET PROCESSED ===")
print(f"Lignes : {len(df):,}")
print(f"Colonnes : {len(df.columns)}")


print("\n=== VALEURS MANQUANTES ===")

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("Aucune valeur manquante.")
else:
    print(missing)

numeric_df = df.select_dtypes(include=[np.number])

inf_counts = pd.Series(
    {
        col: np.isinf(numeric_df[col]).sum()
        for col in numeric_df.columns
    }
)

inf_counts = inf_counts[inf_counts > 0]

print("\n=== VALEURS INFINIES ===")

if inf_counts.empty:
    print("Aucune valeur infinie.")
else:
    print(inf_counts)

print("\n=== PER ===")
print(df["per"].describe())



mode_stats = (
    df.groupby("mode")
    .agg(
        observations=("per", "size"),
        per_mean=("per", "mean"),
        per_median=("per", "median"),
        per_std=("per", "std"),
        per_min=("per", "min"),
        per_max=("per", "max"),
    )
    .sort_values("per_mean")
)

print("\n=== STATISTIQUES PAR MODE ===")
print(mode_stats)

mode_stats.to_csv(
    "results/per_statistics_by_mode.csv"
)


per_by_mode = (
    df.groupby("mode")["per"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(9, 5))

per_by_mode.plot(kind="bar")

plt.ylabel("PER moyen")
plt.xlabel("Modulation")
plt.title("PER moyen par modulation")

plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/per_by_mode.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10, 6))

for mode in sorted(df["mode"].unique()):

    subset = df[df["mode"] == mode]

    elevation_per = (
        subset.groupby("elevation_bin")["per"]
        .mean()
    )

    plt.plot(
        elevation_per.index,
        elevation_per.values,
        marker="o",
        label=mode
    )

plt.xlabel("Élévation (°)")
plt.ylabel("PER moyen")
plt.title("PER en fonction de l'élévation")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/per_vs_elevation.png",
    dpi=300
)

plt.close()




plt.figure(figsize=(10, 6))

for mode in sorted(df["mode"].unique()):

    subset = df[df["mode"] == mode]

    plt.scatter(
        subset["snr_mean_db"],
        subset["per"],
        s=15,
        alpha=0.5,
        label=mode
    )

plt.xlabel("SNR moyen (dB)")
plt.ylabel("PER")
plt.title("PER en fonction du SNR")
plt.legend()

plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/per_vs_snr.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10, 6))

for mode in sorted(df["mode"].unique()):

    subset = df[df["mode"] == mode]

    plt.scatter(
        subset["distance_mean_km"],
        subset["per"],
        s=15,
        alpha=0.5,
        label=mode
    )

plt.xlabel("Distance satellite-sol (km)")
plt.ylabel("PER")
plt.title("PER en fonction de la distance")

plt.legend()
plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/per_vs_distance.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(10, 6))

for mode in sorted(df["mode"].unique()):

    subset = df[df["mode"] == mode]

    plt.scatter(
        subset["doppler_shift_mean_hz"],
        subset["per"],
        s=15,
        alpha=0.5,
        label=mode
    )

plt.xlabel("Doppler moyen (Hz)")
plt.ylabel("PER")
plt.title("PER en fonction du Doppler")

plt.legend()
plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/per_vs_doppler.png",
    dpi=300
)

plt.close()



correlations = (
    numeric_df.corr()["per"]
    .drop("per")
    .sort_values()
)

print("\n=== CORRELATIONS AVEC PER ===")
print(correlations)

correlations.to_csv(
    "results/per_correlations.csv",
    header=["correlation_with_per"]
)

print("\nAnalyse terminée.")
print(f"Figures disponibles dans : {FIGURES_PATH}")
