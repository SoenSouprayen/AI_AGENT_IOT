import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

INPUT_PATH = os.getenv("DATASET_PATH")
OUTPUT_PATH = "data/processed/scenario_dataset.csv"

CHUNK_SIZE = 100_000
ELEVATION_BIN_SIZE = 5

USECOLS = [
    "scheme",
    "mode",
    "N",
    "environment",
    "replication",
    "scenario_seed",

    "elevation_deg",
    "distance_km",
    "visibility_window_s",

    "orbital_velocity_ms",
    "relative_velocity_ms",
    "relative_velocity_signed_ms",

    "fspl_db",
    "total_path_loss_db",
    "effective_path_loss_db",

    "rx_power_dbm",
    "snr_db",

    "doppler_shift_hz",
    "doppler_rate_hz_s",
    "doppler_peak_rate_hz_s",
    "doppler_snr_penalty_db",

    "collision",
    "doppler_lost",
    "visibility_lost",
    "success"
]

GROUP_COLUMNS = [
    "scheme",
    "mode",
    "N",
    "environment",
    "replication",
    "scenario_seed",
    "elevation_bin"
]

aggregated_chunks = []

print("=== PREPARATION DU DATASET ===")
print("Source :", INPUT_PATH)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_PATH,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    # -----------------------------
    # Elevation bins de 5 degrés
    # -----------------------------

    chunk["elevation_bin"] = (
        chunk["elevation_deg"] // ELEVATION_BIN_SIZE
    ) * ELEVATION_BIN_SIZE

    # Évite éventuellement un bin 90-95
    chunk.loc[
        chunk["elevation_bin"] >= 90,
        "elevation_bin"
    ] = 85

    # -----------------------------
    # Booléens -> entiers
    # -----------------------------

    for column in [
        "success",
        "collision",
        "doppler_lost",
        "visibility_lost"
    ]:
        chunk[column] = (
            chunk[column]
            .fillna(False)
            .astype(bool)
            .astype(int)
        )

    # -----------------------------
    # Agrégation du chunk
    # -----------------------------

    agg = (
        chunk
        .groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            n_packets=("success", "size"),
            n_success=("success", "sum"),

            elevation_sum=("elevation_deg", "sum"),
            distance_sum=("distance_km", "sum"),

            orbital_velocity_sum=(
                "orbital_velocity_ms", "sum"
            ),

            relative_velocity_sum=(
                "relative_velocity_ms", "sum"
            ),

            relative_velocity_signed_sum=(
                "relative_velocity_signed_ms", "sum"
            ),

            fspl_sum=("fspl_db", "sum"),

            total_path_loss_sum=(
                "total_path_loss_db", "sum"
            ),

            effective_path_loss_sum=(
                "effective_path_loss_db", "sum"
            ),

            rx_power_sum=("rx_power_dbm", "sum"),
            snr_sum=("snr_db", "sum"),

            doppler_shift_sum=(
                "doppler_shift_hz", "sum"
            ),

            doppler_rate_sum=(
                "doppler_rate_hz_s", "sum"
            ),

            doppler_peak_rate_sum=(
                "doppler_peak_rate_hz_s", "sum"
            ),

            doppler_snr_penalty_sum=(
                "doppler_snr_penalty_db", "sum"
            ),

            collision_count=("collision", "sum"),
            doppler_lost_count=("doppler_lost", "sum"),
            visibility_lost_count=("visibility_lost", "sum")
        )
        .reset_index()
    )

    aggregated_chunks.append(agg)

    print(
        f"\rChunk {chunk_number} | "
        f"{chunk_number * CHUNK_SIZE:,} paquets traités",
        end=""
    )

print("\nFusion des agrégations...")

# Seulement les petites agrégations sont concaténées,
# pas les 6.4 millions de paquets.
temp = pd.concat(
    aggregated_chunks,
    ignore_index=True
)

# Un même groupe peut apparaître dans plusieurs chunks.
# On additionne donc les statistiques.
sum_columns = [
    column
    for column in temp.columns
    if column not in GROUP_COLUMNS
]

df = (
    temp
    .groupby(GROUP_COLUMNS, dropna=False)[sum_columns]
    .sum()
    .reset_index()
)

# --------------------------------------
# Calcul des moyennes
# --------------------------------------

df["elevation_mean_deg"] = (
    df["elevation_sum"] / df["n_packets"]
)

df["distance_mean_km"] = (
    df["distance_sum"] / df["n_packets"]
)

df["orbital_velocity_mean_ms"] = (
    df["orbital_velocity_sum"] / df["n_packets"]
)

df["relative_velocity_mean_ms"] = (
    df["relative_velocity_sum"] / df["n_packets"]
)

df["relative_velocity_signed_mean_ms"] = (
    df["relative_velocity_signed_sum"] /
    df["n_packets"]
)

df["fspl_mean_db"] = (
    df["fspl_sum"] / df["n_packets"]
)

df["total_path_loss_mean_db"] = (
    df["total_path_loss_sum"] /
    df["n_packets"]
)

df["effective_path_loss_mean_db"] = (
    df["effective_path_loss_sum"] /
    df["n_packets"]
)

df["rx_power_mean_dbm"] = (
    df["rx_power_sum"] / df["n_packets"]
)

df["snr_mean_db"] = (
    df["snr_sum"] / df["n_packets"]
)

df["doppler_shift_mean_hz"] = (
    df["doppler_shift_sum"] /
    df["n_packets"]
)

df["doppler_rate_mean_hz_s"] = (
    df["doppler_rate_sum"] /
    df["n_packets"]
)

df["doppler_peak_rate_mean_hz_s"] = (
    df["doppler_peak_rate_sum"] /
    df["n_packets"]
)

df["doppler_snr_penalty_mean_db"] = (
    df["doppler_snr_penalty_sum"] /
    df["n_packets"]
)

# --------------------------------------
# Résultats
# --------------------------------------

df["per"] = (
    1 -
    df["n_success"] /
    df["n_packets"]
)

df["collision_rate"] = (
    df["collision_count"] /
    df["n_packets"]
)

df["doppler_loss_rate"] = (
    df["doppler_lost_count"] /
    df["n_packets"]
)

df["visibility_loss_rate"] = (
    df["visibility_lost_count"] /
    df["n_packets"]
)

# --------------------------------------
# Colonnes finales
# --------------------------------------

FINAL_COLUMNS = [
    "scheme",
    "mode",
    "N",
    "environment",
    "replication",
    "scenario_seed",
    "elevation_bin",

    "elevation_mean_deg",
    "distance_mean_km",

    "orbital_velocity_mean_ms",
    "relative_velocity_mean_ms",
    "relative_velocity_signed_mean_ms",

    "fspl_mean_db",
    "total_path_loss_mean_db",
    "effective_path_loss_mean_db",

    "rx_power_mean_dbm",
    "snr_mean_db",

    "doppler_shift_mean_hz",
    "doppler_rate_mean_hz_s",
    "doppler_peak_rate_mean_hz_s",
    "doppler_snr_penalty_mean_db",

    "n_packets",
    "n_success",

    "collision_rate",
    "doppler_loss_rate",
    "visibility_loss_rate",

    "per"
]

df = df[FINAL_COLUMNS]

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n\n=== RESULTAT ===")
print("Lignes :", len(df))
print("Fichier :", OUTPUT_PATH)

print("\n=== MODES ===")
print(df["mode"].value_counts().sort_index())

print("\n=== PER PAR MODE ===")

print(
    df.groupby("mode")["per"]
    .agg(["mean", "median", "min", "max"])
    .sort_index()
)

print("\n=== APERCU ===")
print(df.head(10).to_string(index=False))