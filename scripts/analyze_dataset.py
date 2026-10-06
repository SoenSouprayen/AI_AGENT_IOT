import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.getenv("DATASET_PATH")

columns = [
    "scheme",
    "mode",
    "N",
    "environment",
    "elevation_deg",
    "max_elevation_deg",
    "orbital_velocity_ms",
    "relative_velocity_ms",
    "snr_db",
    "rx_power_dbm",
    "doppler_shift_hz",
    "collision",
    "visibility_lost",
    "success",
    "replication",
    "scenario_seed",
]

schemes = set()
modes = set()
environments = set()
elevations = set()

total_packets = 0
success_packets = 0

for i, chunk in enumerate(
    pd.read_csv(
        CSV_PATH,
        usecols=columns,
        chunksize=100_000
    )
):
    total_packets += len(chunk)

    schemes.update(chunk["scheme"].dropna().unique())
    modes.update(chunk["mode"].dropna().unique())
    environments.update(chunk["environment"].dropna().unique())

    # arrondi uniquement pour l'affichage
    elevations.update(
        chunk["max_elevation_deg"]
        .dropna()
        .round(2)
        .unique()
    )

    success_packets += chunk["success"].fillna(False).astype(bool).sum()

    print(
        f"\r{i + 1} morceaux | "
        f"{total_packets:,} paquets analysés",
        end=""
    )

print("\n\n=== DATASET ===")
print(f"Paquets : {total_packets:,}")

print("\n=== SCHEMES ===")
print(sorted(schemes))

print("\n=== MODES ===")
print(sorted(modes))

print("\n=== ENVIRONNEMENTS ===")
print(sorted(environments))

print("\n=== MAX ELEVATIONS ===")
print(sorted(elevations))

global_per = 1 - (success_packets / total_packets)

print("\n=== PER GLOBAL ===")
print(global_per)