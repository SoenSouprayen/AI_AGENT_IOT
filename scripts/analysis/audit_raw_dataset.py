import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.audit_agent_dataset import CONDITION_COLUMNS, audit_dataset


RAW_MEANS = {
    "elevation_deg": "elevation_mean_deg",
    "distance_km": "distance_mean_km",
    "orbital_velocity_ms": "orbital_velocity_mean_ms",
    "relative_velocity_ms": "relative_velocity_mean_ms",
    "relative_velocity_signed_ms": "relative_velocity_signed_mean_ms",
    "doppler_shift_hz": "doppler_shift_mean_hz",
    "doppler_rate_hz_s": "doppler_rate_mean_hz_s",
    "doppler_peak_rate_hz_s": "doppler_peak_rate_mean_hz_s",
    "rx_power_dbm": "rx_power_mean_dbm",
    "snr_db": "snr_mean_db",
    "visibility_window_s": "visibility_window_mean_s",
}
KEYS = [*CONDITION_COLUMNS, "mode"]


def aggregate_raw(path, chunksize=100000, bin_size=5):
    if chunksize <= 0:
        raise ValueError("chunksize doit être positif.")
    if bin_size not in (1, 5):
        raise ValueError("bin_size doit être 1 ou 5 degrés.")
    path = Path(path)
    before = path.stat()
    columns = [name for name in KEYS if name != "elevation_bin"]
    columns += [*RAW_MEANS, "success", "max_elevation_deg"]
    total = None
    ranges = {}
    rows = 0
    with pd.read_csv(path, usecols=columns, chunksize=chunksize) as chunks:
        for chunk in chunks:
            if chunk.isna().any().any():
                raise ValueError("Valeurs manquantes dans les colonnes auditées du CSV brut.")
            numeric = chunk[[*RAW_MEANS, "max_elevation_deg", "N", "replication", "scenario_seed"]]
            if not all(pd.api.types.is_numeric_dtype(chunk[name]) for name in numeric):
                raise ValueError("Les paramètres du CSV brut doivent être numériques.")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError("Paramètres non finis dans le CSV brut.")
            if not chunk["success"].isin([True, False, 0, 1]).all():
                raise ValueError("success doit contenir uniquement True/False ou 0/1.")
            if not chunk["elevation_deg"].between(0, 90).all():
                raise ValueError("Élévation brute hors de [0, 90].")
            for name in ("N", "replication", "scenario_seed"):
                if ((chunk[name] % 1 != 0) | (chunk[name] < 0)).any():
                    raise ValueError(f"{name} doit être un entier positif ou nul.")
            if (chunk["N"] == 0).any():
                raise ValueError("N doit être strictement positif.")
            for name in (*RAW_MEANS, "max_elevation_deg"):
                minimum, maximum = float(chunk[name].min()), float(chunk[name].max())
                previous = ranges.get(name, [minimum, maximum])
                ranges[name] = [min(previous[0], minimum), max(previous[1], maximum)]
            chunk["elevation_bin"] = (chunk["elevation_deg"] // bin_size * bin_size).clip(upper=90 - bin_size)
            chunk["n_packets"] = 1
            chunk["n_success"] = chunk["success"].astype(int)
            grouped = chunk.groupby(KEYS, dropna=False)[[*RAW_MEANS, "n_packets", "n_success"]].sum()
            total = grouped if total is None else total.add(grouped, fill_value=0)
            rows += len(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Le CSV brut a changé pendant sa lecture ; relancer après la copie.")
    if total is None or rows == 0:
        raise ValueError("CSV brut vide.")
    result = total.reset_index()
    for raw, mean in RAW_MEANS.items():
        result[mean] = result[raw] / result["n_packets"]
    result = result.drop(columns=list(RAW_MEANS))
    result[["n_packets", "n_success"]] = result[["n_packets", "n_success"]].astype("int64")
    result["per"] = 1 - result["n_success"] / result["n_packets"]
    audit_dataset(result)
    return result, {"raw_rows": rows, "raw_size_bytes": after.st_size,
                    "elevation_bin_size_deg": bin_size, "raw_ranges": ranges}


def compare_aggregation(rebuilt, existing):
    if existing.duplicated(KEYS).any():
        raise ValueError("Clés dupliquées dans le CSV agrégé existant.")
    shared = [name for name in (*RAW_MEANS.values(), "n_packets", "n_success", "per")
              if name in existing]
    merged = rebuilt.merge(existing[KEYS + shared], on=KEYS, how="outer",
                           suffixes=("_raw", "_existing"), indicator=True, validate="one_to_one")
    paired = merged[merged["_merge"] == "both"]
    columns = {}
    for name in shared:
        exact = name in ("n_packets", "n_success")
        actual = paired[f"{name}_raw"].to_numpy(dtype=float)
        expected = paired[f"{name}_existing"].to_numpy(dtype=float)
        close = np.isclose(actual, expected, rtol=0 if exact else 1e-10,
                           atol=0 if exact else 1e-10)
        columns[name] = {"mismatched_rows": int((~close).sum()),
                         "max_absolute_difference": float(np.max(np.abs(actual - expected))) if len(paired) else None}
    unmatched = int((merged["_merge"] != "both").sum())
    return {
        "paired_rows": len(paired), "unmatched_rows": unmatched, "columns": columns,
        "matches_existing": unmatched == 0 and all(item["mismatched_rows"] == 0 for item in columns.values()),
        "additional_metadata_preserved": [name for name in RAW_MEANS.values() if name not in existing],
    }


def main():
    parser = argparse.ArgumentParser(description="Relire le CSV brut par blocs et vérifier les agrégats sans les remplacer.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/scenario_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--bin-size", type=int, choices=(1, 5), default=5)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"Dossier déjà présent : {args.output_dir}")
        rebuilt, report = aggregate_raw(args.raw, args.chunksize, args.bin_size)
        report["raw_path"] = str(args.raw.resolve())
        report["aggregation_comparison"] = (
            compare_aggregation(rebuilt, pd.read_csv(args.processed)) if args.bin_size == 5
            else {"status": "not_comparable", "reason": "Tranches de 1 degré au lieu des 5 degrés du CSV existant."}
        )
        report["dataset_audit"] = audit_dataset(rebuilt)
        report["limitations"] = [
            "Lecture complète des colonnes sélectionnées, sans code du simulateur.",
            f"Regroupement par tranches de {args.bin_size} degré(s) ; la variabilité interne et les deux phases du passage restent moyennées.",
            "Des tranches plus fines contiennent moins de paquets ; un changement de gagnant observé ne prouve pas une adaptation robuste.",
            "Aucun nouveau scénario, entraînement ou résultat de généralisation produit ici.",
        ]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        rebuilt.to_csv(args.output_dir / "reaggregated_dataset.csv", index=False)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        (args.output_dir / "report.json").write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
