import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES, validate_frame


CONDITION_COLUMNS = ["N", "environment", "scenario_seed", "replication", "elevation_bin"]


def audit_dataset(frame):
    validate_frame(frame)
    required = {*CONDITION_COLUMNS, "per", "n_packets", "n_success"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Valeurs manquantes dans les conditions ou les résultats.")
    for name in ("per", "n_packets", "n_success"):
        values = pd.to_numeric(frame[name], errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name} doit être numérique et fini.")
    packets = frame["n_packets"].to_numpy(dtype=float)
    successes = frame["n_success"].to_numpy(dtype=float)
    per = frame["per"].to_numpy(dtype=float)
    if (packets <= 0).any() or (packets % 1 != 0).any():
        raise ValueError("n_packets doit contenir des entiers positifs.")
    if ((successes < 0) | (successes > packets) | (successes % 1 != 0)).any():
        raise ValueError("n_success incompatible avec n_packets.")
    if ((per < 0) | (per > 1)).any():
        raise ValueError("per doit être une fraction entre 0 et 1, pas un pourcentage.")
    if not np.allclose(per, 1 - successes / packets, rtol=0, atol=1e-10):
        raise ValueError("per ne correspond pas à 1 - n_success / n_packets.")
    if frame.duplicated([*CONDITION_COLUMNS, "mode"]).any():
        raise ValueError("Plusieurs lignes pour une même condition et un même mode.")
    groups = frame.groupby(CONDITION_COLUMNS, dropna=False, sort=True)
    incomplete = sum(set(group["mode"]) != set(MODES) for _, group in groups)
    if incomplete:
        raise ValueError(f"{incomplete} conditions ne contiennent pas les huit modes.")
    optimal_sets = []
    for condition, group in groups:
        minimum = group["per"].min()
        optimal = set(group.loc[np.isclose(group["per"], minimum, rtol=0, atol=1e-12), "mode"])
        optimal_sets.append((int(condition[0]), optimal))
    common_by_N = {}
    for load in sorted({load for load, _ in optimal_sets}):
        common = set(MODES)
        for condition_load, optimal in optimal_sets:
            if condition_load == load:
                common &= optimal
        common_by_N[str(load)] = [mode for mode in MODES if mode in common]
    spreads = groups[list(GEOMETRY_FEATURES)].max() - groups[list(GEOMETRY_FEATURES)].min()
    differences = {
        feature: {
            "conditions_different": int((spreads[feature] > 1e-6).sum()),
            "max_range": float(spreads[feature].max()),
        }
        for feature in GEOMETRY_FEATURES
    }
    return {
        "rows": len(frame),
        "conditions": len(optimal_sets),
        "environments": sorted(frame["environment"].astype(str).unique().tolist()),
        "N_values": sorted(int(value) for value in frame["N"].unique()),
        "seed_replication_groups": len(frame[["scenario_seed", "replication"]].drop_duplicates()),
        "conditions_with_optimal_ties": sum(len(optimal) > 1 for _, optimal in optimal_sets),
        "modes_optimal_for_every_condition_at_N": common_by_N,
        "N_only_rule_suffices_on_observed_data": all(bool(modes) for modes in common_by_N.values()),
        "geometry_spread_across_modes": differences,
        "packet_count_range": [int(packets.min()), int(packets.max())],
        "limitations": [
            "Audit descriptif du CSV agrégé, pas une validation hors échantillon.",
            "Les différences de géométrie entre modes doivent être expliquées par le simulateur.",
            "Une règle liée à N sur ces observations ne prouve pas une adaptation géométrique.",
            "Le CSV brut et le code du simulateur ne sont pas audités par ce script.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Audit descriptif, sans entraînement ni modification du dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/scenario_dataset.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_dataset(pd.read_csv(args.dataset))
        report["dataset"] = str(args.dataset.resolve())
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
        print(encoded)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
