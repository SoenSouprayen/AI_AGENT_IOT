import argparse
import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.geometry_schema import MODES


ELEVATIONS = (5, 15, 30, 45, 60)
VELOCITIES_MS = (0, 7800, 15000)


def make_plan(loads, repetitions=9, holdout_groups=50, seed=42, velocity_kind="unspecified"):
    if not loads or any(type(load) is not int or load <= 0 for load in loads):
        raise ValueError("N doit contenir des entiers strictement positifs.")
    if len(set(loads)) != len(loads):
        raise ValueError("Les valeurs de N doivent être distinctes.")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("Le nombre de répétitions doit être un entier positif.")
    if velocity_kind not in ("radial_signed", "speed_norm"):
        raise ValueError("Préciser velocity_kind : radial_signed ou speed_norm.")
    contexts = [(elevation, velocity, repetition)
                for elevation in ELEVATIONS
                for velocity in VELOCITIES_MS
                for repetition in range(1, repetitions + 1)]
    if type(holdout_groups) is not int or not 0 < holdout_groups < len(contexts):
        raise ValueError("Réserver au moins un groupe test et un groupe de développement.")
    generator = random.Random(seed)
    holdout = set(generator.sample(range(len(contexts)), holdout_groups))
    simulation_seeds = generator.sample(range(1, 2**32), len(contexts))
    rows = []
    for index, (elevation, velocity, repetition) in enumerate(contexts):
        scenario_id = f"grid-{index + 1:04d}"
        for load in loads:
            for mode in MODES:
                rows.append({
                    "scenario_id": scenario_id,
                    "condition_id": f"{scenario_id}-N{load}",
                    "split": "test" if index in holdout else "development",
                    "scenario_seed": simulation_seeds[index],
                    "replication": repetition,
                    "N": load,
                    "elevation_deg": elevation,
                    "velocity_ms": velocity,
                    "velocity_kind": velocity_kind,
                    "mode": mode,
                    "status": "planned_not_simulated",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Crée un plan uniquement ; aucun PER ni simulation n'est produit.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--N", type=int, nargs="+", default=[10000, 50000, 100000])
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--holdout-groups", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--velocity-kind", choices=("radial_signed", "speed_norm"), required=True)
    args = parser.parse_args()
    try:
        rows = make_plan(args.N, args.repetitions, args.holdout_groups, args.seed, args.velocity_kind)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "simulation_plan.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "status": "plan_only_not_simulated",
            "runs": len(rows),
            "physical_grid_groups": len(rows) // (len(args.N) * len(MODES)),
            "test_groups": args.holdout_groups,
            "test_conditions_including_N": args.holdout_groups * len(args.N),
            "seed": args.seed,
            "velocity_kind": args.velocity_kind,
            "split_scope": "Nouvelles répétitions de la même grille ; pas une preuve de généralisation à des géométries inédites.",
            "required_result_columns": [
                "condition_id", "mode", "n_packets", "n_success", "per",
                "rx_power_dbm", "snr_db", "visibility_window_s",
                "distance_km", "doppler_shift_hz", "doppler_rate_hz_s",
            ],
            "constraints": [
                "Ce plan contrôlé ne représente pas des trajectoires TLE déjà validées physiquement.",
                "Confirmer la définition de vitesse avec le simulateur ; aucune conversion Doppler implicite.",
                "Configurer fréquence, BW, CR, payload, puissance, bruit et environnement dans le simulateur.",
                "Un seed identique ne garantit pas à lui seul les mêmes réalisations entre modes.",
                "Conserver le même contexte exogène pour les huit modes et toutes les valeurs de N du groupe.",
                "Réserver les groupes test avant tout entraînement ou ajustement des prompts.",
                f"Les {args.holdout_groups} groupes test couvrent {args.holdout_groups * len(args.N)} conditions en incluant N.",
                "per = 1 - n_success/n_packets, fraction entre 0 et 1 ; conserver les comptes.",
                "Ne pas concaténer directement ce plan au CSV A_Geometry : il ne contient aucun résultat.",
                "Prévoir un adaptateur du simulateur et un export vers les colonnes A_Geometry existantes.",
            ],
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
