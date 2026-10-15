import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.decision.modulation_selector import select_best_modulation
from src.prediction.geometry_schema import DEFAULT_MODEL_PATH, GEOMETRY_FEATURES
from src.prediction.per_predictor import PERPredictor


def main():
    parser = argparse.ArgumentParser(description="Vérification fonctionnelle sur une ligne du CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data/processed/scenario_dataset.csv")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--N", type=int)
    arguments = parser.parse_args()
    dataset = pd.read_csv(arguments.dataset)
    if not 0 <= arguments.row < len(dataset):
        parser.error("--row doit être un index de ligne valide (à partir de 0).")
    row = dataset.iloc[arguments.row]
    geometry = {name: float(row[name]) for name in GEOMETRY_FEATURES}
    N = arguments.N if arguments.N is not None else row["N"]
    result = select_best_modulation(geometry, N, PERPredictor(arguments.model))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Test fonctionnel sur données d'entraînement ; aucune validation géométrique.")


if __name__ == "__main__":
    main()
