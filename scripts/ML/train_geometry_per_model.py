import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.prediction.geometry_schema import (
    DEFAULT_MODEL_PATH, FEATURES, MODES, NUMERIC_FEATURES, validate_frame,
)


def train_model(dataset_path, model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if model_path.name in {"best_per_predictor.joblib", "per_predictor.joblib"}:
        raise ValueError("Ce nom est réservé aux modèles précédents.")
    if model_path.exists():
        raise FileExistsError(f"Modèle déjà présent : {model_path}. Choisir un autre nom.")
    dataset = pd.read_csv(dataset_path)
    features = validate_frame(dataset)
    if set(features["mode"]) != set(MODES):
        raise ValueError("Le dataset doit contenir les huit modes.")
    if "per" not in dataset:
        raise ValueError("Colonne cible manquante : per")
    target = dataset["per"]
    if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
        raise ValueError("per doit être numérique.")
    if not np.isfinite(target.to_numpy()).all() or not target.between(0, 1).all():
        raise ValueError("per doit être fini et compris entre 0 et 1.")
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("categorical", OneHotEncoder(
                categories=[list(MODES)], handle_unknown="error", sparse_output=False,
            ), ["mode"]),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ])),
        ("model", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1,
        )),
    ])
    pipeline.fit(features, target)
    pipeline.geometry_metadata_ = {
        "feature_set": "A_Geometry",
        "features": list(FEATURES),
        "modes": list(MODES),
        "N_values": sorted(int(value) for value in features["N"].unique()),
        "geometry_ranges": {
            name: [float(features[name].min()), float(features[name].max())]
            for name in FEATURES[2:]
        },
        "training_rows": len(features),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("xb") as model_file:
        joblib.dump(pipeline, model_file)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Entraîner le PER A_Geometry sur tout le CSV.")
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data/processed/scenario_dataset.csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    arguments = parser.parse_args()
    pipeline = train_model(arguments.dataset, arguments.model)
    print(f"Pipeline sauvegardé : {arguments.model}")
    print(f"Lignes d'entraînement : {pipeline.geometry_metadata_['training_rows']}")
    print("Entraînement complet ; aucune métrique de validation calculée ici.")


if __name__ == "__main__":
    main()
