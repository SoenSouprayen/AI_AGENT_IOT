import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.prediction.geometry_schema import DEFAULT_MODEL_PATH, FEATURES, MODES, make_frame


class PERPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, allow_out_of_domain=False):
        self.pipeline = joblib.load(model_path)
        self.allow_out_of_domain = allow_out_of_domain
        metadata = getattr(self.pipeline, "geometry_metadata_", {})
        if (
            not isinstance(self.pipeline, Pipeline)
            or metadata.get("feature_set") != "A_Geometry"
            or metadata.get("features") != list(FEATURES)
            or metadata.get("modes") != list(MODES)
            or list(getattr(self.pipeline, "feature_names_in_", [])) != list(FEATURES)
            or not isinstance(self.pipeline.named_steps.get("model"), RandomForestRegressor)
        ):
            raise ValueError("Pipeline incompatible : un modèle A_Geometry à 10 features est requis.")
        self.metadata = metadata

    def predict_modes(self, geometry, N, modes=MODES):
        frame = make_frame(geometry, modes, N)
        outside = []
        if N not in self.metadata["N_values"]:
            outside.append(f"N={N}, valeurs entraînées={self.metadata['N_values']}")
        for name, (minimum, maximum) in self.metadata["geometry_ranges"].items():
            if not minimum <= geometry[name] <= maximum:
                outside.append(f"{name} hors [{minimum}, {maximum}]")
        if outside:
            message = "Entrée hors domaine d'entraînement : " + "; ".join(outside)
            if not self.allow_out_of_domain:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        predictions = np.asarray(self.pipeline.predict(frame), dtype=float)
        if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
            raise ValueError("Prédictions PER invalides.")
        if ((predictions < 0) | (predictions > 1)).any():
            raise ValueError("Le pipeline a produit un PER hors [0, 1].")
        return dict(zip(frame["mode"], predictions.tolist()))

    def predict_per(self, geometry, mode, N):
        return self.predict_modes(geometry, N, (mode,))[mode]
