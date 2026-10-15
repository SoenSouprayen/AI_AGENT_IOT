from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models/per_model/per_predictor_A_Geometry.joblib"
MODES = ("DR8", "DR9", "SF7", "SF8", "SF9", "SF10", "SF11", "SF12")
GEOMETRY_FEATURES = (
    "elevation_mean_deg",
    "distance_mean_km",
    "orbital_velocity_mean_ms",
    "relative_velocity_mean_ms",
    "relative_velocity_signed_mean_ms",
    "doppler_shift_mean_hz",
    "doppler_rate_mean_hz_s",
    "doppler_peak_rate_mean_hz_s",
)
FEATURES = ("mode", "N", *GEOMETRY_FEATURES)
NUMERIC_FEATURES = FEATURES[1:]


def validate_frame(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if frame.empty:
        raise ValueError("Les données sont vides.")
    if not frame["mode"].isin(MODES).all():
        raise ValueError(f"mode doit appartenir à {MODES}.")
    for feature in NUMERIC_FEATURES:
        if not frame[feature].map(
            lambda value: isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"{feature} doit être numérique, sans booléen.")
    numeric = frame[list(NUMERIC_FEATURES)].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Les features doivent être finies, sans NaN ni infini.")
    if not ((numeric["N"] > 0) & (numeric["N"] % 1 == 0)).all():
        raise ValueError("N doit être un entier strictement positif.")
    if not numeric["elevation_mean_deg"].between(0, 90).all():
        raise ValueError("elevation_mean_deg doit être entre 0 et 90.")
    for feature in ("distance_mean_km", "orbital_velocity_mean_ms"):
        if not (numeric[feature] > 0).all():
            raise ValueError(f"{feature} doit être strictement positif.")
    for feature in ("relative_velocity_mean_ms", "doppler_peak_rate_mean_hz_s"):
        if not (numeric[feature] >= 0).all():
            raise ValueError(f"{feature} doit être positif ou nul.")
    return frame[list(FEATURES)].copy()


def make_frame(geometry, modes, N):
    missing = set(GEOMETRY_FEATURES) - set(geometry)
    if missing:
        raise ValueError(f"Features de géométrie manquantes : {sorted(missing)}")
    rows = [
        {"mode": mode, "N": N, **{name: geometry[name] for name in GEOMETRY_FEATURES}}
        for mode in modes
    ]
    return validate_frame(pd.DataFrame(rows))
