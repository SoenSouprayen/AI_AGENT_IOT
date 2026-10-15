import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATASET = "data/processed/scenario_dataset.csv"
MODEL_PATH = "models/per_model/per_predictor.joblib"

df = pd.read_csv(DATASET)

# -----------------------------------
# Features
# -----------------------------------

FEATURES = [
    "mode",
    "N",

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
    "doppler_snr_penalty_mean_db"
]

TARGET = "per"

# -----------------------------------
# Split par scénario
# -----------------------------------

scenario_columns = [
    "scenario_seed",
    "replication"
]

scenarios = (
    df[scenario_columns]
    .drop_duplicates()
    .sample(frac=1, random_state=42)
)

split_index = int(len(scenarios) * 0.8)

train_scenarios = scenarios.iloc[:split_index]
test_scenarios = scenarios.iloc[split_index:]

train_keys = set(
    map(tuple, train_scenarios.values)
)

test_keys = set(
    map(tuple, test_scenarios.values)
)

row_keys = list(
    zip(
        df["scenario_seed"],
        df["replication"]
    )
)

train_mask = [
    key in train_keys
    for key in row_keys
]

test_mask = [
    key in test_keys
    for key in row_keys
]

train = df[train_mask].copy()
test = df[test_mask].copy()

print("=== SPLIT ===")
print("Train :", len(train))
print("Test  :", len(test))

print(
    "Scénarios train :",
    len(train_scenarios)
)

print(
    "Scénarios test  :",
    len(test_scenarios)
)

# -----------------------------------
# X / y
# -----------------------------------

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

categorical = ["mode"]

numeric = [
    feature
    for feature in FEATURES
    if feature not in categorical
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "mode",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical
        ),
        (
            "numeric",
            "passthrough",
            numeric
        )
    ]
)

# -----------------------------------
# Modèle
# -----------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42
            )
        )
    ]
)

# Pondération selon le nombre de paquets
weights = train["n_packets"]

model.fit(
    X_train,
    y_train,
    regressor__sample_weight=weights
)

# -----------------------------------
# Prediction
# -----------------------------------

pred = model.predict(X_test)

# Le PER doit rester entre 0 et 1
pred = np.clip(pred, 0, 1)

mae = mean_absolute_error(
    y_test,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

r2 = r2_score(
    y_test,
    pred
)

correlation = np.corrcoef(
    y_test,
    pred
)[0, 1]

print("\n=== PER PREDICTION ===")

print(f"MAE         : {mae:.6f}")
print(f"RMSE        : {rmse:.6f}")
print(f"R²          : {r2:.6f}")
print(f"Correlation : {correlation:.6f}")

# -----------------------------------
# Sauvegarde prédictions
# -----------------------------------

results = test[
    [
        "scenario_seed",
        "replication",
        "mode",
        "N",
        "elevation_bin",
        "per"
    ]
].copy()

results["predicted_per"] = pred

results["absolute_error"] = np.abs(
    results["per"] -
    results["predicted_per"]
)

os.makedirs(
    "results",
    exist_ok=True
)

results.to_csv(
    "results/per_predictions_test.csv",
    index=False
)

# -----------------------------------
# Sauvegarde modèle
# -----------------------------------

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_PATH
)

print(
    "\nModèle sauvegardé :",
    MODEL_PATH
)

print(
    "Prédictions :",
    "results/per_predictions_test.csv"
) 