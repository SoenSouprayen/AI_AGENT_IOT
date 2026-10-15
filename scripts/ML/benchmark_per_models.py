import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/processed/scenario_dataset.csv"
RESULTS_PATH = "results/model_comparison.csv"
MODEL_DIR = "models/per_model"

os.makedirs("results", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

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

CATEGORICAL_FEATURES = [
    "mode"
]

NUMERIC_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]

# ============================================================
# SPLIT PAR SCENARIO
# ============================================================

scenario_columns = [
    "scenario_seed",
    "replication"
]

scenarios = (
    df[scenario_columns]
    .drop_duplicates()
    .sample(
        frac=1,
        random_state=42
    )
    .reset_index(drop=True)
)

split_index = int(
    len(scenarios) * 0.8
)

train_scenarios = scenarios.iloc[
    :split_index
]

test_scenarios = scenarios.iloc[
    split_index:
]

train_keys = set(
    map(
        tuple,
        train_scenarios.values
    )
)

test_keys = set(
    map(
        tuple,
        test_scenarios.values
    )
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

print("========================================")
print("             DATASET")
print("========================================")

print(
    f"Train : {len(train)} lignes"
)

print(
    f"Test  : {len(test)} lignes"
)

print(
    f"Scénarios train : {len(train_scenarios)}"
)

print(
    f"Scénarios test  : {len(test_scenarios)}"
)

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        ),
        (
            "numeric",
            StandardScaler(),
            NUMERIC_FEATURES
        )
    ]
)

# ============================================================
# MODELES
# ============================================================

models = {

    "LinearRegression":
        LinearRegression(),

    "Ridge":
        Ridge(
            alpha=1.0
        ),

    "RandomForest":
        RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "ExtraTrees":
        ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "GradientBoosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

    "HistGradientBoosting":
        HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        )
}

# ============================================================
# BENCHMARK
# ============================================================

results = []

best_model = None
best_model_name = None
best_mae = float("inf")

for name, estimator in models.items():

    print()
    print("========================================")
    print(f"MODEL : {name}")
    print("========================================")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                estimator
            )
        ]
    )

    # ----------------------------------------
    # Training
    # ----------------------------------------

    start_train = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter()
        - start_train
    )

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    start_prediction = time.perf_counter()

    predictions = pipeline.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - start_prediction
    )

    # PER physique : [0, 1]

    predictions = np.clip(
        predictions,
        0,
        1
    )

    # ----------------------------------------
    # Metrics
    # ----------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    correlation = np.corrcoef(
        y_test,
        predictions
    )[0, 1]

    prediction_latency_ms = (
        prediction_time /
        len(X_test)
        * 1000
    )

    print(
        f"MAE         : {mae:.6f}"
    )

    print(
        f"RMSE        : {rmse:.6f}"
    )

    print(
        f"R²          : {r2:.6f}"
    )

    print(
        f"Correlation : {correlation:.6f}"
    )

    print(
        f"Training    : {training_time:.4f} s"
    )

    print(
        "Latency     : "
        f"{prediction_latency_ms:.6f} ms/sample"
    )

    results.append(
        {
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Correlation": correlation,
            "TrainingTime_s": training_time,
            "Latency_ms": prediction_latency_ms
        }
    )

    # ----------------------------------------
    # Best model
    # ----------------------------------------

    if mae < best_mae:

        best_mae = mae
        best_model = pipeline
        best_model_name = name

# ============================================================
# RESULTATS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "MAE"
)

print()
print("========================================")
print("       CLASSEMENT DES MODELES")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

# ============================================================
# SAUVEGARDE DU MEILLEUR MODELE
# ============================================================

best_path = os.path.join(
    MODEL_DIR,
    "best_per_predictor.joblib"
)

joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "features": FEATURES
    },
    best_path
)

print()
print("========================================")
print("            BEST MODEL")
print("========================================")

print(
    f"Model : {best_model_name}"
)

print(
    f"MAE   : {best_mae:.6f}"
)

print(
    f"Saved : {best_path}"
)

print(
    f"Results : {RESULTS_PATH}"
)