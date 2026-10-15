import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/processed/scenario_dataset.csv"
RESULTS_PATH = "results/feature_sets_comparison.csv"

os.makedirs("results", exist_ok=True)

df = pd.read_csv(DATASET_PATH)

# ============================================================
# FEATURE SETS A / B / C
# ============================================================

FEATURE_SETS = {

    # --------------------------------------------------------
    # A — GEOMETRY
    # --------------------------------------------------------

    "A_Geometry": [
        "mode",
        "N",

        "elevation_mean_deg",
        "distance_mean_km",

        "orbital_velocity_mean_ms",
        "relative_velocity_mean_ms",
        "relative_velocity_signed_mean_ms",

        "doppler_shift_mean_hz",
        "doppler_rate_mean_hz_s",
        "doppler_peak_rate_mean_hz_s"
    ],

    # --------------------------------------------------------
    # B — GEOMETRY + LINK BUDGET
    # --------------------------------------------------------

    "B_Geometry_LinkBudget": [
        "mode",
        "N",

        "elevation_mean_deg",
        "distance_mean_km",

        "orbital_velocity_mean_ms",
        "relative_velocity_mean_ms",
        "relative_velocity_signed_mean_ms",

        "doppler_shift_mean_hz",
        "doppler_rate_mean_hz_s",
        "doppler_peak_rate_mean_hz_s",

        "fspl_mean_db",
        "total_path_loss_mean_db"
    ],

    # --------------------------------------------------------
    # C — FULL CHANNEL
    # --------------------------------------------------------

    "C_Full_Channel": [
        "mode",
        "N",

        "elevation_mean_deg",
        "distance_mean_km",

        "orbital_velocity_mean_ms",
        "relative_velocity_mean_ms",
        "relative_velocity_signed_mean_ms",

        "doppler_shift_mean_hz",
        "doppler_rate_mean_hz_s",
        "doppler_peak_rate_mean_hz_s",

        "fspl_mean_db",
        "total_path_loss_mean_db",

        "effective_path_loss_mean_db",
        "rx_power_mean_dbm",
        "snr_mean_db",

        "doppler_snr_penalty_mean_db"
    ]
}

# ============================================================
# GROUPES
# ============================================================

groups = (
    df["scenario_seed"].astype(str)
    + "_"
    + df["replication"].astype(str)
)

print("========================================")
print("              DATASET")
print("========================================")

print(f"Lignes          : {len(df)}")
print(f"Groupes uniques : {groups.nunique()}")
print(f"Modulations     : {df['mode'].nunique()}")

# ============================================================
# GROUP K-FOLD
# ============================================================

gkf = GroupKFold(n_splits=5)

all_results = []
all_folds = []

# ============================================================
# TEST A / B / C
# ============================================================

for feature_set_name, features in FEATURE_SETS.items():

    print()
    print("========================================")
    print(feature_set_name)
    print("========================================")

    print(f"Nombre de features : {len(features)}")

    categorical = ["mode"]

    numeric = [
        feature
        for feature in features
        if feature not in categorical
    ]

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(
            df[features],
            df["per"],
            groups=groups
        ),
        start=1
    ):

        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        X_train = train[features]
        y_train = train["per"]

        X_test = test[features]
        y_test = test["per"]

        # ----------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    categorical
                ),
                (
                    "numeric",
                    StandardScaler(),
                    numeric
                )
            ]
        )

        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        model = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    model
                )
            ]
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        pred = pipeline.predict(
            X_test
        )

        pred = np.clip(
            pred,
            0,
            1
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

        corr = np.corrcoef(
            y_test,
            pred
        )[0, 1]

        result = {
            "feature_set": feature_set_name,
            "fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Correlation": corr
        }

        fold_results.append(result)
        all_folds.append(result)

        print(
            f"Fold {fold} | "
            f"MAE={mae:.6f} | "
            f"RMSE={rmse:.6f} | "
            f"R²={r2:.6f} | "
            f"Corr={corr:.6f}"
        )

    # ========================================================
    # MOYENNE DES 5 FOLDS
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    summary = {
        "feature_set": feature_set_name,

        "MAE_mean":
            fold_df["MAE"].mean(),

        "MAE_std":
            fold_df["MAE"].std(),

        "RMSE_mean":
            fold_df["RMSE"].mean(),

        "RMSE_std":
            fold_df["RMSE"].std(),

        "R2_mean":
            fold_df["R2"].mean(),

        "R2_std":
            fold_df["R2"].std(),

        "Correlation_mean":
            fold_df["Correlation"].mean(),

        "Correlation_std":
            fold_df["Correlation"].std()
    }

    all_results.append(summary)

# ============================================================
# RESULTATS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

print()
print("========================================")
print("     COMPARAISON A / B / C")
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

pd.DataFrame(
    all_folds
).to_csv(
    "results/feature_sets_folds.csv",
    index=False
)

print()
print("Résultats :", RESULTS_PATH)