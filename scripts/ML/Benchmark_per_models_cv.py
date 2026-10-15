import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
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
# DATASET
# ============================================================

DATASET_PATH = "data/processed/scenario_dataset.csv"
RESULTS_PATH = "results/model_comparison_cv.csv"

os.makedirs("results", exist_ok=True)

df = pd.read_csv(DATASET_PATH)

# ============================================================
# FEATURES
# ============================================================

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

CATEGORICAL = ["mode"]

NUMERIC = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL
]

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
print(f"Lignes           : {len(df)}")
print(f"Groupes uniques  : {groups.nunique()}")
print(f"Modulations      : {df['mode'].nunique()}")

# ============================================================
# MODELES
# ============================================================

models = {

    "LinearRegression":
        LinearRegression(),

    "Ridge":
        Ridge(alpha=1.0),

    "RandomForest":
        RandomForestRegressor(
            n_estimators=400,
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
# GROUP K-FOLD
# ============================================================

gkf = GroupKFold(n_splits=5)

all_results = []
all_fold_results = []

# ============================================================
# BOUCLE SUR LES MODELES
# ============================================================

for name, estimator in models.items():

    print()
    print("========================================")
    print(f"MODEL : {name}")
    print("========================================")

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(
            df[FEATURES],
            df[TARGET],
            groups=groups
        ),
        start=1
    ):

        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        # Vérification anti-leakage
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        assert train_groups.isdisjoint(test_groups)

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        # Nouveau preprocessing pour chaque fold
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    CATEGORICAL
                ),
                (
                    "numeric",
                    StandardScaler(),
                    NUMERIC
                )
            ]
        )

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

        # -----------------------------------
        # Training
        # -----------------------------------

        pipeline.fit(
            X_train,
            y_train
        )

        # -----------------------------------
        # Prediction
        # -----------------------------------

        pred = pipeline.predict(X_test)

        pred = np.clip(
            pred,
            0,
            1
        )

        # -----------------------------------
        # Metrics
        # -----------------------------------

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

        fold_result = {
            "model": name,
            "fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Correlation": corr,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups)
        }

        fold_results.append(
            fold_result
        )

        all_fold_results.append(
            fold_result
        )

        print(
            f"Fold {fold} | "
            f"MAE={mae:.6f} | "
            f"RMSE={rmse:.6f} | "
            f"R²={r2:.6f} | "
            f"Corr={corr:.6f}"
        )

    # ========================================================
    # MOYENNES DU MODELE
    # ========================================================

    fold_df = pd.DataFrame(
        fold_results
    )

    result = {

        "model": name,

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

    all_results.append(
        result
    )

    print()
    print(
        f"MAE  : "
        f"{result['MAE_mean']:.6f} "
        f"± {result['MAE_std']:.6f}"
    )

    print(
        f"RMSE : "
        f"{result['RMSE_mean']:.6f} "
        f"± {result['RMSE_std']:.6f}"
    )

    print(
        f"R²   : "
        f"{result['R2_mean']:.6f} "
        f"± {result['R2_std']:.6f}"
    )

    print(
        f"Corr : "
        f"{result['Correlation_mean']:.6f} "
        f"± {result['Correlation_std']:.6f}"
    )

# ============================================================
# CLASSEMENT FINAL
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    "MAE_mean"
)

print()
print("========================================")
print("       CLASSEMENT GROUP K-FOLD")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)

# ============================================================
# SAUVEGARDE
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

pd.DataFrame(
    all_fold_results
).to_csv(
    "results/model_comparison_cv_folds.csv",
    index=False
)

print()
print(
    "Résultats :",
    RESULTS_PATH
)