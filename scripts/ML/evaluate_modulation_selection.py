import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/processed/scenario_dataset.csv"

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# ============================================================
# DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

# ============================================================
# FEATURES
# Configuration A_Geometry
# ============================================================

FEATURES = [
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
]

TARGET = "per"

CATEGORICAL = [
    "mode"
]

NUMERIC = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL
]

# ============================================================
# CONDITION PHYSIQUE
# ============================================================

CONDITION_COLUMNS = [
    "N",
    "scenario_seed",
    "replication",
    "elevation_bin"
]

# ============================================================
# GROUPES POUR CROSS-VALIDATION
# ============================================================

groups = (
    df["scenario_seed"].astype(str)
    + "_"
    + df["replication"].astype(str)
)

print("========================================")
print("              DATASET")
print("========================================")

print(
    f"Lignes          : {len(df)}"
)

print(
    f"Groupes uniques : {groups.nunique()}"
)

print(
    f"Modulations     : {df['mode'].nunique()}"
)

print(
    f"Conditions      : "
    f"{df[CONDITION_COLUMNS].drop_duplicates().shape[0]}"
)

# ============================================================
# GROUP K-FOLD
# ============================================================

gkf = GroupKFold(
    n_splits=5
)

# Tableau qui contiendra toutes les prédictions
# out-of-fold
oof_predictions = []

# ============================================================
# CROSS VALIDATION
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(
        df[FEATURES],
        df[TARGET],
        groups=groups
    ),
    start=1
):

    print()
    print(
        f"========== FOLD {fold} =========="
    )

    train = df.iloc[
        train_idx
    ].copy()

    test = df.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------
    # Vérification anti-leakage
    # --------------------------------------------------------

    train_groups = set(
        groups.iloc[train_idx]
    )

    test_groups = set(
        groups.iloc[test_idx]
    )

    assert train_groups.isdisjoint(
        test_groups
    )

    print(
        f"Train : {len(train)} lignes"
    )

    print(
        f"Test  : {len(test)} lignes"
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train[
        FEATURES
    ]

    y_train = train[
        TARGET
    ]

    X_test = test[
        FEATURES
    ]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = pipeline.predict(
        X_test
    )

    predictions = np.clip(
        predictions,
        0,
        1
    )

    # --------------------------------------------------------
    # Sauvegarde OOF
    # --------------------------------------------------------

    fold_predictions = test[
        CONDITION_COLUMNS
        + [
            "scheme",
            "mode",
            "per"
        ]
    ].copy()

    fold_predictions[
        "predicted_per"
    ] = predictions

    fold_predictions[
        "fold"
    ] = fold

    oof_predictions.append(
        fold_predictions
    )

# ============================================================
# CONCATENATION DES PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    oof_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "modulation_oof_predictions.csv"
    ),
    index=False
)

print()
print("========================================")
print("      PREDICTIONS OOF TERMINEES")
print("========================================")

print(
    f"Lignes prédites : "
    f"{len(predictions_df)}"
)

# ============================================================
# VERIFICATION : 8 MODULATIONS PAR CONDITION
# ============================================================

coverage = (
    predictions_df
    .groupby(
        CONDITION_COLUMNS
    )["mode"]
    .nunique()
)

print()
print("========================================")
print("        COUVERTURE DES MODES")
print("========================================")

print(
    coverage.value_counts()
    .sort_index()
)

valid_conditions = (
    coverage[
        coverage == 8
    ]
    .index
)

print()
print(
    f"Conditions avec les 8 modes : "
    f"{len(valid_conditions)}"
)

# ============================================================
# MODE OPTIMAL REEL
# ============================================================

actual_best_idx = (
    predictions_df
    .groupby(
        CONDITION_COLUMNS
    )["per"]
    .idxmin()
)

actual_best = (
    predictions_df
    .loc[
        actual_best_idx,
        CONDITION_COLUMNS
        + [
            "mode",
            "per"
        ]
    ]
    .rename(
        columns={
            "mode":
                "actual_best_mode",

            "per":
                "actual_best_per"
        }
    )
)

# ============================================================
# MODE OPTIMAL PREDIT
# ============================================================

predicted_best_idx = (
    predictions_df
    .groupby(
        CONDITION_COLUMNS
    )["predicted_per"]
    .idxmin()
)

predicted_best = (
    predictions_df
    .loc[
        predicted_best_idx,
        CONDITION_COLUMNS
        + [
            "mode",
            "predicted_per"
        ]
    ]
    .rename(
        columns={
            "mode":
                "predicted_best_mode",

            "predicted_per":
                "predicted_best_per"
        }
    )
)

# ============================================================
# FUSION
# ============================================================

decisions = actual_best.merge(
    predicted_best,
    on=CONDITION_COLUMNS,
    how="inner"
)

# ============================================================
# PER REEL DU MODE CHOISI PAR LE MODELE
# ============================================================

chosen_real_per = (
    predictions_df[
        CONDITION_COLUMNS
        + [
            "mode",
            "per"
        ]
    ]
    .rename(
        columns={
            "mode":
                "predicted_best_mode",

            "per":
                "chosen_actual_per"
        }
    )
)

decisions = decisions.merge(
    chosen_real_per,
    on=(
        CONDITION_COLUMNS
        + [
            "predicted_best_mode"
        ]
    ),
    how="left"
)

# ============================================================
# DECISION CORRECTE ?
# ============================================================

decisions[
    "correct"
] = (
    decisions[
        "actual_best_mode"
    ]
    ==
    decisions[
        "predicted_best_mode"
    ]
)

# ============================================================
# REGRET
# ============================================================

decisions[
    "regret"
] = (
    decisions[
        "chosen_actual_per"
    ]
    -
    decisions[
        "actual_best_per"
    ]
)

# Petite protection numérique

decisions[
    "regret"
] = decisions[
    "regret"
].clip(
    lower=0
)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    decisions[
        "actual_best_mode"
    ],
    decisions[
        "predicted_best_mode"
    ]
)

# ============================================================
# BASELINE : TOUJOURS DR8
# ============================================================

baseline_accuracy = (
    decisions[
        "actual_best_mode"
    ]
    == "DR8"
).mean()

# ============================================================
# RESULTATS
# ============================================================

print()
print("========================================")
print("     SELECTION DE MODULATION")
print("========================================")

print(
    f"Conditions évaluées : "
    f"{len(decisions)}"
)

print()

print(
    f"Accuracy Random Forest : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Baseline Always DR8    : "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Gain                   : "
    f"{(accuracy - baseline_accuracy) * 100:.2f} points"
)

# ============================================================
# DISTRIBUTION DES MODES
# ============================================================

print()
print("========================================")
print("       MODES OPTIMAUX REELS")
print("========================================")

print(
    decisions[
        "actual_best_mode"
    ]
    .value_counts()
)

print()
print("========================================")
print("       MODES CHOISIS PAR RF")
print("========================================")

print(
    decisions[
        "predicted_best_mode"
    ]
    .value_counts()
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

all_modes = [
    "DR8",
    "DR9",
    "SF7",
    "SF8",
    "SF9",
    "SF10",
    "SF11",
    "SF12"
]

cm = confusion_matrix(
    decisions[
        "actual_best_mode"
    ],
    decisions[
        "predicted_best_mode"
    ],
    labels=all_modes
)

cm_df = pd.DataFrame(
    cm,
    index=all_modes,
    columns=all_modes
)

print()
print("========================================")
print("         CONFUSION MATRIX")
print("========================================")

print(
    cm_df
)

cm_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "modulation_confusion_matrix.csv"
    )
)

# ============================================================
# REGRET
# ============================================================

print()
print("========================================")
print("             PER REGRET")
print("========================================")

print(
    f"Mean   : "
    f"{decisions['regret'].mean():.6f}"
)

print(
    f"Median : "
    f"{decisions['regret'].median():.6f}"
)

print(
    f"P95    : "
    f"{decisions['regret'].quantile(0.95):.6f}"
)

print(
    f"Max    : "
    f"{decisions['regret'].max():.6f}"
)

print(
    f"Regret <= 0.01 : "
    f"{(decisions['regret'] <= 0.01).mean() * 100:.2f}%"
)

# ============================================================
# DR8 / DR9 UNIQUEMENT
# ============================================================

dr89 = decisions[
    decisions[
        "actual_best_mode"
    ].isin(
        [
            "DR8",
            "DR9"
        ]
    )
].copy()

dr89_accuracy = accuracy_score(
    dr89[
        "actual_best_mode"
    ],
    dr89[
        "predicted_best_mode"
    ]
)

print()
print("========================================")
print("        DECISION DR8 / DR9")
print("========================================")

print(
    f"Conditions : "
    f"{len(dr89)}"
)

print(
    f"Accuracy   : "
    f"{dr89_accuracy * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("========================================")
print("      CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        decisions[
            "actual_best_mode"
        ],
        decisions[
            "predicted_best_mode"
        ],
        zero_division=0
    )
)

# ============================================================
# SAUVEGARDE DES DECISIONS
# ============================================================

decisions.to_csv(
    os.path.join(
        RESULTS_DIR,
        "modulation_selection_results.csv"
    ),
    index=False
)

print()
print("========================================")
print("              FICHIERS")
print("========================================")

print(
    "results/modulation_oof_predictions.csv"
)

print(
    "results/modulation_selection_results.csv"
)

print(
    "results/modulation_confusion_matrix.csv"
)