import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.audit_agent_dataset import CONDITION_COLUMNS, audit_dataset
from src.prediction.geometry_schema import FEATURES, GEOMETRY_FEATURES, MODES


COMMON_FEATURES = ("mode", "N", *GEOMETRY_FEATURES[:6])


def make_pipeline(features, trees):
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("mode", OneHotEncoder(categories=[list(MODES)], handle_unknown="error"), ["mode"]),
            ("numeric", "passthrough", [name for name in features if name != "mode"]),
        ])),
        ("model", RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                        random_state=42, n_jobs=-1)),
    ])


def decision_metrics(frame, predicted, fixed_modes=None):
    decisions = []
    for _, condition in frame.groupby(CONDITION_COLUMNS, sort=True):
        observed = condition.set_index("mode")["per"]
        if fixed_modes is None:
            scores = dict(zip(condition["mode"], predicted[condition.index]))
            selected = min(MODES, key=scores.get)
        else:
            selected = fixed_modes[condition.index[0]]
        minimum = float(observed.min())
        chosen = float(observed[selected])
        decisions.append((abs(chosen - minimum) <= 1e-12, max(0, chosen - minimum)))
    regrets = [regret for _, regret in decisions]
    return {"conditions": len(decisions),
            "decision_accuracy": float(np.mean([correct for correct, _ in decisions])),
            "mean_regret": float(np.mean(regrets)), "max_regret": float(max(regrets)),
            "p95_regret": float(np.quantile(regrets, 0.95))}


def evaluate_dataset(frame, folds=5, trees=400):
    frame = frame.reset_index(drop=True)
    audit_dataset(frame)
    if type(folds) is not int or not 2 <= folds <= frame["scenario_seed"].nunique():
        raise ValueError("Il faut entre 2 et autant de folds que de seeds distincts.")
    if type(trees) is not int or trees <= 0:
        raise ValueError("trees doit être un entier positif.")
    spreads = frame.groupby(CONDITION_COLUMNS)[list(GEOMETRY_FEATURES[:6])].agg(["min", "max"])
    if any(((spreads[name]["max"] - spreads[name]["min"]) > 1e-6).any()
           for name in GEOMETRY_FEATURES[:6]):
        raise ValueError("Les six features dites communes diffèrent entre modes ; audit requis.")
    feature_sets = {"RF_mode_N": ("mode", "N"), "RF_common_geometry": COMMON_FEATURES,
                    "RF_A_Geometry_diagnostic": FEATURES}
    predictions = {name: np.full(len(frame), np.nan) for name in feature_sets}
    rule_modes = np.full(len(frame), "", dtype=object)
    fold_ids = np.zeros(len(frame), dtype=int)
    splits = []
    for fold, (train_indices, test_indices) in enumerate(
        GroupKFold(n_splits=folds).split(frame, groups=frame["scenario_seed"]), start=1
    ):
        train, test = frame.iloc[train_indices], frame.iloc[test_indices]
        train_seeds = sorted(int(value) for value in train["scenario_seed"].unique())
        test_seeds = sorted(int(value) for value in test["scenario_seed"].unique())
        splits.append({"fold": fold, "train_seeds": train_seeds, "test_seeds": test_seeds})
        fold_ids[test_indices] = fold
        rule = {}
        for load, subset in train.groupby("N"):
            means = subset.groupby("mode")["per"].mean().to_dict()
            rule[load] = min(MODES, key=means.get)
        if not set(test["N"]).issubset(rule):
            raise ValueError("Un N du test est absent de l'entraînement ; règle N non définie.")
        rule_modes[test_indices] = test["N"].map(rule).to_numpy()
        for name, features in feature_sets.items():
            model = make_pipeline(features, trees)
            model.fit(train[list(features)], train["per"])
            predictions[name][test_indices] = model.predict(test[list(features)])
    metrics = {}
    for name, predicted in predictions.items():
        metrics[name] = {"per_mae": float(mean_absolute_error(frame["per"], predicted)),
                         "per_r2": float(r2_score(frame["per"], predicted)),
                         **decision_metrics(frame, predicted)}
    metrics["N_rule_learned_on_train"] = decision_metrics(frame, None, rule_modes)
    metrics["always_DR8"] = decision_metrics(frame, None, np.full(len(frame), "DR8", dtype=object))
    output = frame[[*CONDITION_COLUMNS, "mode", "per"]].copy()
    output["fold"] = fold_ids
    output["N_rule_mode"] = rule_modes
    for name, predicted in predictions.items():
        output[name] = predicted
    report = {
        "rows": len(frame), "folds": folds, "trees": trees, "metrics": metrics,
        "split_group": "scenario_seed (toutes les réplications, charges, modes et élévations ensemble)",
        "splits": splits,
        "feature_sets": {name: list(features) for name, features in feature_sets.items()},
        "limitations": [
            "Validation croisée de développement sur le CSV existant, pas un nouveau test final indépendant.",
            "Nouveaux seeds de scénarios existants ; aucune preuve de généralisation à toute géométrie.",
            "RF_A_Geometry_diagnostic utilise les features propres à chaque ligne candidate ; deux taux Doppler diffèrent entre modes.",
            "RF_common_geometry exclut ces deux taux ; les six autres features sont vérifiées communes à chaque condition.",
            "Les features agrégées doivent aussi être disponibles avant la décision pour une utilisation réelle.",
            "La règle N est apprise uniquement sur les données d'entraînement de chaque fold.",
            "Métriques non pondérées par le nombre de paquets ; ex aequo optimaux acceptés à 1e-12 près.",
            "Aucun modèle de production remplacé ; aucun ajustement d'hyperparamètres ou test LLM effectué.",
        ],
    }
    return report, output


def main():
    parser = argparse.ArgumentParser(description="Comparer les modèles sur des seeds séparés, uniquement depuis le CSV.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/scenario_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()
    try:
        if args.output_dir.exists():
            raise FileExistsError(f"Dossier déjà présent : {args.output_dir}")
        report, predictions = evaluate_dataset(pd.read_csv(args.dataset), args.folds, args.trees)
        report["dataset"] = str(args.dataset.resolve())
        args.output_dir.mkdir(parents=True, exist_ok=False)
        predictions.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        (args.output_dir / "report.json").write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
