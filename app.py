import argparse
import json
from pathlib import Path

import pandas as pd

from src.agent.modulation_agent import ModulationAgent
from src.prediction.geometry_schema import DEFAULT_MODEL_PATH, GEOMETRY_FEATURES, PROJECT_ROOT
from src.prediction.per_predictor import PERPredictor
from src.utils.ollama_client import OllamaConfig


def read_scenario(arguments):
    if arguments.scenario:
        with arguments.scenario.open(encoding="utf-8") as scenario_file:
            scenario = json.load(scenario_file)
        if not isinstance(scenario, dict) or not isinstance(scenario.get("geometry"), dict):
            raise ValueError("Le JSON doit contenir geometry (objet) et N (entier).")
        geometry = scenario["geometry"]
        N = scenario["N"]
        source = str(arguments.scenario)
    else:
        dataset = pd.read_csv(arguments.dataset)
        if not 0 <= arguments.row < len(dataset):
            raise ValueError("--row doit être un index de ligne valide (à partir de 0).")
        row = dataset.iloc[arguments.row]
        geometry = {name: row[name] for name in GEOMETRY_FEATURES}
        N = row["N"]
        source = f"CSV d'entraînement, ligne {arguments.row} ; démonstration fonctionnelle"
    if arguments.N is not None:
        N = arguments.N
    return geometry, N, source


def main():
    parser = argparse.ArgumentParser(description="Agent PER avec sélection Python et explication Ollama.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path,
                        default=PROJECT_ROOT / "data/processed/scenario_dataset.csv")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--scenario", type=Path, help="JSON contenant geometry et N.")
    parser.add_argument("--N", type=int)
    parser.add_argument("--question", default="Quelle modulation recommandes-tu et pourquoi ?")
    parser.add_argument("--llm-model", help="Nom exact d'un modèle déjà installé dans Ollama.")
    parser.add_argument("--ollama-url")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--require-llm", action="store_true", help="Code de sortie 2 si le LLM échoue.")
    parser.add_argument("--interactive", action="store_true", help="Questions successives sur le même scénario.")
    arguments = parser.parse_args()
    if arguments.no_llm and arguments.require_llm:
        parser.error("--no-llm et --require-llm sont incompatibles.")
    try:
        geometry, N, source = read_scenario(arguments)
        config = None
        if not arguments.no_llm and any(value is not None for value in (
            arguments.llm_model, arguments.ollama_url, arguments.timeout,
        )):
            config = OllamaConfig.from_environment(
                arguments.llm_model, arguments.ollama_url, arguments.timeout,
            )
        agent = ModulationAgent(
            predictor=PERPredictor(arguments.model),
            ollama_config=config,
            use_llm=not arguments.no_llm,
        )
        question = arguments.question
        while True:
            result = agent.recommend(geometry, N, question)
            result["input_source"] = source
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if arguments.require_llm and result["llm_status"] != "generated":
                return 2
            if not arguments.interactive:
                return 0
            try:
                question = input("\nQuestion (ou quit) : ").strip()
            except EOFError:
                return 0
            if question.lower() in {"quit", "exit", "quitter"}:
                return 0
            if not question:
                question = arguments.question
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
