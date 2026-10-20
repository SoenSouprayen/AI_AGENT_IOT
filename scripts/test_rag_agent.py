import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.rag.case_base import PERCaseBase, DEFAULT_RAG_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de récupération RAG PER.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data/processed/scenario_dataset.csv",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Erreur : fichier {args.dataset} introuvable.")
        sys.exit(1)

    # Récupération d'un exemple dans le CSV
    df = pd.read_csv(args.dataset)
    row = df.iloc[args.row]
    query_geometry = {feat: float(row[feat]) for feat in DEFAULT_RAG_FEATURES if feat in row}

    # Initialisation & requête
    rag = PERCaseBase(dataset_path=args.dataset)
    cases = rag.search_similar_cases(query_geometry=query_geometry, top_k=args.top_k)

    print("--- Résultats JSON ---")
    print(json.dumps(cases, indent=2, ensure_ascii=False))

    print("\n--- Contexte Prompt LLM ---")
    print(rag.format_context_for_llm(cases))


if __name__ == "__main__":
    main()
