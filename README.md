```bash
cd /Users/darius/AI_AGENT_IOT

# Docker Desktop doit être démarré. Lance l’interface + Ollama + Llama.
# Premier lancement : images Docker + modèle Llama (environ 2 Go pour le modèle seul).
# Prévoir environ 10 Go libres ; plusieurs minutes de téléchargement sont possibles.
docker compose up --build -d
open http://localhost:8501

# Voir les services et les messages de démarrage.
docker compose ps -a
docker compose logs -f ollama-init agent

# Arrêter les conteneurs, sans supprimer le modèle Llama téléchargé.
docker compose down

# Si le port 8501 est déjà utilisé par l’interface Python locale.
AGENT_PORT=8502 docker compose up --build -d
open http://localhost:8502

# Sur Mac, Ollama en conteneur peut être plus lent que l’application native.
# Les commandes ci-dessous restent disponibles pour le lancement sans Docker.
source .venv/bin/activate

# Interface graphique : satellite déplaçable, modèle à 1° et explications Ollama.
python interface.py

open -a Ollama
ollama list

python app.py --no-llm --N 10000
python app.py --no-llm --N 50000
python app.py --no-llm --N 100000

python app.py --llm-model llama3.2:3b --N 10000 --require-llm --question "Explique simplement pourquoi ce mode est recommandé."

python app.py --llm-model llama3.2:3b --N 100000 --require-llm --question "Compare les PER de DR8 et DR9 et explique le choix."

python -m unittest discover -s tests -p 'test_*.py' -v

python scripts/analysis/audit_agent_dataset.py

# Génère un CSV distinct à 1° ; conserve le CSV à 5° et le modèle actuel.
python scripts/Database/prepare_dataset.py --input data/raw/packet_level_dataset_with_geometry_doppler_rp002_1_0_5.csv
python scripts/analysis/audit_agent_dataset.py --dataset data/processed/scenario_dataset_1deg.csv

# Entraînement à 1° dans un nouveau fichier, sans écrasement des modèles existants.
python scripts/ML/train_geometry_per_model.py --dataset data/processed/scenario_dataset_1deg.csv --model models/per_model/per_predictor_A_Geometry_1deg.joblib

# Utiliser explicitement le modèle et le dataset à 1°.
python app.py --model models/per_model/per_predictor_A_Geometry_1deg.joblib --dataset data/processed/scenario_dataset_1deg.csv --N 10000 --no-llm
python app.py --model models/per_model/per_predictor_A_Geometry_1deg.joblib --dataset data/processed/scenario_dataset_1deg.csv --llm-model llama3.2:3b --require-llm --interactive

# Étude fondée uniquement sur le CSV fourni. Choisir un nouveau dossier à chaque exécution.
python scripts/analysis/audit_raw_dataset.py --output-dir results/csv_raw_audit
python scripts/ML/evaluate_csv_agent.py --output-dir results/csv_agent_evaluation

python scripts/analysis/audit_raw_dataset.py --bin-size 1 --output-dir results/csv_raw_audit_1deg
python scripts/ML/evaluate_csv_agent.py --dataset results/csv_raw_audit_1deg/reaggregated_dataset.csv --output-dir results/csv_agent_evaluation_1deg

python app.py --llm-model llama3.2:3b --interactive
```
