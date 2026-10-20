```bash
cd /Users/darius/AI_AGENT_IOT

# Lancement recommandé sur Mac : Ollama local + site, sans activer la venv.
# Ouvre Ollama si nécessaire, télécharge Llama uniquement s’il manque,
# précharge le modèle puis ouvre le navigateur. Ctrl+C arrête le site, pas Ollama.
# Ollama doit être installé ; les données et le modèle PER doivent déjà être présents.
bash run_project.sh

# Facultatif : choisir un autre port ou lancer sans ouvrir le navigateur.
bash run_project.sh --port 8502
bash run_project.sh --no-browser

# Installation des dépendances et lancement manuel, si nécessaire.
source .venv/bin/activate
python -m pip install -r requirements.txt

open -a Ollama
ollama list

# Interface CSV : satellite déplaçable, modèle à 1° et questions à Ollama.
python interface.py

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
