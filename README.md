```bash
cd /Users/darius/AI_AGENT_IOT
source .venv/bin/activate

open -a Ollama
ollama list

python app.py --no-llm --N 10000
python app.py --no-llm --N 50000
python app.py --no-llm --N 100000

python app.py --llm-model llama3.2:3b --N 10000 --require-llm --question "Explique simplement pourquoi ce mode est recommandé."

python app.py --llm-model llama3.2:3b --N 100000 --require-llm --question "Compare les PER de DR8 et DR9 et explique le choix."

python -m unittest discover -s tests -p 'test_*.py' -v

python scripts/analysis/audit_agent_dataset.py

# Étude fondée uniquement sur le CSV fourni. Choisir un nouveau dossier à chaque exécution.
python scripts/analysis/audit_raw_dataset.py --output-dir results/csv_raw_audit
python scripts/ML/evaluate_csv_agent.py --output-dir results/csv_agent_evaluation

python scripts/analysis/audit_raw_dataset.py --bin-size 1 --output-dir results/csv_raw_audit_1deg
python scripts/ML/evaluate_csv_agent.py --dataset results/csv_raw_audit_1deg/reaggregated_dataset.csv --output-dir results/csv_agent_evaluation_1deg

python app.py --llm-model llama3.2:3b --interactive
```
