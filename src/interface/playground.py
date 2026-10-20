import math
from numbers import Real
from pathlib import Path
import threading

import pandas as pd

from src.agent.modulation_agent import ModulationAgent
from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES, validate_frame


class SatellitePlayground:
    def __init__(self, dataset_path, predictor, ollama_config):
        self.dataset_path = Path(dataset_path)
        self.dataset = pd.read_csv(self.dataset_path)
        validate_frame(self.dataset)
        required = {"scenario_seed", "replication", "elevation_bin"}
        if not required.issubset(self.dataset.columns):
            raise ValueError("Le CSV doit contenir scenario_seed, replication et elevation_bin.")
        if self.dataset[list(required)].isna().any().any():
            raise ValueError("Les identifiants de scénarios ne doivent pas être vides.")
        self.reference = self.dataset[self.dataset["mode"] == "DR8"].copy()
        if self.reference.empty or set(self.dataset["mode"]) != set(MODES):
            raise ValueError("Les huit modes sont requis dans le CSV.")
        self.predictor = predictor
        self.ollama_config = ollama_config
        self.prediction_lock = threading.Lock()
        self.explanation_lock = threading.Lock()

    def configuration(self):
        scenarios = self.reference[["scenario_seed", "replication"]].drop_duplicates()
        return {
            "loads": sorted(int(value) for value in self.reference["N"].unique()),
            "scenarios": [{"seed": int(row.scenario_seed), "replication": int(row.replication),
                           "loads": sorted(int(value) for value in self.reference.loc[
                               self.reference["scenario_seed"] == row.scenario_seed, "N"
                           ].unique())}
                          for row in scenarios.itertuples()],
            "dataset": self.dataset_path.name,
            "training_rows": self.predictor.metadata.get("training_rows"),
            "llm_model": self.ollama_config.model,
            "modes": list(MODES),
        }

    def select_scenario(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Une entrée JSON objet est requise.")
        if payload.get("source", "csv") != "csv":
            raise ValueError("Seuls les scénarios du CSV sont disponibles.")
        load, seed = payload.get("N"), payload.get("seed")
        position = payload.get("position_deg")
        if type(load) is not int or type(seed) is not int:
            raise ValueError("N et seed doivent être des entiers.")
        if not isinstance(position, Real) or isinstance(position, bool) or not math.isfinite(position):
            raise ValueError("La position doit être un nombre fini.")
        if not 5 <= position <= 175:
            raise ValueError("La position doit être comprise entre 5 et 175 degrés.")
        elevation = min(position, 180 - position)
        candidates = self.reference[(self.reference["N"] == load) & (self.reference["scenario_seed"] == seed)]
        if candidates.empty:
            raise ValueError("Combinaison N / scénario absente du CSV.")
        row = candidates.loc[(candidates["elevation_mean_deg"] - elevation).abs().idxmin()]
        geometry = {name: float(row[name]) for name in GEOMETRY_FEATURES}
        return geometry, load, {
            "requested_elevation_deg": float(elevation),
            "used_elevation_deg": geometry["elevation_mean_deg"],
            "elevation_bin": float(row["elevation_bin"]),
            "seed": seed,
            "replication": int(row["replication"]),
            "reference_mode": "DR8",
            "dataset": self.dataset_path.name,
            "note": "Ligne DR8 la plus proche : ses huit features sont réutilisées pour tous les modes, comme dans l'agent CLI.",
        }

    def predict(self, payload):
        geometry, load, source = self.select_scenario(payload)
        with self.prediction_lock:
            result = ModulationAgent(predictor=self.predictor, use_llm=False).recommend(geometry, load)
        result["input_source"] = source
        return result

    def explain(self, payload):
        return self._generate_response(
            payload,
            "Explique simplement le mode recommandé, le sens du PER et les limites de cette démonstration issue du CSV. "
            "Le dessin du satellite est illustratif et ne constitue pas une simulation orbitale.",
        )

    def ask(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Une entrée JSON objet est requise.")
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être un texte non vide.")
        if len(question) > 500:
            raise ValueError("La question doit contenir au maximum 500 caractères.")
        return self._generate_response(payload, question.strip())

    def _generate_response(self, payload, question):
        geometry, load, source = self.select_scenario(payload)
        if not self.explanation_lock.acquire(blocking=False):
            raise BlockingIOError("Une explication est déjà en cours. Réessaie dans un instant.")
        try:
            agent = ModulationAgent(predictor=self.predictor, ollama_config=self.ollama_config)
            result = agent.recommend(geometry, load, question)
            result["input_source"] = source
            return result
        finally:
            self.explanation_lock.release()
