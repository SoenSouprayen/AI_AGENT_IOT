from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, StrictInt

from src.decision.modulation_selector import select_best_modulation


class SelectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometry: dict[str, Any]
    N: StrictInt


class PredictionInput(SelectionInput):
    mode: str


def build_tools(predictor):
    def predict_per(geometry, mode, N):
        return {"mode": mode, "predicted_per": predictor.predict_per(geometry, mode, N)}

    def select_modulation(geometry, N):
        return select_best_modulation(geometry, N, predictor)

    return {
        "predict_per": StructuredTool.from_function(
            func=predict_per,
            name="predict_per",
            description="Prédire le PER d'un mode à partir des huit features géométriques exactes et N.",
            args_schema=PredictionInput,
        ),
        "select_best_modulation": StructuredTool.from_function(
            func=select_modulation,
            name="select_best_modulation",
            description="Prédire les huit PER et sélectionner leur minimum avec le pipeline A_Geometry.",
            args_schema=SelectionInput,
        ),
    }
