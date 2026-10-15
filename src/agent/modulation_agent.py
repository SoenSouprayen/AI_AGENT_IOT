import json

from langchain_core.output_parsers import StrOutputParser

from src.agent.prompts import EXPLANATION_PROMPT, SCIENTIFIC_LIMITATION
from src.agent.tools import build_tools
from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES, make_frame
from src.prediction.per_predictor import PERPredictor
from src.utils.ollama_client import OllamaConfig, create_chat_model


class ModulationAgent:
    def __init__(self, predictor=None, llm=None, ollama_config=None, use_llm=True):
        self.predictor = predictor if predictor is not None else PERPredictor()
        self.tools = build_tools(self.predictor)
        self.llm = llm
        self.ollama_config = ollama_config
        self.use_llm = use_llm

    def recommend(self, geometry, N, question="Quelle modulation recommandes-tu et pourquoi ?"):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être un texte non vide.")
        frame = make_frame(geometry, MODES, N)
        geometry = {name: float(frame.iloc[0][name]) for name in GEOMETRY_FEATURES}
        N = int(frame.iloc[0]["N"])
        decision = self.tools["select_best_modulation"].invoke({"geometry": geometry, "N": N})
        result = {
            "scenario": {"geometry": geometry, "N": N},
            "decision": decision,
            "limitation": SCIENTIFIC_LIMITATION,
            "llm_status": "disabled",
            "explanation": None,
            "llm_error": None,
            "explanation_status": "not_requested",
        }
        if not self.use_llm:
            return result
        try:
            if self.llm is None:
                config = self.ollama_config or OllamaConfig.from_environment()
                self.llm = create_chat_model(config)
            chain = EXPLANATION_PROMPT | self.llm | StrOutputParser()
            explanation = chain.invoke({
                "question": question,
                "scenario": json.dumps(result["scenario"], ensure_ascii=False),
                "decision": json.dumps(decision, ensure_ascii=False),
                "limitation": SCIENTIFIC_LIMITATION,
            })
            if not explanation.strip():
                raise ValueError("Le LLM a retourné une explication vide.")
            result["explanation"] = explanation.strip()
            result["llm_status"] = "generated"
            result["explanation_status"] = "unverified_text"
        except Exception as error:
            result["llm_status"] = "unavailable"
            result["llm_error"] = f"{type(error).__name__}: {error}"
        return result
