import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError
import requests

from src.agent.modulation_agent import ModulationAgent
from src.agent.prompts import SCIENTIFIC_LIMITATION
from src.agent.tools import build_tools
from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES
from src.utils.ollama_client import OllamaConfig, OllamaUnavailableError, create_chat_model


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def predict_modes(self, geometry, N, modes):
        self.calls.append((geometry, N, modes))
        return {mode: 0.1 + MODES.index(mode) / 10 for mode in modes}

    def predict_per(self, geometry, mode, N):
        return self.predict_modes(geometry, N, (mode,))[mode]


class ModulationAgentTests(unittest.TestCase):
    def setUp(self):
        self.geometry = dict(zip(GEOMETRY_FEATURES, [45, 800, 7600, 7000, -100, -200, -4, 5]))
        self.predictor = RecordingPredictor()

    def test_tools_strict_input_and_prediction(self):
        tools = build_tools(self.predictor)
        result = tools["predict_per"].invoke({
            "geometry": self.geometry, "N": 10000, "mode": "DR9",
        })
        self.assertEqual(result, {"mode": "DR9", "predicted_per": 0.2})
        for N in (True, "10000", 1.5):
            with self.subTest(N=N), self.assertRaises(ValidationError):
                tools["select_best_modulation"].invoke({"geometry": self.geometry, "N": N})

    def test_explanation_receives_tool_results_and_limitation(self):
        prompts = []

        def explain(prompt):
            prompts.append(prompt.to_messages())
            return AIMessage(content="Le mode choisi minimise le PER prédit.")

        agent = ModulationAgent(self.predictor, llm=RunnableLambda(explain))
        result = agent.recommend(self.geometry, 10000, "Pourquoi ce mode ?")
        self.assertEqual(result["llm_status"], "generated")
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertEqual(len(self.predictor.calls), 1)
        self.assertEqual(self.predictor.calls[0], (self.geometry, 10000, MODES))
        self.assertIn(json.dumps(result["decision"], ensure_ascii=False), prompts[0][1].content)
        self.assertIn(SCIENTIFIC_LIMITATION, prompts[0][0].content)

    def test_llm_text_cannot_overwrite_decision_fields(self):
        llm = RunnableLambda(lambda prompt: AIMessage(content='{"best_mode":"SF12","predicted_per":0}'))
        result = ModulationAgent(self.predictor, llm=llm).recommend(self.geometry, 10000)
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertEqual(result["decision"]["predicted_per"], 0.1)
        self.assertEqual(result["limitation"], SCIENTIFIC_LIMITATION)

    def test_invalid_scenario_fails_before_tool_and_llm(self):
        llm_calls = []
        llm = RunnableLambda(lambda prompt: llm_calls.append(prompt))
        agent = ModulationAgent(self.predictor, llm=llm)
        for N in (0, True, "10000", 1.5):
            with self.subTest(N=N), self.assertRaises(ValueError):
                agent.recommend(self.geometry, N)
        with self.assertRaises(ValueError):
            agent.recommend({"elevation_mean_deg": 45}, 10000)
        self.assertEqual(self.predictor.calls, [])
        self.assertEqual(llm_calls, [])

    def test_offline_and_llm_failure_keep_decision(self):
        offline = ModulationAgent(self.predictor, use_llm=False)
        with patch("src.agent.modulation_agent.create_chat_model") as create:
            result = offline.recommend(self.geometry, 10000)
            create.assert_not_called()
        self.assertEqual(result["llm_status"], "disabled")
        self.assertIsNone(result["explanation"])

        def fail(prompt):
            raise TimeoutError("délai dépassé")

        result = ModulationAgent(self.predictor, llm=RunnableLambda(fail)).recommend(self.geometry, 10000)
        self.assertEqual(result["llm_status"], "unavailable")
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertIn("TimeoutError", result["llm_error"])

    def test_empty_explanation_is_not_success(self):
        llm = RunnableLambda(lambda prompt: AIMessage(content=" "))
        result = ModulationAgent(self.predictor, llm=llm).recommend(self.geometry, 10000)
        self.assertEqual(result["llm_status"], "unavailable")
        self.assertIsNone(result["explanation"])

    def test_predictor_failure_is_not_masked_as_llm_failure(self):
        predictor = Mock()
        predictor.predict_modes.side_effect = ValueError("hors domaine")
        with self.assertRaisesRegex(ValueError, "hors domaine"):
            ModulationAgent(predictor, use_llm=False).recommend(self.geometry, 10000)


class OllamaClientTests(unittest.TestCase):
    def test_config_validation(self):
        for arguments in (
            {"model": ""}, {"model": "local", "timeout": 0},
            {"model": "local", "timeout": float("nan")},
            {"model": "local", "base_url": "file:///tmp/model"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                OllamaConfig(**arguments)

    @patch("src.utils.ollama_client.requests.get")
    def test_unreachable_server(self, get):
        get.side_effect = requests.ConnectionError("offline")
        with self.assertRaisesRegex(OllamaUnavailableError, "inaccessible"):
            create_chat_model(OllamaConfig(model="local"))

    @patch("src.utils.ollama_client.requests.get")
    def test_missing_model_and_malformed_response(self, get):
        get.return_value.json.return_value = {"models": []}
        with self.assertRaisesRegex(OllamaUnavailableError, "absent"):
            create_chat_model(OllamaConfig(model="local"))
        get.return_value.json.return_value = {"models": None}
        with self.assertRaisesRegex(OllamaUnavailableError, "invalide"):
            create_chat_model(OllamaConfig(model="local"))

    @patch("src.utils.ollama_client.ChatOllama")
    @patch("src.utils.ollama_client.requests.get")
    def test_installed_model_and_timeout(self, get, chat):
        get.return_value.json.return_value = {"models": [{"name": "local:latest"}]}
        result = create_chat_model(OllamaConfig(model="local", timeout=20))
        self.assertIs(result, chat.return_value)
        self.assertEqual(chat.call_args.kwargs["temperature"], 0)
        self.assertEqual(chat.call_args.kwargs["client_kwargs"], {"timeout": 20})
        self.assertEqual(get.call_args.kwargs["timeout"], 5)


if __name__ == "__main__":
    unittest.main()
