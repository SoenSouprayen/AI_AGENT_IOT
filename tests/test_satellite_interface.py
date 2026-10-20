from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from interface import make_handler
from src.interface.playground import SatellitePlayground
from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES
from src.utils.ollama_client import OllamaConfig


class InterfacePredictor:
    metadata = {"training_rows": 48}

    def predict_modes(self, geometry, N, modes):
        return {mode: 0.1 + MODES.index(mode) / 10 for mode in modes}


class SatelliteInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        rows = []
        for load in (10000, 100000):
            for elevation in (5.5, 45.5, 89.5):
                for mode in MODES:
                    rows.append({
                        **dict(zip(GEOMETRY_FEATURES, [elevation, 800, 7600, 7000, -100, -200, -4, 5])),
                        "N": load, "mode": mode, "scenario_seed": 42,
                        "replication": 1, "elevation_bin": int(elevation),
                    })
        dataset = Path(self.directory.name) / "dataset.csv"
        pd.DataFrame(rows).to_csv(dataset, index=False)
        self.playground = SatellitePlayground(dataset, InterfacePredictor(), OllamaConfig(model="test"))
        self.payload = {"position_deg": 45, "N": 10000, "seed": 42}

    def test_nearest_actual_csv_row_and_mirror_position(self):
        geometry, load, source = self.playground.select_scenario(self.payload)
        mirrored, _, mirror_source = self.playground.select_scenario({**self.payload, "position_deg": 135})
        self.assertEqual(geometry, mirrored)
        self.assertEqual(source, mirror_source)
        self.assertEqual(geometry["elevation_mean_deg"], 45.5)
        self.assertEqual(source["requested_elevation_deg"], 45)
        self.assertEqual(load, 10000)

    def test_boundaries_use_nearest_observed_geometry(self):
        for position, expected in ((5, 5.5), (90, 89.5), (175, 5.5)):
            with self.subTest(position=position):
                geometry, _, _ = self.playground.select_scenario({**self.payload, "position_deg": position})
                self.assertEqual(geometry["elevation_mean_deg"], expected)

    def test_scenarios_are_associated_with_available_loads(self):
        self.playground.reference.loc[self.playground.reference["N"] == 100000, "scenario_seed"] = 43
        scenarios = self.playground.configuration()["scenarios"]
        self.assertEqual({item["seed"]: item["loads"] for item in scenarios}, {42: [10000], 43: [100000]})
        with self.assertRaises(ValueError):
            self.playground.select_scenario({**self.payload, "N": 100000})
        _, load, source = self.playground.select_scenario({**self.payload, "N": 100000, "seed": 43})
        self.assertEqual((load, source["seed"]), (100000, 43))

    def test_prediction_has_eight_modes_without_calling_ollama(self):
        with patch("src.agent.modulation_agent.create_chat_model") as llm:
            result = self.playground.predict(self.payload)
        llm.assert_not_called()
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertEqual(len(result["decision"]["per_by_mode"]), 8)
        self.assertEqual(result["llm_status"], "disabled")

    def test_invalid_requests_and_unknown_scenarios(self):
        invalid = [None, [], {}, {**self.payload, "N": True}, {**self.payload, "seed": "42"},
                   {**self.payload, "seed": 999}, {**self.payload, "N": 25000}]
        invalid += [{**self.payload, "position_deg": value} for value in (None, True, "45", -1, 180, float("nan"), float("inf"))]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.playground.select_scenario(payload)

    def test_explanation_lock_is_released_after_error(self):
        with patch("src.interface.playground.ModulationAgent") as agent:
            agent.return_value.recommend.side_effect = RuntimeError("failure")
            with self.assertRaises(RuntimeError):
                self.playground.explain(self.payload)
        self.assertFalse(self.playground.explanation_lock.locked())
        self.playground.explanation_lock.acquire()
        try:
            with self.assertRaises(BlockingIOError):
                self.playground.explain(self.payload)
        finally:
            self.playground.explanation_lock.release()

    def test_question_receives_current_scenario_and_preserves_decision(self):
        prompts = []

        def answer(prompt):
            prompts.append(prompt.to_messages())
            return AIMessage(content="Le mode DR8 minimise le PER prédit.")

        question = "Pourquoi DR8 plutôt que DR9 ?"
        with patch("src.agent.modulation_agent.create_chat_model", return_value=RunnableLambda(answer)):
            result = self.playground.ask({**self.payload, "question": question})
        self.assertEqual(result["llm_status"], "generated")
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertEqual(result["scenario"]["N"], 10000)
        self.assertEqual(result["input_source"]["used_elevation_deg"], 45.5)
        self.assertIn(question, prompts[0][1].content)
        self.assertIn('"DR9"', prompts[0][1].content)

    def test_invalid_questions_do_not_call_ollama(self):
        with patch("src.agent.modulation_agent.create_chat_model") as llm:
            for question in (None, False, 123, [], "", "   ", "x" * 501):
                with self.subTest(question=question), self.assertRaises(ValueError):
                    self.playground.ask({**self.payload, "question": question})
        llm.assert_not_called()

    def test_question_and_explanation_share_the_generation_lock(self):
        self.playground.explanation_lock.acquire()
        try:
            with self.assertRaises(BlockingIOError):
                self.playground.ask({**self.payload, "question": "Pourquoi ce mode ?"})
        finally:
            self.playground.explanation_lock.release()

    def test_question_http_handler_without_socket(self):
        def send(body):
            encoded = json.dumps(body).encode()
            request = b"POST /api/ask HTTP/1.1\r\nHost: localhost:8501\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(encoded)).encode() + b"\r\n\r\n" + encoded

            class MemoryHandler(make_handler(self.playground)):
                def setup(self):
                    self.rfile = BytesIO(request)
                    self.wfile = BytesIO()

                def finish(self):
                    pass

                def log_message(self, format, *args):
                    pass

            handler = MemoryHandler(None, ("127.0.0.1", 12345), SimpleNamespace(server_port=8501))
            headers, content = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
            return headers, json.loads(content)

        with patch("src.agent.modulation_agent.create_chat_model", return_value=RunnableLambda(
            lambda prompt: AIMessage(content="DR8 est recommandé dans ce scénario.")
        )):
            headers, answer = send({**self.payload, "question": "Pourquoi ce mode ?"})
        self.assertIn(b"200 OK", headers)
        self.assertEqual(answer["llm_status"], "generated")
        self.assertEqual(answer["decision"]["best_mode"], "DR8")
        headers, answer = send({**self.payload, "question": " "})
        self.assertIn(b"400 Bad Request", headers)
        self.assertIn("non vide", answer["error"])

    def test_http_static_predict_validation_and_origin(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.playground))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base + "/api/config", timeout=5) as response:
                self.assertEqual(json.load(response)["loads"], [10000, 100000])
            with urlopen(base + "/", timeout=5) as response:
                self.assertIn(b'id="satellite"', response.read())
                self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            request = Request(base + "/api/predict", data=json.dumps(self.payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)["decision"]["best_mode"], "DR8")
            with patch("src.agent.modulation_agent.create_chat_model", return_value=RunnableLambda(
                lambda prompt: AIMessage(content="Voici une explication du scénario.")
            )):
                question = Request(base + "/api/ask", data=json.dumps({**self.payload, "question": "Que signifie le PER ?"}).encode(),
                                   headers={"Content-Type": "application/json"})
                with urlopen(question, timeout=5) as response:
                    answer = json.load(response)
                self.assertEqual(answer["llm_status"], "generated")
                self.assertEqual(answer["decision"]["best_mode"], "DR8")
            with urlopen(base + "/questions.css", timeout=5) as response:
                self.assertIn(b"question-composer", response.read())
            for path, data, headers, expected in (
                ("/api/predict", b"broken", {"Content-Type": "application/json"}, 400),
                ("/api/predict", b"[]", {"Content-Type": "application/json"}, 400),
                ("/api/ask", json.dumps({**self.payload, "question": ""}).encode(), {"Content-Type": "application/json"}, 400),
                ("/api/ask", b"[]", {"Content-Type": "application/json"}, 400),
                ("/api/predict", b"{}", {"Content-Type": "text/plain"}, 400),
                ("/api/predict", b"{}", {"Content-Type": "application/json", "Origin": "https://untrusted.example"}, 403),
                ("/api/predict", b"{}", {"Content-Type": "application/json", "Host": "untrusted.example"}, 403),
                ("/missing", b"{}", {"Content-Type": "application/json"}, 404),
            ):
                with self.subTest(path=path, expected=expected), self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + path, data=data, headers=headers), timeout=5)
                self.assertEqual(error.exception.code, expected)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
