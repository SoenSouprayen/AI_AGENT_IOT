from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import app
from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES


class AgentCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.scenario = Path(self.directory.name) / "scenario.json"
        geometry = dict(zip(GEOMETRY_FEATURES, [45, 800, 7600, 7000, -100, -200, -4, 5]))
        self.scenario.write_text(json.dumps({"geometry": geometry, "N": 10000}))
        self.predictor = Mock()
        self.predictor.predict_modes.return_value = {mode: 0.1 for mode in MODES}

    def test_offline_cli_has_only_agent_fields(self):
        output = io.StringIO()
        with patch.object(app, "PERPredictor", return_value=self.predictor), \
                patch("sys.argv", ["app.py", "--scenario", str(self.scenario), "--no-llm"]), \
                redirect_stdout(output):
            self.assertEqual(app.main(), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(set(result), {
            "scenario", "decision", "limitation", "llm_status", "explanation",
            "llm_error", "explanation_status", "input_source",
        })
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertEqual(result["llm_status"], "disabled")

    def test_invalid_N_is_rejected_before_prediction(self):
        errors = io.StringIO()
        with patch.object(app, "PERPredictor", return_value=self.predictor), \
                patch("sys.argv", ["app.py", "--scenario", str(self.scenario), "--N", "0", "--no-llm"]), \
                redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
            app.main()
        self.assertEqual(raised.exception.code, 1)
        self.predictor.predict_modes.assert_not_called()
        self.assertIn("N doit être un entier strictement positif", errors.getvalue())

    def test_required_llm_failure_keeps_numeric_decision(self):
        output = io.StringIO()
        with patch.object(app, "PERPredictor", return_value=self.predictor), \
                patch("src.agent.modulation_agent.create_chat_model", side_effect=RuntimeError("offline")), \
                patch("sys.argv", ["app.py", "--scenario", str(self.scenario),
                                   "--llm-model", "test-local", "--require-llm"]), redirect_stdout(output):
            self.assertEqual(app.main(), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result["decision"]["best_mode"], "DR8")
        self.assertEqual(result["llm_status"], "unavailable")
        self.assertIsNone(result["explanation"])


if __name__ == "__main__":
    unittest.main()
