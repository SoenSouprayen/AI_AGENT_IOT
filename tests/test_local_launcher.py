import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from scripts import run_local


class LocalLauncherTests(unittest.TestCase):
    def test_project_keeps_only_local_launch_configuration(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("Dockerfile", "compose.yaml", ".dockerignore", "requirements-docker.txt", "tests/test_docker_setup.py"):
            self.assertFalse((root / name).exists(), name)
        self.assertTrue((root / "run_project.sh").is_file())
        self.assertIn("bash run_project.sh", (root / "README.md").read_text())
        self.assertNotIn("docker compose", (root / "README.md").read_text())

    def test_running_model_is_reused_and_preloaded(self):
        with patch.object(run_local, "local_models", return_value={"llama3.2:3b"}), \
                patch.object(run_local.subprocess, "run") as command, \
                patch.object(run_local, "ollama_request") as request:
            run_local.ensure_model("llama3.2:3b")
        command.assert_not_called()
        request.assert_called_once_with("/api/generate", {
            "model": "llama3.2:3b", "prompt": "", "stream": False, "keep_alive": "15m",
        }, timeout=180)

    def test_missing_model_is_pulled_only_from_local_server(self):
        with patch.object(run_local, "local_models", side_effect=[set(), {"llama3.2:3b"}]), \
                patch.object(run_local, "ollama_command", return_value="/bin/ollama"), \
                patch.object(run_local.os, "environ", {"OLLAMA_HOST": "http://remote.invalid"}), \
                patch.object(run_local.subprocess, "run") as command, \
                patch.object(run_local, "ollama_request"):
            run_local.ensure_model("llama3.2:3b")
        self.assertEqual(command.call_args.args[0], ["/bin/ollama", "pull", "llama3.2:3b"])
        self.assertEqual(command.call_args.kwargs["env"]["OLLAMA_HOST"], run_local.OLLAMA_URL)

    def test_mac_application_is_started_and_waited_for(self):
        with patch.object(run_local, "local_models", return_value=None), \
                patch.object(run_local.sys, "platform", "darwin"), \
                patch.object(run_local.subprocess, "run") as command, \
                patch.object(run_local, "wait_for_ollama", return_value={"llama3.2:3b"}) as wait, \
                patch.object(run_local, "ollama_request"):
            run_local.ensure_model("llama3.2:3b")
        command.assert_called_once_with(["open", "-a", "Ollama"], check=True)
        wait.assert_called_once_with()

    def test_server_timeout_is_bounded(self):
        with patch.object(run_local.time, "monotonic", side_effect=[0, 0, 61]), \
                patch.object(run_local.time, "sleep"), \
                patch.object(run_local, "local_models", return_value=None):
            with self.assertRaisesRegex(ValueError, "60 secondes"):
                run_local.wait_for_ollama()

    def test_download_error_prevents_preload(self):
        with patch.object(run_local, "local_models", return_value=set()), \
                patch.object(run_local, "ollama_command", return_value="ollama"), \
                patch.object(run_local.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "pull")), \
                patch.object(run_local, "ollama_request") as request:
            with self.assertRaises(subprocess.CalledProcessError):
                run_local.ensure_model("llama3.2:3b")
        request.assert_not_called()

    def test_local_api_errors_are_not_accepted_as_ready(self):
        for result in ({}, {"models": None}, {"models": [{}]}):
            with self.subTest(result=result), patch.object(run_local, "ollama_request", return_value=result):
                self.assertIsNone(run_local.local_models())
        with patch.object(run_local, "ollama_request", side_effect=URLError("offline")):
            self.assertIsNone(run_local.local_models())

    def test_api_requests_are_local_and_preload_has_no_prompt(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"done": true}'
        with patch.object(run_local.LOCAL_HTTP, "open", return_value=response) as opener:
            run_local.ollama_request("/api/generate", {"prompt": ""}, timeout=180)
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(json.loads(request.data), {"prompt": ""})

    def test_busy_default_port_uses_next_port_without_stopping_any_service(self):
        probe = MagicMock()
        probe.__enter__.return_value.bind.side_effect = [OSError("busy"), None]
        with patch.object(run_local.socket, "socket", return_value=probe):
            self.assertEqual(run_local.choose_port(), 8502)
        probe.__enter__.return_value.bind.assert_any_call(("127.0.0.1", 8502))
        probe.__enter__.return_value.bind.side_effect = OSError("busy")
        with patch.object(run_local.socket, "socket", return_value=probe), self.assertRaises(ValueError):
            run_local.choose_port(8501)

    def test_missing_data_does_not_trigger_ollama_download(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_local, "PROJECT_ROOT", Path(directory)), \
                patch.object(run_local, "ensure_model") as ensure:
            self.assertEqual(run_local.main([]), 1)
        ensure.assert_not_called()

    def test_interface_uses_current_python_and_explicit_local_url(self):
        with patch.object(run_local, "check_project"), \
                patch.object(run_local, "choose_port", return_value=8502), \
                patch.object(run_local, "ensure_model"), \
                patch.object(run_local.subprocess, "Popen") as launch:
            launch.return_value.wait.return_value = 0
            launch.return_value.poll.return_value = 0
            self.assertEqual(run_local.main(["--no-browser"]), 0)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], run_local.sys.executable)
        self.assertEqual(command[command.index("--ollama-url") + 1], run_local.OLLAMA_URL)
        self.assertEqual(command[command.index("--port") + 1], "8502")
        self.assertIn("--no-browser", command)

    def test_ctrl_c_stops_only_launched_interface(self):
        with patch.object(run_local, "check_project"), \
                patch.object(run_local, "choose_port", return_value=8501), \
                patch.object(run_local, "ensure_model"), \
                patch.object(run_local.subprocess, "Popen") as launch:
            launch.return_value.wait.side_effect = [KeyboardInterrupt(), 0]
            launch.return_value.poll.return_value = None
            self.assertEqual(run_local.main([]), 130)
        launch.return_value.terminate.assert_called_once_with()
        launch.return_value.kill.assert_not_called()

    def test_invalid_port_and_model_are_rejected_before_startup(self):
        with patch.object(run_local, "ensure_model") as ensure:
            for arguments in (["--port", "0"], ["--port", "65536"], ["--llm-model", " "]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    run_local.main(arguments)
        ensure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
