import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("docker"), "Docker CLI nécessaire pour valider Compose.")
class DockerSetupTests(unittest.TestCase):
    def configuration(self, **overrides):
        environment = os.environ.copy()
        environment.update({"AGENT_PORT": "8501", "LLM_MODEL": "llama3.2:3b", "OLLAMA_TIMEOUT": "180"})
        environment.update(overrides)
        result = subprocess.run(
            ["docker", "compose", "config", "--format", "json"],
            cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_services_wait_for_ollama_and_model(self):
        services = self.configuration()["services"]
        self.assertEqual(set(services), {"agent", "ollama", "ollama-init"})
        self.assertEqual(services["ollama-init"]["depends_on"]["ollama"]["condition"], "service_healthy")
        self.assertEqual(services["agent"]["depends_on"]["ollama-init"]["condition"], "service_completed_successfully")
        self.assertEqual(services["agent"]["environment"]["OLLAMA_BASE_URL"], "http://ollama:11434")
        self.assertIn("ollama_models", services["ollama"]["volumes"][0]["source"])

    def test_local_port_and_readonly_mounts(self):
        services = self.configuration()["services"]
        agent = services["agent"]
        self.assertEqual(agent["ports"][0]["host_ip"], "127.0.0.1")
        self.assertEqual(agent["ports"][0]["target"], 8501)
        self.assertNotIn("ports", services["ollama"])
        self.assertTrue(agent["read_only"])
        self.assertEqual(len(agent["volumes"]), 2)
        for volume in agent["volumes"]:
            self.assertTrue(volume["read_only"])
            self.assertFalse(volume.get("bind", {}).get("create_host_path", False))
            self.assertNotIn("/raw/", volume["source"])

    def test_custom_port_and_model_are_consistent(self):
        services = self.configuration(AGENT_PORT="8502", LLM_MODEL="llama3.2:1b")["services"]
        agent = services["agent"]
        self.assertEqual(agent["ports"][0]["target"], 8502)
        self.assertEqual(str(agent["ports"][0]["published"]), "8502")
        self.assertEqual(str(agent["environment"]["AGENT_PORT"]), "8502")
        self.assertIn("8502", agent["command"])
        self.assertIn("llama3.2:1b", agent["command"])
        self.assertEqual(services["ollama-init"]["environment"]["LLM_MODEL"], "llama3.2:1b")


if __name__ == "__main__":
    unittest.main()
