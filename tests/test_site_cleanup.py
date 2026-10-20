from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import subprocess
import unittest


STATIC = Path(__file__).resolve().parents[1] / "src/interface/static"


class PageParser(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "id" in attributes:
            self.ids.append(attributes["id"])


class SiteCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (STATIC / "index.html").read_text()
        cls.script = (STATIC / "app.js").read_text()
        cls.ids = PageParser(cls.html).ids

    def test_name_is_consistent(self):
        self.assertIn("<title>AI AGENT IOT</title>", self.html)
        self.assertIn('aria-label="Accueil AI AGENT IOT"', self.html)
        self.assertIn('<h1 class="brand-title">AI AGENT IOT</h1>', self.html)
        self.assertNotIn("Accueil Orbit", self.html)

    def test_removed_panels_are_not_in_html_or_javascript(self):
        self.assertNotIn("Le satellite bouge.", self.html)
        self.assertNotIn("L’explication, en clair", self.html)
        self.assertNotIn('class="intro"', self.html)
        for name in ("explain", "explanation", "llm-status", "llm-model"):
            self.assertNotIn(name, self.ids)
            self.assertNotIn(f'byId("{name}")', self.script)

    def test_literal_javascript_ids_exist_and_ids_are_unique(self):
        referenced = set(re.findall(r'byId\("([^"\n]+)"\)', self.script))
        self.assertFalse(referenced - set(self.ids))
        self.assertEqual(len(self.ids), len(set(self.ids)))

    def test_satellite_results_and_questions_remain(self):
        for name in ("satellite", "position", "load", "scenario", "best-mode", "mode-bars", "question-form", "question-input", "question-send"):
            self.assertIn(name, self.ids)
        self.assertIn('request("/api/ask"', self.script)
        self.assertNotIn('"/api/explain"', self.script)
        self.assertIn("askQuestion(question)", self.script)

    def test_orbital_mode_is_removed(self):
        for name in ("source-mode", "live-form", "live-norad", "live-compute"):
            self.assertNotIn(name, self.ids)
        self.assertNotIn("/live.css", self.html)
        self.assertNotIn("/api/live", self.script)
        self.assertNotIn("CelesTrak", self.html)
        self.assertNotIn("CelesTrak", self.script)
        root = STATIC.parents[2]
        self.assertFalse((root / "src/geometry/live_satellite.py").exists())
        self.assertFalse((STATIC / "live.css").exists())
        self.assertNotIn("skyfield", (root / "requirements.txt").read_text().lower())

    @unittest.skipUnless(shutil.which("node"), "Node nécessaire pour vérifier la syntaxe JavaScript.")
    def test_javascript_syntax(self):
        result = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
