import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import webbrowser

from src.interface.playground import SatellitePlayground
from src.prediction.geometry_schema import PROJECT_ROOT
from src.prediction.per_predictor import PERPredictor
from src.utils.ollama_client import OllamaConfig


ASSETS = Path(__file__).resolve().parent / "src/interface/static"


def make_handler(playground):
    class Handler(BaseHTTPRequestHandler):
        def send_content(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, status, payload):
            self.send_content(status, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")

        def allowed_request(self):
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in hosts or (origin and origin not in {f"http://{host}" for host in hosts}):
                self.send_json(403, {"error": "Cette interface est réservée au navigateur local."})
                return False
            return True

        def do_GET(self):
            if not self.allowed_request():
                return
            if self.path == "/api/config":
                self.send_json(200, playground.configuration())
                return
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8"),
                      "/questions.css": ("questions.css", "text/css; charset=utf-8")}
            if self.path not in assets:
                self.send_json(404, {"error": "Page introuvable."})
                return
            filename, content_type = assets[self.path]
            self.send_content(200, (ASSETS / filename).read_bytes(), content_type)

        def do_POST(self):
            if not self.allowed_request():
                return
            actions = {"/api/predict": playground.predict, "/api/explain": playground.explain, "/api/ask": playground.ask}
            if self.path not in actions:
                self.send_json(404, {"error": "Action inconnue."})
                return
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Un contenu application/json est requis.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("La requête doit contenir entre 1 et 4096 octets.")
                payload = json.loads(self.rfile.read(length))
                result = actions[self.path](payload)
                self.send_json(200, result)
            except BlockingIOError as error:
                self.send_json(409, {"error": str(error)})
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception:
                self.log_error("Erreur interne lors de la recommandation")
                self.send_json(500, {"error": "La recommandation a échoué. Vérifier le modèle et le CSV dans le terminal."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Interface locale : satellite déplaçable et agent PER.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/processed/scenario_dataset_1deg.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/per_model/per_predictor_A_Geometry_1deg.joblib")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    parser.add_argument("--ollama-url")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        if not 0 <= args.port <= 65535:
            raise ValueError("Le port doit être entre 0 et 65535.")
        playground = SatellitePlayground(
            args.dataset, PERPredictor(args.model),
            OllamaConfig.from_environment(args.llm_model, args.ollama_url, args.timeout),
        )
        with ThreadingHTTPServer((args.host, args.port), make_handler(playground)) as server:
            server.daemon_threads = True
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"Interface disponible : {url}\nCtrl+C pour arrêter.", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nInterface arrêtée.")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
