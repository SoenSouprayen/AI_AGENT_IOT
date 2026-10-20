import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLLAMA_URL = "http://127.0.0.1:11434"
LOCAL_HTTP = build_opener(ProxyHandler({}))


def ollama_request(path, payload=None, timeout=2):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(OLLAMA_URL + path, data=body, headers={"Content-Type": "application/json"})
    with LOCAL_HTTP.open(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"Réponse Ollama invalide : {result}")
    return result


def local_models():
    try:
        result = ollama_request("/api/tags")
        return {entry["name"] for entry in result["models"]}
    except (OSError, URLError, ValueError, KeyError, TypeError):
        return None


def wait_for_ollama(timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        models = local_models()
        if models is not None:
            return models
        time.sleep(1)
    raise ValueError("Ollama local ne répond pas après 60 secondes. Ouvrir l’application Ollama puis réessayer.")


def ollama_command():
    executable = shutil.which("ollama")
    bundled = Path("/Applications/Ollama.app/Contents/Resources/ollama")
    if executable:
        return executable
    if bundled.is_file():
        return str(bundled)
    raise ValueError("Ollama n’est pas installé. Installer l’application depuis https://ollama.com/download/mac.")


def ensure_model(model):
    models = local_models()
    if models is None:
        if sys.platform != "darwin":
            raise ValueError("Démarrer Ollama avec « ollama serve », puis relancer cette commande.")
        print("Démarrage de l’application Ollama…", flush=True)
        subprocess.run(["open", "-a", "Ollama"], check=True)
        models = wait_for_ollama()
    expected = model if ":" in model else f"{model}:latest"
    if model not in models and expected not in models:
        print(f"Premier lancement : téléchargement de {model} (peut prendre plusieurs minutes)…", flush=True)
        environment = os.environ.copy()
        environment["OLLAMA_HOST"] = OLLAMA_URL
        subprocess.run([ollama_command(), "pull", model], env=environment, check=True)
        models = local_models()
        if models is None or not {model, expected}.intersection(models):
            raise ValueError("Le modèle reste indisponible après le téléchargement.")
    print("Chargement du modèle en mémoire…", flush=True)
    ollama_request("/api/generate", {"model": model, "prompt": "", "stream": False, "keep_alive": "15m"}, timeout=180)


def choose_port(requested=None):
    ports = [requested] if requested is not None else range(8501, 8511)
    for port in ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise ValueError("Port occupé ou inaccessible. Choisir un autre port avec --port 8511.")


def check_project():
    required = ("interface.py", "data/processed/scenario_dataset_1deg.csv",
                "models/per_model/per_predictor_A_Geometry_1deg.joblib")
    missing = [name for name in required if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise ValueError("Fichiers requis absents : " + ", ".join(missing))
    modules = ("pandas", "numpy", "sklearn", "joblib", "dotenv", "requests", "langchain_core", "langchain_ollama")
    if any(importlib.util.find_spec(name) is None for name in modules):
        raise ValueError("Dépendances manquantes : exécuter .venv/bin/python -m pip install -r requirements.txt.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lancer le site avec l’application Ollama locale.")
    parser.add_argument("--llm-model", default="llama3.2:3b")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("Le port doit être compris entre 1 et 65535.")
    if not args.llm_model.strip() or args.llm_model.startswith("-"):
        parser.error("Nom de modèle invalide.")
    child = None
    try:
        check_project()
        port = choose_port(args.port)
        ensure_model(args.llm_model)
        command = [sys.executable, str(PROJECT_ROOT / "interface.py"), "--host", "127.0.0.1",
                   "--port", str(port), "--ollama-url", OLLAMA_URL, "--llm-model", args.llm_model]
        if args.no_browser:
            command.append("--no-browser")
        print(f"Site local : http://localhost:{port} — Ctrl+C pour arrêter le site.", flush=True)
        if port != 8501:
            print("Le port 8501 est indisponible ; un autre port a été choisi. Les services existants restent inchangés.", flush=True)
        child = subprocess.Popen(command, cwd=PROJECT_ROOT)
        return child.wait()
    except KeyboardInterrupt:
        print("\nSite arrêté. L’application Ollama reste disponible.", flush=True)
        return 130
    except (OSError, URLError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
