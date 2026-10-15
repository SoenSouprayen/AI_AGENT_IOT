from dataclasses import dataclass
import math
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import requests

from src.prediction.geometry_schema import PROJECT_ROOT


class OllamaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("Préciser --llm-model ou OLLAMA_MODEL.")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL doit être une URL HTTP ou HTTPS.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Le délai Ollama doit être fini et strictement positif.")

    @classmethod
    def from_environment(cls, model=None, base_url=None, timeout=None):
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        return cls(
            model=model if model is not None else os.getenv("OLLAMA_MODEL", ""),
            base_url=base_url if base_url is not None else os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434",
            ),
            timeout=timeout if timeout is not None else float(os.getenv("OLLAMA_TIMEOUT", "120")),
        )


def create_chat_model(config):
    try:
        response = requests.get(
            f"{config.base_url.rstrip('/')}/api/tags", timeout=min(config.timeout, 5),
        )
        response.raise_for_status()
        models = {entry["name"] for entry in response.json()["models"]}
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        raise OllamaUnavailableError(
            f"Serveur Ollama inaccessible ou réponse invalide à {config.base_url}. "
            "Démarrer Ollama, puis réessayer."
        ) from error
    requested = config.model if ":" in config.model else f"{config.model}:latest"
    if config.model not in models and requested not in models:
        raise OllamaUnavailableError(
            f"Modèle Ollama absent : {config.model}. "
            "Le télécharger avec ollama pull avant de réessayer."
        )
    return ChatOllama(
        model=config.model,
        base_url=config.base_url,
        temperature=0,
        num_predict=512,
        client_kwargs={"timeout": config.timeout},
    )
