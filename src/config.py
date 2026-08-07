"""Configuración central del agente (pydantic-settings).

Carga variables de entorno (.env) automáticamente, con tipos y defaults
seguros. Sin API keys → modo mock determinista (idéntico para tests y demo
offline).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.knowledge import default_system_prompt

#: Variable de entorno de la API key por proveedor LiteLLM.
PROVIDER_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

#: Pares (variable de entorno, atributo de Settings).
_KEY_FIELDS: tuple[tuple[str, str], ...] = (
    ("GROQ_API_KEY", "groq_api_key"),
    ("OPENROUTER_API_KEY", "openrouter_api_key"),
    ("OPENAI_API_KEY", "openai_api_key"),
    ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    ("GEMINI_API_KEY", "gemini_api_key"),
)

#: Variable de entorno de la key para cada backend de embeddings (no-LLM).
EMBEDDING_KEY_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def provider_key_env(model: str) -> str:
    """Nombre de la variable de entorno de la key para un modelo LiteLLM."""
    return PROVIDER_KEY_ENV.get(model.split("/")[0], "")


class Settings(BaseSettings):
    """Settings tipadas; las variables de entorno se mapean case-insensitive
    (``GROQ_API_KEY`` → ``groq_api_key``)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = "groq/openai/gpt-oss-20b"
    embedding_backend: Literal["offline", "tei", "openrouter", "openai", "gemini"] = "offline"
    tei_base_url: str = ""
    tei_embed_model: str = "openai/BAAI/bge-m3"
    tei_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    #: URL del servidor TEI del cross-encoder (bge-reranker-v2-m3), separado del
    #: de embeddings porque son 2 contenedores. Vacía → no hay rerank local.
    tei_rerank_base_url: str = ""
    openrouter_embed_model: str = "qwen/qwen3-embedding-8b"
    openai_embed_model: str = "openai/text-embedding-3-small"
    gemini_embed_model: str = "gemini/text-embedding-004"
    rerank_enabled: bool = False
    rerank_model: str = "cohere/rerank-v3.5"
    top_k: int = 3
    relevance_threshold: float = 0.15
    #: Umbral de relevancia para embeddings (otra escala de cosines, ~0.3-0.8).
    #: El 0.15 está calibrado para TF-IDF (0.1-0.4); con bge-m3 todo pasaba.
    semantic_threshold: float = 0.56
    max_turns: int = 2
    system_prompt: str = "auto"  # "auto" | ruta a un .txt | texto literal
    ui: Literal["textual", "rich"] = "textual"

    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    #: Keys de proveedores de rerank en la nube (vía litellm.rerank). No entran
    #: en ``api_keys`` para no interferir con el chequeo de LLM (has_llm_key).
    cohere_api_key: str = ""
    jina_api_key: str = ""
    together_api_key: str = ""

    @property
    def retrieval_threshold(self) -> float:
        """Umbral de relevancia según el backend: TF-IDF (offline) usa 0.15
        (escala 0.1-0.4); embeddings semánticos usan 0.60 (cosines ~0.3-0.8)."""
        if self.embedding_backend == "offline":
            return self.relevance_threshold
        return self.semantic_threshold

    @property
    def api_keys(self) -> dict[str, str]:
        """API keys no vacías, por nombre de variable original."""
        return {
            env: getattr(self, attr)
            for env, attr in _KEY_FIELDS
            if getattr(self, attr)
        }

    @property
    def has_llm_key(self) -> bool:
        """¿Hay key para el proveedor del modelo? Sin proveedor conocido, cualquier key."""
        key_env = provider_key_env(self.llm_model)
        if key_env:
            return bool(self.api_keys.get(key_env))
        return bool(self.api_keys)

    @property
    def system_prompt_text(self) -> str:
        """Resuelve ``SYSTEM_PROMPT``: ``auto`` → prompt por defecto; una ruta →
        contenido del archivo; cualquier otro valor se usa como texto literal."""
        if self.system_prompt == "auto":
            return default_system_prompt()
        path = Path(self.system_prompt)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return self.system_prompt


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de Settings."""
    return Settings()
