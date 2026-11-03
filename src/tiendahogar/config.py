"""Agent settings (pydantic-settings).

Read from environment variables and ``.env``. Without an API key the agent
runs with the deterministic mock LLM.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from tiendahogar.knowledge import default_system_prompt

#: API key environment variable per LiteLLM provider.
PROVIDER_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

#: (environment variable, Settings attribute) pairs.
_KEY_FIELDS: tuple[tuple[str, str], ...] = (
    ("GROQ_API_KEY", "groq_api_key"),
    ("OPENROUTER_API_KEY", "openrouter_api_key"),
    ("OPENAI_API_KEY", "openai_api_key"),
    ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    ("GEMINI_API_KEY", "gemini_api_key"),
)

#: API key environment variable per embedding backend.
EMBEDDING_KEY_ENV: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def provider_key_env(model: str) -> str:
    """Name of the API key environment variable for a LiteLLM model."""
    return PROVIDER_KEY_ENV.get(model.split("/")[0], "")


class Settings(BaseSettings):
    """Typed settings; environment variables match case-insensitively
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
    #: URL of the TEI cross-encoder server (bge-reranker-v2-m3), a separate
    #: container from the embeddings one.
    tei_rerank_base_url: str = ""
    openrouter_embed_model: str = "qwen/qwen3-embedding-8b"
    openai_embed_model: str = "openai/text-embedding-3-small"
    gemini_embed_model: str = "gemini/text-embedding-004"
    rerank_enabled: bool = False
    rerank_model: str = "cohere/rerank-v3.5"
    top_k: int = 3
    relevance_threshold: float = 0.15
    #: Relevance threshold for embeddings (cosine scale ~0.3-0.8); 0.15 is
    #: calibrated for TF-IDF (~0.1-0.4).
    semantic_threshold: float = 0.56
    max_turns: int = 2
    system_prompt: str = "auto"  # "auto" | path to a .txt | literal text

    groq_api_key: str = ""
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    #: Cloud rerank provider keys (litellm.rerank); kept out of ``api_keys``.
    cohere_api_key: str = ""
    jina_api_key: str = ""
    together_api_key: str = ""

    @property
    def retrieval_threshold(self) -> float:
        """Relevance threshold for the active backend."""
        if self.embedding_backend == "offline":
            return self.relevance_threshold
        return self.semantic_threshold

    @property
    def api_keys(self) -> dict[str, str]:
        """Non-empty API keys, by environment variable name."""
        return {
            env: getattr(self, attr)
            for env, attr in _KEY_FIELDS
            if getattr(self, attr)
        }

    @property
    def has_llm_key(self) -> bool:
        """Whether the model's provider has a key (any key for unknown providers)."""
        key_env = provider_key_env(self.llm_model)
        if key_env:
            return bool(self.api_keys.get(key_env))
        return bool(self.api_keys)

    @property
    def system_prompt_text(self) -> str:
        """Resolve ``SYSTEM_PROMPT``: ``auto`` → default prompt; a file path →
        its content; anything else is used as literal text."""
        if self.system_prompt == "auto":
            return default_system_prompt()
        path = Path(self.system_prompt)
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return self.system_prompt


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
