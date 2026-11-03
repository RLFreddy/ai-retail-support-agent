"""Agent settings, read from environment variables and ``.env``.

Provider API keys (``GROQ_API_KEY``, ``OPENAI_API_KEY``...) are not settings:
Pydantic AI reads them from the environment, and the CLI loads ``.env`` into
it at startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import FilePath, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    #: Pydantic AI ``provider:model`` name, or ``mock`` for the offline model.
    #: Without the provider's API key the agent falls back to the mock.
    llm_model: str = "groq:openai/gpt-oss-20b"
    #: Model requests per question: tool call, answer and one retry.
    max_requests: int = 3
    #: Plain-text file that replaces the default system prompt (every language).
    system_prompt_file: FilePath | None = None

    #: Empty → lexical BM25 retrieval. A Pydantic AI embedding model name
    #: (e.g. openai:text-embedding-3-small, ollama:nomic-embed-text), or the
    #: model served at EMBEDDING_BASE_URL (e.g. BAAI/bge-m3 on a local TEI).
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    #: TEI cross-encoder endpoint, e.g. http://localhost:8001/rerank. Empty → off.
    rerank_url: str = ""
    rerank_model: str = ""
    top_k: int = 3
    #: BM25 and cosine scores live on different scales.
    lexical_threshold: float = 0.5
    semantic_threshold: float = 0.56

    #: ``text`` or ``json`` (one object per line, for log pipelines).
    log_format: Literal["text", "json"] = "text"
    log_level: str = "WARNING"

    @field_validator("llm_model")
    @classmethod
    def _provider_prefixed(cls, value: str) -> str:
        if value != "mock" and ":" not in value:
            raise ValueError(
                f"LLM_MODEL must be 'provider:model' (e.g. groq:openai/gpt-oss-20b), got {value!r}"
            )
        return value

    @property
    def system_prompt_override(self) -> str | None:
        """The custom prompt, or ``None`` for the default of each language."""
        if self.system_prompt_file is None:
            return None
        return self.system_prompt_file.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
