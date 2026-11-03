"""promptfoo provider that runs the support agent and returns its decision.

The provider ``config`` selects the variant under test:

- ``llm``: ``mock`` (deterministic, offline) or ``real`` (uses ``LLM_MODEL``
  and its API key from the environment).
- ``embedding_backend``: ``offline`` (lexical) or a semantic backend (``tei``…).
- ``rerank``: enable the cross-encoder reranker.

The output is a JSON object with ``answer``, ``action`` and ``sources`` so the
assertions can grade the agent's decision, not just its wording.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEI_URL = "http://localhost:8080/v1"


def build_agent(config: dict):
    from src.agent import SupportAgent
    from src.config import Settings
    from src.llm import MockLLM, build_client
    from src.retrieval import build_retriever

    backend = config.get("embedding_backend", "offline")
    overrides: dict = {"embedding_backend": backend, "rerank_enabled": config.get("rerank", False)}
    settings = Settings(**overrides)
    if backend == "tei" and not settings.tei_base_url:
        settings = Settings(**overrides, tei_base_url=DEFAULT_TEI_URL)

    llm = MockLLM() if config.get("llm", "mock") == "mock" else build_client(settings)
    return SupportAgent(
        llm=llm,
        retriever=build_retriever(settings),
        max_turns=settings.max_turns,
        top_k=settings.top_k,
        relevance_threshold=settings.retrieval_threshold,
        system_prompt=settings.system_prompt_text,
    )


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        response = build_agent(options.get("config", {})).run(prompt)
    except Exception as exc:  # noqa: BLE001 - report infra failures as errors, not fails
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "output": json.dumps(
            {"answer": response.answer, "action": response.action, "sources": response.sources},
            ensure_ascii=False,
        )
    }
