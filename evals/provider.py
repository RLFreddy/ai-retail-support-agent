"""promptfoo provider that runs the support agent and returns its decision.

The provider ``config`` selects the variant under test:

- ``llm``: ``mock`` (deterministic, offline) or ``real`` (uses ``LLM_MODEL``
  and its API key from the environment).
- ``semantic``: use bge-m3 on the local TEI server instead of lexical BM25.
- ``rerank``: also rerank with the local TEI cross-encoder.

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

TEI_EMBEDDINGS = {"embedding_model": "BAAI/bge-m3",
                  "embedding_base_url": "http://localhost:8080/v1"}
TEI_RERANK = {"rerank_url": "http://localhost:8001/rerank"}


def build_agent(config: dict):
    from tiendahogar.agent import SupportAgent
    from tiendahogar.config import Settings
    from tiendahogar.llm import MockLLM, build_client
    from tiendahogar.retrieval import build_retriever

    overrides: dict = {}
    if config.get("semantic"):
        overrides |= TEI_EMBEDDINGS
    if config.get("rerank"):
        overrides |= TEI_RERANK
    settings = Settings(**overrides)

    llm = MockLLM() if config.get("llm", "mock") == "mock" else build_client(settings)
    return SupportAgent(
        llm=llm,
        retriever=build_retriever(settings),
        max_turns=settings.max_turns,
        top_k=settings.top_k,
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
