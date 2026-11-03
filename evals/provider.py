"""promptfoo provider that runs the support agent and returns its decision.

The provider ``config`` selects the variant under test:

- ``llm``: ``mock`` (deterministic, offline) or ``real`` (``LLM_MODEL`` and
  its API key from the environment, e.g. ``--env-file .env``).
- ``semantic``: use bge-m3 on the local TEI server instead of lexical BM25.
- ``rerank``: also rerank with the local TEI cross-encoder.

Test variables: ``language`` (``en`` by default) picks the agent's language,
and ``previous`` is asked first so that its reply becomes the history of the
message under test (a follow-up).

The output is a JSON object with ``answer``, ``action`` and ``sources`` so the
assertions can grade the agent's decision, not just its wording. Token usage
is reported to promptfoo, which adds it (and the latency) to the results.
"""

from __future__ import annotations

import json

from tiendahogar.agent import build_agent as build
from tiendahogar.config import Settings
from tiendahogar.domain import Language, Turn

TEI_EMBEDDINGS = {
    "embedding_model": "BAAI/bge-m3",
    "embedding_base_url": "http://localhost:8080/v1",
}
TEI_RERANK = {"rerank_url": "http://localhost:8001/rerank"}


def build_agent(config: dict, language: Language = Language.EN):
    overrides: dict = {}
    if config.get("semantic"):
        overrides |= TEI_EMBEDDINGS
    if config.get("rerank"):
        overrides |= TEI_RERANK
    if config.get("llm", "mock") == "mock":
        overrides["llm_model"] = "mock"
    return build(Settings(**overrides), language)


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        test_vars = context.get("vars", {})
        agent = build_agent(options.get("config", {}), Language(test_vars.get("language", "en")))
        history: list[Turn] = []
        tokens = [0, 0]
        if previous := test_vars.get("previous"):
            first = agent.run(previous)
            history = [Turn("user", previous), Turn("assistant", first.answer)]
            tokens = [first.input_tokens, first.output_tokens]
        response = agent.run(prompt, history)
    except Exception as exc:  # noqa: BLE001 - report infra failures as errors, not fails
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "output": json.dumps(
            {"answer": response.answer, "action": response.action, "sources": response.sources},
            ensure_ascii=False,
        ),
        "tokenUsage": {
            "prompt": tokens[0] + response.input_tokens,
            "completion": tokens[1] + response.output_tokens,
            "total": sum(tokens) + response.input_tokens + response.output_tokens,
        },
    }
