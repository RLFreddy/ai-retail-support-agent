"""TiendaHogar support agent.

A small hand-written loop:

1. Input guardrail (``check_input``).
2. Retrieval over the knowledge base.
3. Tool-calling loop with the LLM (at most ``max_turns`` calls).
4. Run ``get_order_status`` and feed the result back to the model.
5. Output guardrail (``check_output``) with grounding against the corpus.

``run(user_message) -> AgentResponse`` returns the answer, the action
(answer | refer | escalate | block) and the indices of the cited documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tiendahogar.guardrails import (
    GuardrailVerdict,
    check_input,
    check_output,
    extract_known_claims,
)
from tiendahogar.knowledge import DOCUMENTS, SUPPORT_EMAIL, default_system_prompt
from tiendahogar.llm import LLMClient, ToolCall
from tiendahogar.retrieval import BaseRetriever
from tiendahogar.tools import TOOL_SCHEMA, get_order_status

TOOL_NAME = "get_order_status"


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    action: str = "answer"  # answer | refer | escalate | block
    sources: list[int] = field(default_factory=list)
    guardrail: GuardrailVerdict | None = None
    raw_steps: int = 0


def _grounding_prompt(sources: list[int]) -> str:
    """Build the retrieval context block with [source:N] citations."""
    blocks = [
        f"[source:{i}] {DOCUMENTS[i].title}\n{DOCUMENTS[i].content}" for i in sources
    ]
    return (
        "CONTEXT (answer ONLY from these documents; cite each fact with its "
        "[source:N] tag):\n" + "\n\n".join(blocks)
    )


class SupportAgent:
    """The agent: guardrails, retrieval and tool calling in one loop."""

    def __init__(
        self,
        llm: LLMClient,
        retriever: BaseRetriever,
        max_turns: int = 2,
        top_k: int = 3,
        relevance_threshold: float = 0.15,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_turns = max(1, max_turns)
        self.top_k = max(1, top_k)
        self.relevance_threshold = relevance_threshold
        self.system_prompt = system_prompt or default_system_prompt()
        self.tools = tools or [TOOL_SCHEMA]

    def _execute_tool(self, call: ToolCall) -> dict[str, Any]:
        if call.name != TOOL_NAME:
            return {"ok": False, "error": f"unknown tool: {call.name}"}
        try:
            return get_order_status(call.arguments.get("order_id", ""))
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    def run(self, user_message: str) -> AgentResponse:
        verdict = check_input(user_message)
        if verdict.action != "proceed":
            return self._handle_guardrail(verdict, sources=[])

        sources, context = self._retrieve(user_message)
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": user_message},
        ]

        steps = 0
        tool_claims: set[str] = set()
        while steps < self.max_turns:
            steps += 1
            result = self.llm.chat(messages, tools=self.tools)
            if result.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in result.tool_calls
                    ],
                })
                for tc in result.tool_calls:
                    tool_msg = self._execute_tool(tc)
                    if isinstance(tool_msg, dict) and tool_msg.get("found") is True:
                        tool_claims |= extract_known_claims(
                            str(tool_msg.get("estimated_delivery", ""))
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_msg, ensure_ascii=False),
                    })
                continue

            answer = result.content.strip()
            if not answer:
                break
            out_verdict = check_output(answer, known_extra=tool_claims)
            return AgentResponse(
                answer=answer,
                action=self._map_output_action(out_verdict),
                sources=sources,
                guardrail=out_verdict,
                raw_steps=steps,
            )

        return AgentResponse(
            answer=(
                "I could not get an answer from the model within the allowed "
                f"attempts. Please contact {SUPPORT_EMAIL}."
            ),
            action="refer",
            sources=sources,
            raw_steps=steps,
        )

    def _retrieve(self, user_message: str) -> tuple[list[int], str]:
        try:
            results = self.retriever.retrieve(
                user_message, top_k=self.top_k, threshold=self.relevance_threshold
            )
        except Exception:  # noqa: BLE001 - no retrieval means no context
            results = []
        if not results:
            return [], self._no_context_prompt()
        sources = [DOCUMENTS.index(r.document) for r in results]
        sources = list(dict.fromkeys(sources))
        return sources, _grounding_prompt(sources)

    def _no_context_prompt(self) -> str:
        return (
            "No document is relevant to this question. Say you do not have that "
            f"information and suggest contacting {SUPPORT_EMAIL}."
        )

    def _handle_guardrail(self, verdict: GuardrailVerdict, sources: list[int]) -> AgentResponse:
        action_messages = {
            "refer": (
                "A human agent has to handle this topic. Please write to "
                f"{SUPPORT_EMAIL}."
            ),
            "escalate": (
                "This amount needs approval from a human supervisor; the AI "
                f"assistant cannot resolve it. Please write to {SUPPORT_EMAIL}."
            ),
            "block": (
                "I cannot process this request because it is outside the scope "
                "of TiendaHogar support."
            ),
        }
        answer = action_messages.get(
            verdict.action,
            (
                "I do not have information to answer that question. Please "
                f"contact {SUPPORT_EMAIL}."
            ),
        )
        return AgentResponse(
            answer=answer,
            action=verdict.action,
            sources=sources,
            guardrail=verdict,
        )

    def _map_output_action(self, verdict: GuardrailVerdict) -> str:
        return {
            "proceed": "answer",
            "escalate": "escalate",
        }.get(verdict.action, "refer")


def build_agent(settings=None) -> SupportAgent:
    """Build the full agent from the settings."""
    from tiendahogar.config import get_settings
    from tiendahogar.llm import build_client
    from tiendahogar.retrieval import build_retriever

    settings = settings or get_settings()
    return SupportAgent(
        llm=build_client(settings),
        retriever=build_retriever(settings),
        max_turns=settings.max_turns,
        top_k=settings.top_k,
        relevance_threshold=settings.retrieval_threshold,
        system_prompt=settings.system_prompt_text,
    )