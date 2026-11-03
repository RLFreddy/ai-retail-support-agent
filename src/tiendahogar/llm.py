"""LLM clients of the TiendaHogar agent.

- ``LiteLLMClient``: thin wrapper over ``litellm.completion`` with tool calling.
- ``MockLLM``: deterministic stand-in without network, used when there is no
  API key so the agent can be tested and demoed offline.

Both expose ``chat(messages, tools) -> LLMResult``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from tiendahogar.config import provider_key_env
from tiendahogar.knowledge import DOCUMENTS, SUPPORT_EMAIL


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: Any | None = None


class LLMClient(Protocol):
    """Common interface of the LLM clients."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult: ...


class LiteLLMClient:
    """Client over litellm.completion with tool support."""

    def __init__(self, model: str, api_key: str = ""):
        self.model = model
        self.api_key = api_key

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        import litellm

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if tools:
            kwargs["tools"] = tools

        try:
            resp = litellm.completion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - any network or provider failure
            return LLMResult(
                content=f"Model error: {exc}", finish_reason="error"
            )

        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.function.name or ""
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(ToolCall(id=tc.id or "", name=name, arguments=arguments))
        return LLMResult(
            content=getattr(msg, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            raw=resp,
        )


class MockLLM:
    """Deterministic LLM stand-in (no network).

    Used to exercise the agent flow offline: it calls the order tool when the
    message names an order, answers from the retrieved context by keyword
    routing, and otherwise says it has no information.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult:
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        for key, answer in self.responses.items():
            if key.lower() in last_user.lower():
                return LLMResult(content=answer)

        tool_msg = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        if tool_msg is not None:
            return LLMResult(content=self._answer_from_tool(tool_msg))

        found = re.search(r"(ORD-\d+)", last_user, re.IGNORECASE)
        tool_names = [
            t.get("function", {}).get("name")
            for t in (tools or [])
            if t.get("type") == "function"
        ]
        if found and "get_order_status" in tool_names:
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id="mock-1",
                        name="get_order_status",
                        arguments={"order_id": found.group(1)},
                    )
                ],
                finish_reason="tool_calls",
            )

        grounded = self._answer_from_context(messages, last_user)
        if grounded is not None:
            return LLMResult(content=grounded)
        return LLMResult(content="Sorry, I do not have that information.")

    @staticmethod
    def _answer_from_tool(tool_msg: dict) -> str:
        try:
            payload = json.loads(tool_msg["content"])
        except (json.JSONDecodeError, ValueError, TypeError):
            payload = {}
        order_id = payload.get("order_id", "")
        if payload.get("found") is not True:
            return (
                f"I could not find order {order_id}. Please check the number or "
                f"contact {SUPPORT_EMAIL}."
            )
        product = payload.get("product", "")
        status = payload.get("status", "").lower()
        estimate = str(payload.get("estimated_delivery", "")).strip()
        if estimate and estimate != "-":
            return (
                f"Your order {order_id} ({product}) is {status}"
                f" and will arrive in {estimate}."
            )
        return f"Your order {order_id} ({product}) is {status}."

    @staticmethod
    def _answer_from_context(messages: list[dict], last_user: str) -> str | None:
        """Answer with the context snippet that matches the question's keywords,
        citing the source by title. None when there is no context."""
        context = next(
            (
                m["content"]
                for m in messages
                if m.get("role") == "system" and "CONTEXT" in str(m.get("content", ""))
            ),
            "",
        )
        if not context:
            return None

        keywords = {
            "warrant": 0, "guarantee": 0,
            "return": 1,
            "ship": 2, "deliver": 2,
            "refund": 3,
            "contact": 4, "complain": 4, "support": 4, "billing": 4, "legal": 4,
            "dispute": 4, "employee": 4,
        }
        best: tuple[str, int] | None = None
        for token in re.findall(r"[a-z0-9]+", last_user.lower()):
            for kw, idx in keywords.items():
                if token.startswith(kw) and (best is None or len(kw) > len(best[0])):
                    best = (kw, idx)
        if best is None:
            return None

        idx = best[1]
        snippet = re.search(rf"\[source:{idx}\][^\n]*\n([^\[]+)", context)
        if snippet and snippet.group(1).strip():
            return snippet.group(1).strip() + f" (Source: {DOCUMENTS[idx].title})"
        return f"Per our support policy, please contact {SUPPORT_EMAIL}."


def build_client(settings=None) -> LLMClient:
    """Build the LLM client from the settings."""
    from tiendahogar.config import get_settings

    settings = settings or get_settings()
    if not settings.has_llm_key:
        return MockLLM()
    key = settings.api_keys.get(provider_key_env(settings.llm_model))
    return LiteLLMClient(model=settings.llm_model, api_key=key or "")