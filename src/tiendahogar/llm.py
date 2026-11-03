"""Capas de LLM del agente TiendaHogar.

``LiteLLMClient``: envoltorio delgado sobre ``litellm.completion`` con
tool-calling (funciones) y una interfaz mínima común.
``MockLLM``: simulador determinista SIN red. Se usa cuando no hay API key
(``has_llm_key == False``) para que el agente sea 100% testeable/demostrable.

Diseño: ambas clases exponen ``chat(messages, tools) -> LLMResult``, donde
``LLMResult`` normaliza tool calls (lista de ``ToolCall``) + contenido.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from tiendahogar.config import provider_key_env
from tiendahogar.knowledge import DOCUMENTS
from tiendahogar.retrieval import _tokenize


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
    """Interfaz común de los clientes LLM."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResult: ...


class LiteLLMClient:
    """Cliente sobre litellm.completion con soporte de tools."""

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
        except Exception as exc:  # noqa: BLE001 - cualquier fallo de red/proveedor
            return LLMResult(
                content=f"Error del modelo: {exc}", finish_reason="error"
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
    """Simulador determinista del LLM (sin red).

    No genera contenido real. Se usa SOLO para probar el flujo del agente sin
    red: detecta si el último mensaje pide una tool y la "delega" respondiendo
    con una tool call simulada; en caso contrario devuelve un eco fijo.
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

        # Si la tool ya se ejecutó, el modelo "consume" su resultado y responde.
        tool_msg = next(
            (m for m in reversed(messages) if m.get("role") == "tool"), None
        )
        if tool_msg is not None:
            try:
                payload = json.loads(tool_msg["content"])
            except (json.JSONDecodeError, ValueError, TypeError):
                payload = {}
            if payload.get("encontrado") is False:
                return LLMResult(
                    content=(
                        f"No encontré el pedido {payload.get('order_id', '')}. "
                        "Por favor verifica el número o contacta a "
                        "soporte@tiendahogar.example."
                    )
                )
            order_id = payload.get("order_id", "")
            producto = payload.get("producto", "")
            estado = payload.get("estado", "").lower()
            estimate = str(payload.get("entrega_estimada", "")).strip()
            if estimate and estimate != "-":
                return LLMResult(
                    content=(
                        f"Tu pedido {order_id} ({producto}) está {estado}"
                        f" y se entregará en {estimate}."
                    )
                )
            return LLMResult(
                content=f"Tu pedido {order_id} ({producto}) está {estado}."
            )

        # Solo llamamos la tool si el usuario menciona un pedido explícito.
        found = re.search(r"(ORD-\d+)", last_user)
        tool_names = [t.get("function", {}).get("name") for t in (tools or []) if t.get("type") == "function"]
        if found and "consultar_estado_pedido" in tool_names:
            return LLMResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="mock-1",
                        name="consultar_estado_pedido",
                        arguments={"order_id": found.group(1)},
                    )
                ],
                finish_reason="tool_calls",
            )

        # Grounding-aware: respondemos desde el bloque CONTEXTO (si hay).
        grounded = self._answer_from_context(messages, last_user)
        if grounded is not None:
            return LLMResult(content=grounded)
        return LLMResult(content="Lo siento, no tengo esa información.")

    @staticmethod
    def _answer_from_context(messages: list[dict], last_user: str) -> str | None:
        """Extrae el bloque CONTEXTO de los system messages y responde con el
        fragmento más relacionado, citando la fuente por título. None si no hay contexto."""
        context = ""
        for m in messages:
            if m.get("role") == "system" and "CONTEXTO" in str(m.get("content", "")):
                context = m["content"]
                break
        if not context:
            return None

        # km de keywords -> fuente para routing simple.
        user_tokens = set(_tokenize(last_user))
        keywords = {
            "garant": 0,
            "devol": 1,
            "envio": 2,
            "reembols": 3,
            "contact": 4, "queja": 4, "soporte": 4, "factur": 4, "legal": 4,
            "disputa": 4, "trato": 4, "empleado": 4,
        }
        best: tuple[str, int] | None = None
        for token in user_tokens:
            for kw, idx in keywords.items():
                if token.startswith(kw) and (best is None or len(kw) > len(best[0])):
                    best = (kw, idx)

        if best is None:
            return None
        _, idx = best
        snippet = re.search(
            rf"\[fuente:{idx}\][^\n]*\n([^\[]+)", context
        )
        if snippet and snippet.group(1).strip():
            return snippet.group(1).strip() + f" (Fuente: {DOCUMENTS[idx].title})"
        return "Según la política de soporte, contacta a soporte@tiendahogar.example."


def build_client(settings=None) -> LLMClient:
    """Construye el cliente LLM según la configuración."""
    from tiendahogar.config import get_settings

    settings = settings or get_settings()
    if not settings.has_llm_key:
        return MockLLM()
    key = settings.api_keys.get(provider_key_env(settings.llm_model))
    return LiteLLMClient(model=settings.llm_model, api_key=key or "")