"""Agente de soporte TiendaHogar.

Loop propio (sin OpenAI Agents SDK) de ~60 líneas:
1. Guardrail de entrada (check_input).
2. Si procede: RAG (retrieval sobre DOCUMENTS).
3. Loop de tool-calling con LiteLLM (max_turns).
4. Ejecutar la tool consultar_estado_pedido y realimentar el modelo.
5. Guardrail de salida (check_output) + grounding contra el corpus.

Contrato visible:
- ``run(user_message) -> AgentResponse``
- ``AgentResponse.answer``: la respuesta final para el usuario.
- ``AgentResponse.action``: "answer" | "refer" | "escalate" | "block" | "off_topic".
- Grounding: el response lleva ``sources`` (índices [fuente:N]) usados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tiendahogar.guardrails import (
    SUPPORT_EMAIL,
    GuardrailVerdict,
    check_input,
    check_output,
    extract_known_claims,
)
from tiendahogar.knowledge import DOCUMENTS, default_system_prompt
from tiendahogar.llm import LLMClient, ToolCall
from tiendahogar.retrieval import BaseRetriever
from tiendahogar.tools import TOOL_SCHEMA, consultar_estado_pedido

TOOL_NAME = "consultar_estado_pedido"


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    action: str = "answer"  # answer | refer | escalate | block | off_topic
    sources: list[int] = field(default_factory=list)
    guardrail: GuardrailVerdict | None = None
    raw_steps: int = 0


def _grounding_prompt(sources: list[int]) -> str:
    """Construye el bloque de contexto RAG con citas [fuente:N]."""
    blocks = [
        f"[fuente:{i}] {DOCUMENTS[i].title}\n{DOCUMENTS[i].content}" for i in sources
    ]
    return (
        "CONTEXTO (SOLO responde usando estos documentos; cita cada dato con su "
        "formato [fuente:N]):\n" + "\n\n".join(blocks)
    )


class SupportAgent:
    """El agente: guardrails + RAG + tool-calling con un loop propio."""

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
            return {"ok": False, "error": f"tool desconocida: {call.name}"}
        try:
            return consultar_estado_pedido(call.arguments.get("order_id", ""))
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
                    if isinstance(tool_msg, dict) and tool_msg.get("encontrado") is True:
                        tool_claims |= extract_known_claims(
                            str(tool_msg.get("entrega_estimada", ""))
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
                "No he podido obtener una respuesta del modelo en el número de "
                f"intentos permitidos. Por favor contacta a {SUPPORT_EMAIL}."
            ),
            action="refer",
            sources=sources,
            raw_steps=steps,
        )

    # ---------------- helpers ----------------
    def _retrieve(self, user_message: str) -> tuple[list[int], str]:
        try:
            results = self.retriever.retrieve(
                user_message, top_k=self.top_k, threshold=self.relevance_threshold
            )
        except Exception:  # noqa: BLE001 - fallback defensivo: sin retrieval no hay contexto
            results = []
        if not results:
            return [], self._no_context_prompt()
        sources = [DOCUMENTS.index(r.document) for r in results]
        # Evitar duplicados si el retriever devolviera el mismo doc dos veces.
        sources = list(dict.fromkeys(sources))
        return sources, _grounding_prompt(sources)

    def _no_context_prompt(self) -> str:
        return (
            "No hay documentos relevantes para esta consulta. Responde que no "
            "tienes esa información y sugiere contactar soporte@tiendahogar.example."
        )

    def _handle_guardrail(self, verdict: GuardrailVerdict, sources: list[int]) -> AgentResponse:
        action_messages = {
            "refer": (
                "Ese tema debe ser atendido por un agente humano. Por favor "
                f"escribe a {SUPPORT_EMAIL}."
            ),
            "escalate": (
                "El monto indicado requiere aprobación de un supervisor humano; "
                f"el asistente de IA no puede resolverlo. Por favor escribe a "
                f"{SUPPORT_EMAIL}."
            ),
            "block": (
                "No puedo procesar esa solicitud porque parece sesgada o fuera "
                "del alcance del soporte de TiendaHogar."
            ),
        }
        answer = action_messages.get(
            verdict.action,
            (
                "No tengo información para responder esa pregunta. Por favor "
                f"contacta a {SUPPORT_EMAIL}."
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


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_agent(settings=None) -> SupportAgent:
    """Construye el agente completo desde Settings."""
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