"""TiendaHogar support agent.

1. Input guardrail (``check_input``): refer, escalate or block before any model call.
2. Retrieval over the knowledge base; the matches go into the instructions.
3. A Pydantic AI run with the ``get_order_status`` tool and a request limit.
4. Output validation: an answer that states a deadline found in no document
   or tool result gets one retry; if it still fails, the text is discarded and
   the customer is referred to a human.
5. Output guardrail: an answer that approves a refund over $500 is replaced by
   the escalation message.

Conversation memory: ``run`` accepts the earlier turns, sent by the client.
The last ``MAX_HISTORY`` turns go to the model, minus any customer message a
guardrail would stop (and its reply), so a blocked injection never reaches
the model later. Follow-ups such as "and for a blender?" retrieve with the
previous question when they match nothing alone, and a refund split across
two messages ("I want a refund" / "it cost $1200") still escalates.

One ``SupportAgent`` serves one language (documents, prompt and replies);
``build_agents`` builds one per language around a shared model.

``run(message, history) -> AgentReply`` returns the answer, the action
(answer | refer | escalate | block), the indices of the retrieved documents
and the tokens spent; each run is logged with its latency.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, infer_model
from pydantic_ai.usage import RunUsage, UsageLimits

from tiendahogar.config import Settings, get_settings
from tiendahogar.domain import Action, AgentReply, Language, Turn
from tiendahogar.guardrails import (
    check_input,
    check_output,
    extract_known_claims,
    unsupported_claims,
)
from tiendahogar.knowledge import KNOWLEDGE, SUPPORT_EMAIL, default_system_prompt
from tiendahogar.mock import mock_model
from tiendahogar.retrieval import Retriever, build_retriever
from tiendahogar.tools import get_order_status

log = logging.getLogger(__name__)

#: Earlier turns sent to the model: the last three questions and answers.
MAX_HISTORY = 6

DECLINES = {
    Language.EN: {
        Action.REFER: f"A human agent has to handle this topic. Please write to {SUPPORT_EMAIL}.",
        Action.ESCALATE: (
            "This amount needs approval from a human supervisor; the AI assistant cannot "
            f"resolve it. Please write to {SUPPORT_EMAIL}."
        ),
        Action.BLOCK: (
            "I cannot process this request because it is outside the scope of TiendaHogar support."
        ),
    },
    Language.ES: {
        Action.REFER: f"Este tema lo debe atender un agente humano. Escríbenos a {SUPPORT_EMAIL}.",
        Action.ESCALATE: (
            "Este monto necesita la aprobación de un supervisor humano; el asistente de IA no "
            f"puede resolverlo. Escríbenos a {SUPPORT_EMAIL}."
        ),
        Action.BLOCK: (
            "No puedo procesar esta solicitud porque está fuera del alcance del soporte de "
            "TiendaHogar."
        ),
    },
}
UNAVAILABLE = {
    Language.EN: f"I cannot answer right now. Please try again later or contact {SUPPORT_EMAIL}.",
    Language.ES: (
        f"No puedo responder en este momento. Inténtalo más tarde o escribe a {SUPPORT_EMAIL}."
    ),
}
UNVERIFIED = {
    Language.EN: (
        "I could not confirm that answer in our policies. A human agent can help you at "
        f"{SUPPORT_EMAIL}."
    ),
    Language.ES: (
        "No pude confirmar esa respuesta en nuestras políticas. Un agente humano puede "
        f"ayudarte en {SUPPORT_EMAIL}."
    ),
}
_CONTEXT = {
    Language.EN: (
        "Document",
        "CONTEXT. Answer ONLY from these documents and cite each one you use as "
        "(Source: <document title>).",
        "No document is relevant to this question. Say you do not have that "
        f"information and suggest contacting {SUPPORT_EMAIL}.",
    ),
    Language.ES: (
        "Documento",
        "CONTEXTO. Responde SOLO con estos documentos y cita cada uno que uses como "
        "(Fuente: <título del documento>).",
        "Ningún documento responde esta pregunta. Di que no tienes esa información y "
        f"sugiere escribir a {SUPPORT_EMAIL}.",
    ),
}


def build_model(name: str) -> Model:
    """Resolve ``provider:model``; ``mock`` or a missing key gives the offline mock."""
    if name == "mock":
        return mock_model()
    try:
        return infer_model(name)
    except UserError as exc:
        log.warning("%s Falling back to the offline mock model.", exc)
        return mock_model()


def _context(sources: list[int], language: Language = Language.EN) -> str:
    label, header, empty = _CONTEXT[language]
    if not sources:
        return empty
    documents = KNOWLEDGE[language]
    blocks = [f"{label}: {documents[i].title}\n{documents[i].content}" for i in sources]
    return header + "\n\n" + "\n\n".join(blocks)


def _clean_history(history: Sequence[Turn]) -> list[Turn]:
    """The last turns a guardrail lets through, starting with a customer turn."""
    kept: list[Turn] = []
    allowed = False
    for turn in history:
        if turn.role == "user":
            allowed = check_input(turn.content).passed
        if allowed and turn.content.strip():
            kept.append(turn)
    kept = kept[-MAX_HISTORY:]
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


def _to_messages(history: list[Turn]) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(turn.content)])
        if turn.role == "user"
        else ModelResponse(parts=[TextPart(turn.content)])
        for turn in history
    ]


def _tool_claims(messages: list[ModelMessage]) -> set[str]:
    """Deadlines returned by the order tool count as grounded facts."""
    claims: set[str] = set()
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and isinstance(part.content, dict):
                claims |= extract_known_claims(str(part.content.get("estimated_delivery", "")))
    return claims


def _grounded(ctx: RunContext[None], output: str) -> str:
    """Ask the model to retry an answer with deadlines found in no source."""
    unsupported = unsupported_claims(output, _tool_claims(ctx.messages))
    if unsupported:
        raise ModelRetry(
            f"These deadlines are not in the documents or the order data: "
            f"{', '.join(sorted(unsupported))}. Answer again using only those sources."
        )
    return output


class SupportAgent:
    """Guardrails and retrieval around a Pydantic AI agent with the order tool."""

    def __init__(
        self,
        model: Model,
        retriever: Retriever,
        top_k: int = 3,
        max_requests: int = 3,
        system_prompt: str | None = None,
        language: Language = Language.EN,
    ):
        self.retriever = retriever
        self.language = language
        self.top_k = top_k
        self.usage_limits = UsageLimits(request_limit=max_requests)
        self.is_mock = model.model_name == "mock"
        self.agent: Agent[None, str] = Agent(
            model,
            instructions=system_prompt or default_system_prompt(language),
            tools=[get_order_status],
            retries=1,
        )
        self.agent.output_validator(_grounded)

    def run(self, message: str, history: Sequence[Turn] = ()) -> AgentReply:
        start = time.perf_counter()
        usage = RunUsage()  # filled in place, also when the run fails
        reply = replace(
            self._run(message, history, usage),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        log.info(
            "reply",
            extra={
                "action": str(reply.action),
                "sources": reply.sources,
                "history_turns": len(history),
                "model_requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": round((time.perf_counter() - start) * 1000),
            },
        )
        return reply

    def _run(self, message: str, history: Sequence[Turn], usage: RunUsage) -> AgentReply:
        verdict = check_input(message)
        if not verdict.passed:
            log.info("input %s: %s", verdict.action, verdict.reason)
            return AgentReply(DECLINES[self.language][verdict.action], verdict.action)

        turns = _clean_history(history)
        previous = next((t.content for t in reversed(turns) if t.role == "user"), "")
        if previous and check_input(f"{previous}\n{message}").action is Action.ESCALATE:
            log.info("input escalate: refund split across two messages")
            return AgentReply(DECLINES[self.language][Action.ESCALATE], Action.ESCALATE)

        sources = self._retrieve(message)
        if not sources and previous:
            sources = self._retrieve(f"{previous}\n{message}")
        try:
            result = self.agent.run_sync(
                message,
                message_history=_to_messages(turns),
                instructions=_context(sources, self.language),
                usage_limits=self.usage_limits,
                usage=usage,
            )
        except UnexpectedModelBehavior as exc:
            log.warning("no grounded answer after retrying: %s", exc)
            return AgentReply(UNVERIFIED[self.language], Action.REFER, sources)
        except AgentRunError:
            log.exception("model run failed")
            return AgentReply(UNAVAILABLE[self.language], Action.REFER, sources)

        answer = result.output.strip()
        verdict = check_output(answer, known_extra=_tool_claims(result.all_messages()))
        if verdict.action is Action.ESCALATE:
            log.warning("output escalated: %s", verdict.reason)
            return AgentReply(DECLINES[self.language][Action.ESCALATE], Action.ESCALATE, sources)
        if not verdict.passed:
            log.warning("output %s: %s", verdict.action, verdict.reason)
            return AgentReply(UNAVAILABLE[self.language], Action.REFER, sources)
        return AgentReply(answer, Action.ANSWER, sources)

    def _retrieve(self, message: str) -> list[int]:
        try:
            results = self.retriever.retrieve(message, top_k=self.top_k)
        except Exception as exc:  # noqa: BLE001 - any retriever failure degrades the same way
            log.warning("retrieval failed, answering without context: %s", exc)
            results = []
        ids = [doc.id for doc in KNOWLEDGE[self.language]]
        return list(dict.fromkeys(ids.index(r.doc_id) for r in results))


def build_agent(
    settings: Settings | None = None,
    language: Language = Language.EN,
    model: Model | None = None,
) -> SupportAgent:
    """Build the full agent for one language from the settings."""
    settings = settings or get_settings()
    return SupportAgent(
        model=model or build_model(settings.llm_model),
        retriever=build_retriever(settings, language),
        top_k=settings.top_k,
        max_requests=settings.max_requests,
        system_prompt=settings.system_prompt_override,
        language=language,
    )


def build_agents(settings: Settings | None = None) -> dict[Language, SupportAgent]:
    """One agent per language, sharing the model."""
    settings = settings or get_settings()
    model = build_model(settings.llm_model)
    return {language: build_agent(settings, language, model) for language in Language}
