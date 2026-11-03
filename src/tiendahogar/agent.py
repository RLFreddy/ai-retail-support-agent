"""TiendaHogar support agent.

1. Input guardrail (``check_input``): refer, escalate or block before any model call.
2. Retrieval over the knowledge base; the matches go into the instructions.
3. A Pydantic AI run with the ``get_order_status`` tool and a request limit.
4. Output validation: an answer that states a deadline found in no document
   or tool result gets one retry; if it still fails, the text is discarded and
   the customer is referred to a human.
5. Output guardrail: an answer that approves a refund over $500 is replaced by
   the escalation message.

``run(message) -> AgentReply`` returns the answer, the action
(answer | refer | escalate | block), the indices of the retrieved documents
and the tokens spent; each run is logged with its latency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import AgentRunError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models import Model, infer_model
from pydantic_ai.usage import RunUsage, UsageLimits

from tiendahogar.config import Settings, get_settings
from tiendahogar.domain import Action, AgentReply
from tiendahogar.guardrails import (
    check_input,
    check_output,
    extract_known_claims,
    unsupported_claims,
)
from tiendahogar.knowledge import DOCUMENTS, SUPPORT_EMAIL, default_system_prompt
from tiendahogar.mock import mock_model
from tiendahogar.retrieval import Retriever, build_retriever
from tiendahogar.tools import get_order_status

log = logging.getLogger(__name__)

DECLINES = {
    Action.REFER: f"A human agent has to handle this topic. Please write to {SUPPORT_EMAIL}.",
    Action.ESCALATE: (
        "This amount needs approval from a human supervisor; the AI assistant cannot "
        f"resolve it. Please write to {SUPPORT_EMAIL}."
    ),
    Action.BLOCK: "I cannot process this request because it is outside the scope of TiendaHogar "
    "support.",
}
UNAVAILABLE = f"I cannot answer right now. Please try again later or contact {SUPPORT_EMAIL}."
UNVERIFIED = (
    "I could not confirm that answer in our policies. A human agent can help you at "
    f"{SUPPORT_EMAIL}."
)


def build_model(name: str) -> Model:
    """Resolve ``provider:model``; ``mock`` or a missing key gives the offline mock."""
    if name == "mock":
        return mock_model()
    try:
        return infer_model(name)
    except UserError as exc:
        log.warning("%s Falling back to the offline mock model.", exc)
        return mock_model()


def _context(sources: list[int]) -> str:
    if not sources:
        return (
            "No document is relevant to this question. Say you do not have that "
            f"information and suggest contacting {SUPPORT_EMAIL}."
        )
    blocks = [f"Document: {DOCUMENTS[i].title}\n{DOCUMENTS[i].content}" for i in sources]
    return (
        "CONTEXT. Answer ONLY from these documents and cite each one you use as "
        "(Source: <document title>).\n\n" + "\n\n".join(blocks)
    )


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
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.usage_limits = UsageLimits(request_limit=max_requests)
        self.is_mock = model.model_name == "mock"
        self.agent: Agent[None, str] = Agent(
            model,
            instructions=system_prompt or default_system_prompt(),
            tools=[get_order_status],
            retries=1,
        )
        self.agent.output_validator(_grounded)

    def run(self, message: str) -> AgentReply:
        start = time.perf_counter()
        usage = RunUsage()  # filled in place, also when the run fails
        reply = replace(
            self._run(message, usage),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        log.info(
            "reply",
            extra={
                "action": str(reply.action),
                "sources": reply.sources,
                "model_requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": round((time.perf_counter() - start) * 1000),
            },
        )
        return reply

    def _run(self, message: str, usage: RunUsage) -> AgentReply:
        verdict = check_input(message)
        if not verdict.passed:
            log.info("input %s: %s", verdict.action, verdict.reason)
            return AgentReply(DECLINES[verdict.action], verdict.action)

        sources = self._retrieve(message)
        try:
            result = self.agent.run_sync(
                message,
                instructions=_context(sources),
                usage_limits=self.usage_limits,
                usage=usage,
            )
        except UnexpectedModelBehavior as exc:
            log.warning("no grounded answer after retrying: %s", exc)
            return AgentReply(UNVERIFIED, Action.REFER, sources)
        except AgentRunError:
            log.exception("model run failed")
            return AgentReply(UNAVAILABLE, Action.REFER, sources)

        answer = result.output.strip()
        verdict = check_output(answer, known_extra=_tool_claims(result.all_messages()))
        if verdict.action is Action.ESCALATE:
            log.warning("output escalated: %s", verdict.reason)
            return AgentReply(DECLINES[Action.ESCALATE], Action.ESCALATE, sources)
        if not verdict.passed:
            log.warning("output %s: %s", verdict.action, verdict.reason)
            return AgentReply(UNAVAILABLE, Action.REFER, sources)
        return AgentReply(answer, Action.ANSWER, sources)

    def _retrieve(self, message: str) -> list[int]:
        try:
            results = self.retriever.retrieve(message, top_k=self.top_k)
        except Exception as exc:  # noqa: BLE001 - any retriever failure degrades the same way
            log.warning("retrieval failed, answering without context: %s", exc)
            results = []
        ids = [doc.id for doc in DOCUMENTS]
        return list(dict.fromkeys(ids.index(r.doc_id) for r in results))


def build_agent(settings: Settings | None = None) -> SupportAgent:
    """Build the full agent from the settings."""
    settings = settings or get_settings()
    return SupportAgent(
        model=build_model(settings.llm_model),
        retriever=build_retriever(settings),
        top_k=settings.top_k,
        max_requests=settings.max_requests,
        system_prompt=settings.system_prompt_text,
    )
