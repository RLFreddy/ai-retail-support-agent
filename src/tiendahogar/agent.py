"""TiendaHogar support agent.

1. Input guardrail (``check_input``): refer, escalate or block before any model call.
2. Retrieval over the knowledge base; the matches go into the instructions.
3. A Pydantic AI run with the ``get_order_status`` tool and a request limit.
4. Output guardrail (``check_output``) with grounding against the corpus and
   the tool results.

``run(message) -> AgentReply`` returns the answer, the action
(answer | refer | escalate | block) and the indices of the retrieved documents.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent
from pydantic_ai.exceptions import AgentRunError, UserError
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models import Model, infer_model
from pydantic_ai.usage import UsageLimits

from tiendahogar.domain import Action, AgentReply
from tiendahogar.guardrails import check_input, check_output, extract_known_claims
from tiendahogar.knowledge import DOCUMENTS, SUPPORT_EMAIL, default_system_prompt
from tiendahogar.mock import mock_model
from tiendahogar.retrieval import Retriever
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
UNAVAILABLE = (
    f"I cannot answer right now. Please try again later or contact {SUPPORT_EMAIL}."
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
    blocks = [f"[source:{i}] {DOCUMENTS[i].title}\n{DOCUMENTS[i].content}" for i in sources]
    return (
        "CONTEXT (answer ONLY from these documents; cite each fact with its "
        "[source:N] tag):\n" + "\n\n".join(blocks)
    )


class SupportAgent:
    """Guardrails and retrieval around a Pydantic AI agent with the order tool."""

    def __init__(
        self,
        model: Model | str,
        retriever: Retriever,
        top_k: int = 3,
        max_requests: int = 3,
        system_prompt: str | None = None,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.usage_limits = UsageLimits(request_limit=max_requests)
        self.agent = Agent(
            model,
            instructions=system_prompt or default_system_prompt(),
            tools=[get_order_status],
        )

    @property
    def is_mock(self) -> bool:
        return self.agent.model.model_name == "mock"

    def run(self, message: str) -> AgentReply:
        verdict = check_input(message)
        if not verdict.passed:
            log.info("input %s: %s", verdict.action, verdict.reason)
            return AgentReply(DECLINES[verdict.action], verdict.action)

        sources = self._retrieve(message)
        try:
            result = self.agent.run_sync(
                message, instructions=_context(sources), usage_limits=self.usage_limits
            )
        except AgentRunError:
            log.exception("model run failed")
            return AgentReply(UNAVAILABLE, Action.REFER, sources)

        answer = result.output.strip()
        verdict = check_output(answer, known_extra=self._tool_claims(result.all_messages()))
        if not verdict.passed:
            log.info("output %s: %s", verdict.action, verdict.reason)
        action = Action.REFER if verdict.action is Action.BLOCK else verdict.action
        return AgentReply(answer, action, sources)

    def _retrieve(self, message: str) -> list[int]:
        try:
            results = self.retriever.retrieve(message, top_k=self.top_k)
        except Exception as exc:  # noqa: BLE001 - any retriever failure degrades the same way
            log.warning("retrieval failed, answering without context: %s", exc)
            results = []
        ids = [doc.id for doc in DOCUMENTS]
        return list(dict.fromkeys(ids.index(r.doc_id) for r in results))

    @staticmethod
    def _tool_claims(messages: list) -> set[str]:
        """Deadlines returned by the order tool count as grounded facts."""
        claims: set[str] = set()
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and isinstance(part.content, dict):
                    claims |= extract_known_claims(str(part.content.get("estimated_delivery", "")))
        return claims


def build_agent(settings=None) -> SupportAgent:
    """Build the full agent from the settings."""
    from tiendahogar.config import get_settings
    from tiendahogar.retrieval import build_retriever

    settings = settings or get_settings()
    return SupportAgent(
        model=build_model(settings.llm_model),
        retriever=build_retriever(settings),
        top_k=settings.top_k,
        max_requests=settings.max_requests,
        system_prompt=settings.system_prompt_text,
    )
