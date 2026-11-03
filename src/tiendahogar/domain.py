"""Domain types shared by the guardrails, the agent and the interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Language(StrEnum):
    """Language of the conversation: documents, prompt and replies."""

    EN = "en"
    ES = "es"


class Action(StrEnum):
    """What the agent does with a customer message."""

    ANSWER = "answer"
    REFER = "refer"  # hand over to a human at the support email
    ESCALATE = "escalate"  # needs a human supervisor's approval
    BLOCK = "block"  # refused: injection, sensitive data, out of scope


@dataclass(frozen=True)
class Verdict:
    """Guardrail decision. ``ANSWER`` lets the message or answer through."""

    action: Action = Action.ANSWER
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.action is Action.ANSWER


@dataclass(frozen=True)
class AgentReply:
    """The agent's reply: the text for the customer, the action taken, the
    indices of the documents it was grounded on and the model tokens spent."""

    answer: str
    action: Action = Action.ANSWER
    sources: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
