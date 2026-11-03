"""Deterministic offline model for demos, tests and evals.

A Pydantic AI ``FunctionModel`` that behaves like a very literal assistant:
it calls ``get_order_status`` when the message names an order, otherwise it
answers with the retrieved document that matches the question's keywords, and
says it has no information when nothing matches.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from tiendahogar.knowledge import DOCUMENTS, SUPPORT_EMAIL

ORDER_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)

#: Keyword prefix → index of the document that answers it.
KEYWORDS = {
    "warrant": 0,
    "guarantee": 0,
    "return": 1,
    "ship": 2,
    "deliver": 2,
    "refund": 3,
    "contact": 4,
    "complain": 4,
    "support": 4,
    "billing": 4,
    "legal": 4,
    "dispute": 4,
    "employee": 4,
}


def _answer_from_order(status: Mapping[str, object]) -> str:
    order_id = status.get("order_id", "")
    if status.get("found") is not True:
        return (
            f"I could not find order {order_id}. Please check the number or "
            f"contact {SUPPORT_EMAIL}."
        )
    answer = f"Your order {order_id} ({status['product']}) is {str(status['status']).lower()}"
    estimate = status.get("estimated_delivery", "-")
    if estimate != "-":
        answer += f" and will arrive in {estimate}"
    return answer + "."


def _answer_from_context(question: str, context: str) -> str:
    best: tuple[str, int] | None = None
    for token in re.findall(r"[a-z0-9]+", question.lower()):
        for keyword, idx in KEYWORDS.items():
            if token.startswith(keyword) and (best is None or len(keyword) > len(best[0])):
                best = (keyword, idx)
    document = DOCUMENTS[best[1]] if best else None
    if document is None or document.content not in context:
        return "Sorry, I do not have that information."
    return f"{document.content} (Source: {document.title})"


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    request = messages[-1]
    assert isinstance(request, ModelRequest)

    for part in request.parts:
        if isinstance(part, ToolReturnPart) and isinstance(part.content, Mapping):
            return ModelResponse(parts=[TextPart(_answer_from_order(part.content))])

    question = next(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))
    order = ORDER_RE.search(question)
    if order and any(t.name == "get_order_status" for t in info.function_tools):
        return ModelResponse(parts=[ToolCallPart("get_order_status", {"order_id": order.group()})])
    return ModelResponse(parts=[TextPart(_answer_from_context(question, info.instructions or ""))])


def mock_model() -> FunctionModel:
    return FunctionModel(respond, model_name="mock")
