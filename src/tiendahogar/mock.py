"""Deterministic offline model for demos, tests and evals.

A Pydantic AI ``FunctionModel`` that behaves like a very literal assistant:
it calls ``get_order_status`` when the message names an order, otherwise it
answers with the retrieved document that matches the question's keywords, and
says it has no information when nothing matches. It answers in Spanish when
the instructions ask for it.
"""

from __future__ import annotations

import re
import unicodedata
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

from tiendahogar.domain import Language
from tiendahogar.knowledge import KNOWLEDGE, SOURCE_LABEL, SUPPORT_EMAIL

ORDER_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)

#: Keyword prefix (lowercase, no accents) → index of the document that answers it.
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
    "garant": 0,
    "devol": 1,
    "devuel": 1,
    "envi": 2,
    "entreg": 2,
    "reembols": 3,
    "queja": 4,
    "soport": 4,
    "factur": 4,
    "disput": 4,
    "emplead": 4,
}

_TEXTS = {
    Language.EN: {
        "not_found": (
            "I could not find order {order_id}. Please check the number or contact {email}."
        ),
        "status": "Your order {order_id} ({product}) is {status}",
        "eta": " and will arrive in {eta}",
        "unknown": "Sorry, I do not have that information.",
    },
    Language.ES: {
        "not_found": "No encontré el pedido {order_id}. Revisa el número o escribe a {email}.",
        "status": "Tu pedido {order_id} ({product}) está {status}",
        "eta": " y llegará en {eta}",
        "unknown": "Lo siento, no tengo esa información.",
    },
}
#: The order tool speaks English; the mock translates its statuses.
_STATUS_ES = {
    "in transit": "en camino",
    "delivered": "entregado",
    "processing": "en preparación",
    "cancelled": "cancelado",
}
_ETA_ES = re.compile(r"business days?", re.IGNORECASE)


def _language(instructions: str) -> Language:
    return Language.ES if "Responde en español" in instructions else Language.EN


def _plain(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _answer_from_order(status: Mapping[str, object], language: Language) -> str:
    texts = _TEXTS[language]
    order_id = status.get("order_id", "")
    if status.get("found") is not True:
        return texts["not_found"].format(order_id=order_id, email=SUPPORT_EMAIL)
    state = str(status["status"]).lower()
    estimate = str(status.get("estimated_delivery", "-"))
    if language is Language.ES:
        state = _STATUS_ES.get(state, state)
        estimate = _ETA_ES.sub("días hábiles", estimate)
    answer = texts["status"].format(order_id=order_id, product=status["product"], status=state)
    if estimate != "-":
        answer += texts["eta"].format(eta=estimate)
    return answer + "."


def _answer_from_context(question: str, context: str, language: Language) -> str:
    best: tuple[str, int] | None = None
    for token in re.findall(r"\w+", _plain(question)):
        for keyword, idx in KEYWORDS.items():
            if token.startswith(keyword) and (best is None or len(keyword) > len(best[0])):
                best = (keyword, idx)
    document = KNOWLEDGE[language][best[1]] if best else None
    if document is None or document.content not in context:
        return _TEXTS[language]["unknown"]
    return f"{document.content} ({SOURCE_LABEL[language]}: {document.title})"


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    language = _language(info.instructions or "")

    for part in request.parts:
        if isinstance(part, ToolReturnPart) and isinstance(part.content, Mapping):
            return ModelResponse(parts=[TextPart(_answer_from_order(part.content, language))])

    question = next(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))
    order = ORDER_RE.search(question)
    if order and any(t.name == "get_order_status" for t in info.function_tools):
        return ModelResponse(parts=[ToolCallPart("get_order_status", {"order_id": order.group()})])
    answer = _answer_from_context(question, info.instructions or "", language)
    return ModelResponse(parts=[TextPart(answer)])


def mock_model() -> FunctionModel:
    return FunctionModel(respond, model_name="mock")
