"""TiendaHogar knowledge base.

The 5 policy documents of the technical test, translated to English. Each
document is a single chunk: they are about one paragraph long, so finer
chunking adds nothing for this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORT_EMAIL = "support@tiendahogar.example"


@dataclass(frozen=True)
class Document:
    """A knowledge base document."""

    id: str
    title: str
    content: str


DOCUMENTS: list[Document] = [
    Document(
        id="warranty",
        title="Warranty policy",
        content=(
            "All large appliances (refrigerators, washing machines, stoves) have a "
            "warranty of 12 months from the date of purchase. Small appliances "
            "(blenders, irons, toasters) have a warranty of 6 months. The warranty "
            "covers manufacturing defects, not damage caused by misuse."
        ),
    ),
    Document(
        id="returns",
        title="Returns policy",
        content=(
            "Products can be returned within 30 days of purchase if they are unused "
            "and in their original packaging. Returns after 30 days are only accepted "
            "if the product has a defect covered by the warranty. Customized products "
            'and final sale ("clearance") items cannot be returned.'
        ),
    ),
    Document(
        id="shipping",
        title="Shipping times",
        content=(
            "Shipping to the capital: 2-3 business days. Shipping to other cities: "
            "5-7 business days. International shipping is not currently available."
        ),
    ),
    Document(
        id="refunds",
        title="Refunds",
        content=(
            "Refunds are processed within 5-10 business days after the returned "
            "product is received. The money goes back to the original payment method. "
            "Refunds over $500 require approval from a human supervisor — the agent "
            "must not approve them automatically."
        ),
    ),
    Document(
        id="contact",
        title="Contact channels",
        content=(
            "For complaints about how an employee treated a customer, billing "
            "disputes, or any legal matter, the customer must be referred to a human "
            f"agent at {SUPPORT_EMAIL} — the AI assistant must not try to resolve "
            "these cases."
        ),
    ),
]


def default_system_prompt() -> str:
    """System prompt that enforces strict grounding (never invent)."""
    return (
        "You are the support assistant of TiendaHogar, a home appliance store.\n"
        "Rules:\n"
        "1. Use ONLY the information in the provided documents or from the "
        "get_order_status tool. NEVER invent policies, deadlines or order statuses.\n"
        "2. Answer in English, in short and friendly sentences.\n"
        "3. At the end, cite the document behind each fact with the format "
        "(Source: <document title>), e.g. (Source: Warranty policy).\n"
        "4. If the documents do not answer the question, say you do not have that "
        f"information and suggest contacting {SUPPORT_EMAIL}.\n"
        "5. You CANNOT approve refunds over $500: say so and refer the customer.\n"
        "6. Complaints about how an employee treated a customer, billing disputes "
        f"and legal matters: do NOT try to resolve them, refer the customer to "
        f"{SUPPORT_EMAIL}.\n"
    )
