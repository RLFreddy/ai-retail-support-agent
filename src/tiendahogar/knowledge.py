"""TiendaHogar knowledge base, in English and Spanish.

The 5 policy documents of the technical test: the original Spanish texts and
their English translation, in the same order and with the same ids, so a
document index means the same policy in both languages. Each document is a
single chunk: they are about one paragraph long, so finer chunking adds nothing
for this corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from tiendahogar.domain import Language

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


DOCUMENTS_ES: list[Document] = [
    Document(
        id="warranty",
        title="Política de garantía",
        content=(
            "Todos los electrodomésticos grandes (refrigeradoras, lavadoras, estufas) "
            "tienen garantía de 12 meses desde la fecha de compra. Electrodomésticos "
            "pequeños (licuadoras, planchas, tostadoras) tienen garantía de 6 meses. "
            "La garantía cubre defectos de fábrica, no daños por mal uso."
        ),
    ),
    Document(
        id="returns",
        title="Política de devoluciones",
        content=(
            "Los productos pueden devolverse dentro de 30 días de la compra si están "
            "sin usar y en su empaque original. Devoluciones después de 30 días solo "
            "se aceptan si el producto tiene un defecto cubierto por garantía. No se "
            "aceptan devoluciones de productos personalizados o en oferta final "
            '("liquidación").'
        ),
    ),
    Document(
        id="shipping",
        title="Tiempos de envío",
        content=(
            "Envíos a la capital: 2-3 días hábiles. Envíos a otras ciudades: 5-7 días "
            "hábiles. Envíos internacionales no están disponibles actualmente."
        ),
    ),
    Document(
        id="refunds",
        title="Reembolsos",
        content=(
            "Los reembolsos se procesan en 5-10 días hábiles después de recibir el "
            "producto devuelto. Se reembolsa al mismo método de pago original. "
            "Reembolsos mayores a $500 requieren aprobación de un supervisor humano — "
            "el agente no debe aprobarlos automáticamente."
        ),
    ),
    Document(
        id="contact",
        title="Canales de contacto",
        content=(
            "Para quejas sobre el trato de un empleado, disputas de facturación, o "
            "cualquier tema legal, el cliente debe ser referido a un agente humano en "
            f"{SUPPORT_EMAIL} — el asistente de IA no debe intentar resolver estos casos."
        ),
    ),
]

KNOWLEDGE: dict[Language, list[Document]] = {Language.EN: DOCUMENTS, Language.ES: DOCUMENTS_ES}

#: How answers cite a document, per language.
SOURCE_LABEL = {Language.EN: "Source", Language.ES: "Fuente"}


def default_system_prompt(language: Language = Language.EN) -> str:
    """System prompt that enforces strict grounding (never invent)."""
    if language is Language.ES:
        return (
            "Eres el asistente de soporte de TiendaHogar, una tienda de electrodomésticos.\n"
            "Reglas:\n"
            "1. Usa SOLO la información de los documentos provistos o de la herramienta "
            "get_order_status. NUNCA inventes políticas, plazos ni estados de pedido.\n"
            "2. Responde en español, con frases cortas y amables.\n"
            "3. Al final, cita el documento de cada dato con el formato "
            "(Fuente: <título del documento>), p. ej. (Fuente: Política de garantía).\n"
            "4. Si los documentos no responden la pregunta, di que no tienes esa "
            f"información y sugiere escribir a {SUPPORT_EMAIL}.\n"
            "5. NO puedes aprobar reembolsos mayores a $500: dilo y deriva al cliente.\n"
            "6. Quejas sobre el trato de un empleado, disputas de facturación y temas "
            f"legales: NO intentes resolverlos, deriva al cliente a {SUPPORT_EMAIL}.\n"
        )
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
