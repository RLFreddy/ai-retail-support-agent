"""Base de conocimiento de TiendaHogar.

Contiene los 5 documentos de política entregados en la prueba técnica, usados
TAL CUAL (sin editar). Cada documento se modela como un chunk único: son fuentes
cortas (~1 párrafo), por lo que el chunking fino no aporta en este corpus.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Un documento de la base de conocimiento."""

    id: str
    title: str
    content: str


#: Los 5 documentos — textos exactos de la especificación.
DOCUMENTS: list[Document] = [
    Document(
        id="garantia",
        title="Política de garantía",
        content=(
            "Todos los electrodomésticos grandes (refrigeradoras, lavadoras, estufas) "
            "tienen garantía de 12 meses desde la fecha de compra. Electrodomésticos "
            "pequeños (licuadoras, planchas, tostadoras) tienen garantía de 6 meses. "
            "La garantía cubre defectos de fábrica, no daños por mal uso."
        ),
    ),
    Document(
        id="devoluciones",
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
        id="envios",
        title="Tiempos de envío",
        content=(
            "Envíos a la capital: 2-3 días hábiles. Envíos a otras ciudades: 5-7 días "
            "hábiles. Envíos internacionales no están disponibles actualmente."
        ),
    ),
    Document(
        id="reembolsos",
        title="Reembolsos",
        content=(
            "Los reembolsos se procesan en 5-10 días hábiles después de recibir el "
            "producto devuelto. Se reembolsa al mismo método de pago original. "
            "Reembolsos mayores a $500 requieren aprobación de un supervisor humano — "
            "el agente no debe aprobarlos automáticamente."
        ),
    ),
    Document(
        id="contacto",
        title="Canales de contacto",
        content=(
            "Para quejas sobre el trato de un empleado, disputas de facturación, o "
            "cualquier tema legal, el cliente debe ser referido a un agente humano en "
            "soporte@tiendahogar.example — el asistente de IA no debe intentar resolver "
            "estos casos."
        ),
    ),
]

def default_system_prompt() -> str:
    """Prompt de sistema que persigue grounding estricto (no inventar)."""
    return (
        "Eres el asistente de soporte de TiendaHogar, una tienda de electrodomésticos.\n"
        "Reglas:\n"
        "1. Usa SOLO la información de los documentos provistos o de la tool "
        "consultar_estado_pedido. NUNCA inventes políticas, plazos ni estados de pedidos.\n"
        "2. Responde en español, con frases breves y amables.\n"
        "3. Cita al final, con el formato (Fuente: <título del documento>), el documento "
        "del que tomaste cada dato (p. ej. (Fuente: Política de garantía)).\n"
        "4. Si la pregunta no tiene respuesta en los documentos, responde que no tienes "
        "esa información y sugiere contactar soporte@tiendahogar.example.\n"
        "5. Reembolsos mayores a $500 NO pueden ser aprobados por ti: indícalo y deriva.\n"
        "6. Quejas de trato de empleados, disputas de facturación y temas legales: "
        "NO intentes resolverlos, refiere al cliente a soporte@tiendahogar.example.\n"
    )