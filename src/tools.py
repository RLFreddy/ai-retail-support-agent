"""Herramienta `consultar_estado_pedido` del agente de soporte TiendaHogar.

Tabla mock de pedidos (hardcodeada, sin API real) según la especificación de la
prueba técnica. Firma exacta requerida: `consultar_estado_pedido(order_id: str) -> dict`.
"""

from __future__ import annotations

from typing import TypedDict


class OrderStatus(TypedDict, total=False):
    """Forma de retorno de la tool."""

    order_id: str
    producto: str
    estado: str
    entrega_estimada: str
    encontrado: bool


#: Tabla mock de pedidos (tal cual la entrega el enunciado).
ORDERS: dict[str, dict[str, str]] = {
    "ORD-1001": {"producto": "Refrigeradora", "estado": "En tránsito", "entrega_estimada": "3 días hábiles"},
    "ORD-1002": {"producto": "Licuadora", "estado": "Entregado", "entrega_estimada": "-"},
    "ORD-1003": {"producto": "Lavadora", "estado": "Procesando", "entrega_estimada": "6 días hábiles"},
    "ORD-1004": {"producto": "Tostadora", "estado": "Cancelado", "entrega_estimada": "-"},
}


def consultar_estado_pedido(order_id: str) -> dict:
    """Consulta el estado de un pedido en la tabla mock.

    Args:
        order_id: Identificador del pedido (p. ej. "ORD-1001").

    Returns:
        dict con el estado del pedido. Si el pedido no existe, retorna un estado
        claro de "no encontrado" (nunca inventa información).

    Ejemplos:
        >>> consultar_estado_pedido("ORD-1001")
        {'order_id': 'ORD-1001', 'producto': 'Refrigeradora', 'estado': 'En tránsito', 'entrega_estimada': '3 días hábiles', 'encontrado': True}

        >>> consultar_estado_pedido("ORD-9999")
        {'order_id': 'ORD-9999', 'encontrado': False, 'estado': 'no encontrado'}
    """
    if not isinstance(order_id, str) or not order_id.strip():
        return {"order_id": order_id, "estado": "no encontrado", "encontrado": False}

    order_id = order_id.strip().upper()
    info = ORDERS.get(order_id)
    if info is None:
        return {"order_id": order_id, "estado": "no encontrado", "encontrado": False}

    return {
        "order_id": order_id,
        **info,
        "encontrado": True,
    }


#: Docstring visible para el LLM (tool-calling).
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "consultar_estado_pedido",
        "description": (
            "Consulta el estado de un pedido en el sistema de TiendaHogar usando su "
            "order_id (p. ej. ORD-1001). Devuelve el producto, estado y entrega estimada. "
            "Si el pedido no existe, 'encontrado' es False."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Identificador del pedido, formato ORD-XXXX.",
                }
            },
            "required": ["order_id"],
        },
    },
}
