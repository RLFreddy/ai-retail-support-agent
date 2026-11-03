"""The ``get_order_status`` tool: looks an order up in a hardcoded mock table."""

from __future__ import annotations

#: Mock order table from the technical test (no real API behind it).
ORDERS: dict[str, dict[str, str]] = {
    "ORD-1001": {
        "product": "Refrigerator",
        "status": "In transit",
        "estimated_delivery": "3 business days",
    },
    "ORD-1002": {"product": "Blender", "status": "Delivered", "estimated_delivery": "-"},
    "ORD-1003": {
        "product": "Washing machine",
        "status": "Processing",
        "estimated_delivery": "6 business days",
    },
    "ORD-1004": {"product": "Toaster", "status": "Cancelled", "estimated_delivery": "-"},
}


#: What the tool returns. The docstring below is the description the LLM sees.
OrderStatus = dict[str, str | bool]


def get_order_status(order_id: str) -> OrderStatus:
    """Look up the status of a TiendaHogar order by its id (format ORD-XXXX).

    Returns the product, status and estimated delivery. When the order does not
    exist, ``found`` is False and no other data is returned.
    """
    order_id = order_id.strip().upper()
    info = ORDERS.get(order_id)
    if info is None:
        return {"order_id": order_id, "status": "not found", "found": False}
    return {"order_id": order_id, **info, "found": True}
