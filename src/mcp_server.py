"""Servidor MCP (FastMCP) que expone las tools de TiendaHogar.

Los servidores MCP consumen un JSON-RPC por stdio (o SSE); el agente queda
independiente: FastMCP genera el schema de la tool desde la firma/docstring.
Ejecutar: ``uv run python -m src.mcp_server``
"""

from __future__ import annotations

from fastmcp import FastMCP

from src.tools import consultar_estado_pedido

mcp = FastMCP("tiendahogar")


@mcp.tool()
def consultar_estado_pedido_mcp(order_id: str) -> dict:
    """Consulta el estado de un pedido de TiendaHogar por su order_id (ORD-XXXX).

    Args:
        order_id: Identificador del pedido, formato ORD-XXXX.

    Returns:
        dict con producto, estado y entrega estimada; ``encontrado`` es False
        si el pedido no existe.
    """
    return consultar_estado_pedido(order_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
