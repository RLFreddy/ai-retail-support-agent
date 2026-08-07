"""Tests del servidor MCP (sin red: solo el registro de la tool).

Verifica que el servidor FastMCP expone `consultar_estado_pedido` y que su
ejecución delega en la tabla mock sin inventar información.
"""

import asyncio

from src.mcp_server import mcp


def _tools():
    return asyncio.run(mcp.list_tools())


def test_servidor_mcp_expone_la_tool():
    tool_names = {t.name for t in _tools()}
    assert "consultar_estado_pedido_mcp" in tool_names


def test_tool_mcp_responde_correctamente():
    tool = _tools()[0]
    result = tool.fn(order_id="ORD-1001")
    assert result["encontrado"] is True
    assert result["producto"] == "Refrigeradora"


def test_tool_mcp_no_inventa_pedido_inexistente():
    tool = _tools()[0]
    result = tool.fn(order_id="ORD-9999")
    assert result["encontrado"] is False
    assert "producto" not in result
