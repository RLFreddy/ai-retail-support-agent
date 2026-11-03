"""MCP server tests (no network: only the tool registry)."""

import asyncio

from tiendahogar.mcp_server import mcp


def _tool():
    return asyncio.run(mcp.get_tool("get_order_status"))


def test_server_exposes_the_order_tool():
    assert {t.name for t in asyncio.run(mcp.list_tools())} == {"get_order_status"}


def test_tool_answers_an_existing_order():
    result = _tool().fn(order_id="ORD-1001")
    assert result["found"] is True
    assert result["product"] == "Refrigerator"


def test_tool_does_not_invent_an_unknown_order():
    result = _tool().fn(order_id="ORD-9999")
    assert result["found"] is False
    assert "product" not in result
