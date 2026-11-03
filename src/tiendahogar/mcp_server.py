"""MCP server (FastMCP) that exposes the TiendaHogar order tool over stdio.

FastMCP builds the tool schema from the function signature and docstring.
Run it with ``tiendahogar mcp``.
"""

from __future__ import annotations

from fastmcp import FastMCP

from tiendahogar.tools import get_order_status

mcp = FastMCP("tiendahogar")
mcp.tool(get_order_status)


if __name__ == "__main__":
    mcp.run(transport="stdio")
