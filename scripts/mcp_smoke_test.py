#!/usr/bin/env python3
"""Smoke test of the TiendaHogar MCP server.

Starts the real server over stdio as a subprocess, connects as an MCP client,
lists the tools and calls ``get_order_status`` with an existing order
(ORD-1002) and an unknown one (ORD-9999).

Usage:
    uv run python scripts/mcp_smoke_test.py

Exit code 0 when everything works, 1 otherwise.
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TOOL = "get_order_status"
CASES = (("ORD-1002", True), ("ORD-9999", False))


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=["-m", "tiendahogar.mcp_server"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = [t.name for t in (await session.list_tools()).tools]
        print(f"tools: {', '.join(names)}")
        if TOOL not in names:
            print(f"FAIL: tool {TOOL} not found")
            return 1

        failures = 0
        for order_id, expected in CASES:
            result = await session.call_tool(TOOL, {"order_id": order_id})
            text = "".join(getattr(block, "text", "") for block in result.content)
            found = json.loads(text).get("found")
            ok = found is expected
            failures += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {order_id}: {text.strip()}")

    print("smoke test passed" if failures == 0 else f"smoke test: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
