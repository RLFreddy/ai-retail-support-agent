#!/usr/bin/env python3
"""Sonda (smoke test) del servidor MCP de TiendaHogar.

Levanta el servidor real (``uv run python -m tiendahogar.mcp_server``) como subproceso
con transporte stdio, se conecta como cliente MCP, lista las tools expuestas y
hace dos llamadas: un pedido válido (ORD-1002) y uno inexistente (ORD-9999).

Uso:
    uv run python scripts/mcp_smoke_test.py

Salida en español; código de salida 0 = todo bien, 1 = algo falló.
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp.client.stdio import StdioServerParameters, stdio_client

ORDER_OK = "ORD-1002"
ORDER_BAD = "ORD-9999"


def _ordenar(ids: list[str]) -> str:
    return ", ".join(ids)


async def main() -> int:
    parametros = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tiendahogar.mcp_server"],
        env=None,
    )

    async with stdio_client(parametros) as (lectura, escritura):
        from mcp import ClientSession

        async with ClientSession(lectura, escritura) as sesion:
            await sesion.initialize()
            print(f"[1/3] Conexión stdio establecida -> {parametros.command} {' '.join(parametros.args)}")

            tools = await sesion.list_tools()
            nombres = [t.name for t in tools.tools]
            print(f"[2/3] Tools expuestas ({len(nombres)}): {_ordenar(nombres)}")
            if "consultar_estado_pedido_mcp" not in nombres:
                print("ERROR: no se encontró la tool consultar_estado_pedido_mcp")
                return 1

            fallos = 0

            for order_id, esperado in ((ORDER_OK, True), (ORDER_BAD, False)):
                resultado = await sesion.call_tool("consultar_estado_pedido_mcp", {"order_id": order_id})
                texto = "".join(bloque.text or "" for bloque in resultado.content)
                try:
                    payload = json.loads(texto)
                except json.JSONDecodeError:
                    payload = {"respuesta_bruta": texto}
                encontrado = payload.get("encontrado", payload.get("respuesta_bruta"))
                ok = bool(encontrado) == esperado
                estado = "OK" if ok else "FALLO"
                print(f"[3/3] Llamada {order_id}: {estado}")
                print(f"      -> {texto.strip()}")
                if not ok:
                    fallos += 1

            print("-" * 60)
            if fallos == 0:
                print("SMOKE TEST: TODO OK")
                return 0
            print(f"SMOKE TEST: {fallos} fallo(s)")
            return 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:  # noqa: BLE001 - reporta y sale con error
        print(f"ERROR: {exc!r}")
        raise SystemExit(1)
