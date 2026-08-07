"""CLI del agente TiendaHogar.

Tres modos:
- ``tiendahogar``: TUI interactiva con Textual (default; cae a rich si Textual
  no está instalado).
- ``tiendahogar --plain``: REPL simple con rich.
- ``tiendahogar --query <pregunta>`` (o ``--query=<pregunta>``): responde una
  sola consulta y termina (modo programático/prueba).

Uso: ``tiendahogar``  o  ``python -m src.cli``
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

from src.config import get_settings

from . import cli_rich

__all__ = ["_query_from_argv", "_run_interactive", "main"]


def _query_from_argv(argv: list[str]) -> str | None:
    """Devuelve la consulta de ``--query=...`` o ``--query ...``, si existe."""
    for i, arg in enumerate(argv):
        if arg == "--query":
            if i + 1 < len(argv):
                return argv[i + 1]
        elif arg.startswith("--query="):
            return arg.split("=", 1)[1]
    return None


def _run_interactive(ui: str) -> int:
    """Lanza la TUI de Textual si está disponible; si no, el REPL de rich."""
    if ui == "textual":
        try:
            from .cli_textual import run_textual
        except ImportError:
            return cli_rich.run_rich()
        return run_textual()
    return cli_rich.run_rich()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--version" in argv:
        try:
            print(version("tiendahogar-support-agent"))
        except PackageNotFoundError:
            print("tiendahogar-support-agent (sin versión instalada)")
        print(f"Variante: {cli_rich.variant_label(get_settings())}")
        return 0

    query = _query_from_argv(argv)
    if query is not None:
        return cli_rich.run_once(query)

    if "--plain" in argv:
        return cli_rich.run_rich()

    return _run_interactive(get_settings().ui.lower())


if __name__ == "__main__":
    sys.exit(main())
