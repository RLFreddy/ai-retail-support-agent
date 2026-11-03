"""TiendaHogar customer support agent: RAG, tool calling and guardrails."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tiendahogar-support-agent")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"
