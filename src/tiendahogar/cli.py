"""Command-line interface of the TiendaHogar support agent.

Commands:

- ``tiendahogar chat``: interactive chat in the terminal (the default).
- ``tiendahogar ask "<question>"``: answer one question and exit.
- ``tiendahogar mcp``: run the MCP server over stdio.
- ``tiendahogar --version``: print the version and the active configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from tiendahogar.agent import SupportAgent, build_agent
from tiendahogar.config import Settings, get_settings
from tiendahogar.domain import Action, AgentReply

console = Console()

EXIT_WORDS = {"exit", "quit", "q"}
ACTION_STYLES = {
    Action.ANSWER: "green", Action.REFER: "yellow", Action.ESCALATE: "red", Action.BLOCK: "red",
}


def describe_settings(settings: Settings, agent: SupportAgent) -> str:
    """One line with the active model, retrieval backend and reranking."""
    llm = "mock (offline)" if agent.is_mock else settings.llm_model
    retrieval = (
        f"semantic ({settings.embedding_model})" if settings.embedding_model else "lexical (BM25)"
    )
    rerank = "on" if settings.rerank_url else "off"
    return f"LLM: {llm} · retrieval: {retrieval} · rerank: {rerank}"


def print_response(resp: AgentReply) -> None:
    """Print an agent response. Agent text is escaped: it is never rich markup."""
    style = ACTION_STYLES[resp.action]
    if resp.action is Action.ANSWER:
        console.print(f"[{style}]TiendaHogar[/{style}] {escape(resp.answer)}")
        if resp.sources:
            console.print(f"[dim]Sources: {resp.sources}[/dim]")
    else:
        console.print(
            Panel(escape(resp.answer), title=f"guardrail: {resp.action}", border_style=style)
        )


def chat() -> int:
    settings = get_settings()
    agent = build_agent(settings)
    console.print(
        Panel(
            "[bold]TiendaHogar support agent[/bold]\n"
            "Type your question, or 'exit' to quit.\n"
            f"[dim]{escape(describe_settings(settings, agent))}[/dim]",
            border_style="cyan",
        )
    )
    while True:
        try:
            message = Prompt.ask("[bold]You[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            return 0
        if message.strip().lower() in EXIT_WORDS:
            console.print("[dim]Bye![/dim]")
            return 0
        print_response(agent.run(message))


def ask(question: str) -> int:
    settings = get_settings()
    agent = build_agent(settings)
    console.print(f"[dim]{escape(describe_settings(settings, agent))}[/dim]")
    print_response(agent.run(question))
    return 0


def serve_mcp() -> int:
    from tiendahogar.mcp_server import mcp

    mcp.run(transport="stdio")
    return 0


def package_version() -> str:
    try:
        return version("tiendahogar-support-agent")
    except PackageNotFoundError:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiendahogar", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="store_true", help="print version and exit")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("chat", help="interactive chat (default)")
    ask_parser = commands.add_parser("ask", help="answer one question and exit")
    ask_parser.add_argument("question")
    commands.add_parser("mcp", help="run the MCP server over stdio")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Provider API keys are read from the environment by Pydantic AI.
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.version:
        settings = get_settings()
        print(f"tiendahogar {package_version()}")
        print(describe_settings(settings, build_agent(settings)))
        return 0
    if args.command == "ask":
        return ask(args.question)
    if args.command == "mcp":
        return serve_mcp()
    return chat()


if __name__ == "__main__":
    sys.exit(main())
