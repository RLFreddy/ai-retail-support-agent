"""REPL interactivo con rich (fallback si Textual no está disponible)."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.agent import build_agent
from src.config import Settings, get_settings

console = Console()


def variant_parts(settings: Settings) -> dict[str, str]:
    """Partes de la variante activa (desde .env) en el formato de los benchmarks.

    Fuente única de verdad: de aquí derivan ``variant_label`` (línea completa),
    ``variant_short`` (línea compacta para TUI) y ``render_variant`` (panel).
    """
    llm = "mock (sin red)" if not settings.has_llm_key else f"{settings.llm_model} (real)"
    emb_labels = {
        "offline": "offline (TF-IDF)",
        "tei": f"tei ({settings.tei_embed_model})",
        "openrouter": f"openrouter ({settings.openrouter_embed_model})",
        "openai": f"openai ({settings.openai_embed_model})",
        "gemini": f"gemini ({settings.gemini_embed_model})",
    }
    emb = emb_labels.get(settings.embedding_backend, settings.embedding_backend)
    rerank = f"on ({settings.rerank_model})" if settings.rerank_enabled else "off"

    is_groq = settings.llm_model.startswith("groq/")
    if not settings.has_llm_key and settings.embedding_backend == "offline":
        slug = "mock_baseline"
    elif is_groq and settings.embedding_backend == "offline" and not settings.rerank_enabled:
        slug = "groq_tfidf"
    elif is_groq and settings.embedding_backend == "tei" and settings.rerank_enabled:
        slug = "groq_tei_rerank"
    elif is_groq and settings.embedding_backend == "tei" and not settings.rerank_enabled:
        slug = "groq_tei_no_rerank"
    else:
        slug = None

    return {"slug": slug, "llm": llm, "emb": emb, "rerank": rerank}


def variant_label(settings: Settings) -> str:
    """Línea descriptiva completa de la variante activa."""
    p = variant_parts(settings)
    descripcion = f"LLM: {p['llm']} · embeddings: {p['emb']} · rerank: {p['rerank']}"
    return f"{p['slug']}  ·  {descripcion}" if p["slug"] else descripcion


def variant_short(settings: Settings) -> str:
    """Línea compacta de la variante activa (para headers de una sola línea)."""
    p = variant_parts(settings)
    compacto = f"{p['llm']} + {p['emb']} · rerank {p['rerank']}"
    return f"{p['slug']} · {compacto}" if p["slug"] else compacto


def render_variant(settings: Settings) -> Panel:
    """Panel rich con la variante activa en forma de tabla."""
    p = variant_parts(settings)
    tabla = Table(show_header=False, box=None, padding=(0, 1))
    tabla.add_column(style="bold cyan")
    tabla.add_column()
    tabla.add_row("Variante", p["slug"] or "—")
    tabla.add_row("LLM", p["llm"])
    tabla.add_row("Embeddings", p["emb"])
    tabla.add_row("Rerank", p["rerank"])
    return Panel(tabla, title="Variante activa", border_style="cyan")


def run_rich() -> int:
    settings = get_settings()
    agent = build_agent(settings)

    console.print(
        Panel(
            "[bold]TiendaHogar — Agente de Soporte[/bold]\n"
            "Escribe tu consulta o 'salir' para terminar.",
            border_style="cyan",
        )
    )
    console.print(render_variant(settings))

    while True:
        try:
            user = Prompt.ask("[bold]Tú[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]¡Hasta luego![/dim]")
            return 0
        if user.strip().lower() in {"salir", "exit", "quit", "q"}:
            console.print("[dim]¡Hasta luego![/dim]")
            return 0

        resp = agent.run(user)

        action_color = {
            "answer": "green",
            "refer": "yellow",
            "escalate": "red",
            "block": "red",
            "off_topic": "dim",
        }.get(resp.action, "white")
        if resp.action != "answer":
            console.print(
                Panel(
                    f"[{action_color}]{resp.answer}[/{action_color}]",
                    title=f"guardrail: {resp.action}",
                    border_style=action_color,
                )
            )
        else:
            console.print(f"[{action_color}]TiendaHogar[/{action_color}] {resp.answer}")
        if resp.sources and resp.action == "answer":
            console.print(f"[dim]Fuentes: {resp.sources}[/dim]")
    return 0


def run_once(query: str) -> int:
    """Responde una única consulta y termina (modo programático/prueba)."""
    settings = get_settings()
    agent = build_agent(settings)
    console.print(render_variant(settings))
    resp = agent.run(query)
    console.print(resp.answer)
    if resp.action != "answer":
        console.print(f"  [dim](guardrail: {resp.action})[/dim]")
    if resp.sources:
        console.print(f"  [dim]Fuentes: {resp.sources}[/dim]")
    return 0