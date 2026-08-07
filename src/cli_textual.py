"""TUI interactiva del agente con Textual."""

from __future__ import annotations

import sys

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog

from src.agent import build_agent
from src.cli_rich import render_variant
from src.config import get_settings


class SupportChatApp(App):
    """Interfaz de chat para el agente TiendaHogar."""

    TITLE = "TiendaHogar — Agente de Soporte"

    def __init__(self, agent=None, settings=None) -> None:
        super().__init__()
        if settings is None:
            settings = get_settings()
        self._settings = settings
        if agent is None:
            agent = build_agent(settings)
        self.agent = agent

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, markup=True, wrap=True, id="chat")
        yield Input(placeholder="Escribe tu consulta… (o 'salir')", id="msg")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat", RichLog)
        chat.write(
            "[b cyan]TiendaHogar[/b cyan] Hola, soy el asistente de soporte. "
            "Pregúntame sobre garantías, devoluciones, envíos, reembolsos o el "
            "estado de tu pedido (ORD-XXXX)."
        )
        chat.write(render_variant(self._settings))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        chat = self.query_one("#chat", RichLog)
        inp = event.input
        text = event.value.strip()
        if not text:
            return
        if text.lower() in {"salir", "exit", "quit", "q"}:
            self.exit()
            return
        inp.value = ""
        chat.write(f"[b]Tú:[/b] {text}")

        resp = self.agent.run(text)
        color = {
            "answer": "",
            "refer": " [yellow](derivado a soporte humano)[/yellow]",
            "escalate": " [red](escalado a supervisor)[/red]",
            "block": " [red](bloqueado)[/red]",
            "off_topic": " [dim](fuera de tema)[/dim]",
        }.get(resp.action, "")
        chat.write(f"[b cyan]TiendaHogar:[/b cyan] {resp.answer}{color}")
        if resp.sources and resp.action == "answer":
            chat.write(f"[dim]Fuentes: {resp.sources}[/dim]")


def run_textual() -> int:
    SupportChatApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(run_textual())