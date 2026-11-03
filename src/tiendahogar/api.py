"""HTTP API (and optional Gradio UI) for the support agent.

Needs the ``web`` extra. Endpoints:

- ``POST /chat``: ``{"message": "...", "language": "en" | "es"}`` →
  ``{"answer", "action", "sources"}``. The language defaults to English.
- ``GET /health``: liveness plus the active model.
- ``/``: the Gradio chat UI, when mounted.
"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tiendahogar import __version__
from tiendahogar.agent import SupportAgent, build_agents
from tiendahogar.config import get_settings
from tiendahogar.domain import Action, Language


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    language: Language = Language.EN


class ChatResponse(BaseModel):
    answer: str
    action: Action
    sources: list[int]


class Health(BaseModel):
    status: str = "ok"
    version: str
    model: str


def create_app(
    agents: Mapping[Language, SupportAgent] | None = None, *, with_ui: bool = True
) -> FastAPI:
    """Build the app around one agent per language, shared by every request."""
    if agents is None:
        agents = build_agents()
    model = "mock" if agents[Language.EN].is_mock else get_settings().llm_model
    app = FastAPI(title="TiendaHogar support agent", version=__version__)

    @app.get("/health")
    def health() -> Health:
        return Health(version=__version__, model=model)

    # A sync endpoint: FastAPI runs it in a worker thread, so the blocking
    # agent call never stalls the event loop.
    @app.post("/chat")
    def chat(request: ChatRequest) -> ChatResponse:
        reply = agents[request.language].run(request.message)
        return ChatResponse(answer=reply.answer, action=reply.action, sources=reply.sources)

    if with_ui:
        import gradio as gr

        from tiendahogar.ui import THEME, build_ui

        app = gr.mount_gradio_app(app, build_ui(agents), path="/", theme=THEME)
    return app
