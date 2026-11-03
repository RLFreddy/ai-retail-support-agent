"""Tests del dispatch de la CLI.

Cubre el parseo de ``--query`` (ambas formas), ``--version``, la etiqueta de
variante activa, y la regresión del bug por el que ``UI=rich`` (o un fallo al
importar Textual) producía ``UnboundLocalError``.

Los tests son 100% mocks: corren con ``MockLLM + TfidfRetriever``, sin red y
sin leer el ``.env`` real. Las 4 variantes de la prueba técnica solo se prueban
como FUNCIÓN PURA (``Settings`` -> etiqueta/panel); el comportamiento real de
cada variante se mide con los evals de ``evals/``, no aquí.
"""

import asyncio
import io
import sys

import pytest
from rich.console import Console
from textual.widgets import RichLog

from src import cli
from src.cli_rich import render_variant, variant_label, variant_short
from src.cli_textual import SupportChatApp
from src.config import Settings


class _FakeSettings:
    ui = "rich"


def _settings_sin_key() -> Settings:
    """Settings sin ninguna API key: fuerza MockLLM, determinista y sin red."""
    return Settings(groq_api_key="", openrouter_api_key="", openai_api_key="",
                    anthropic_api_key="", gemini_api_key="")


#: Las 4 configuraciones de la prueba técnica, SOLO para la lógica pura de
#: etiquetado/panel. Las keys son falsas: aquí no se ejecuta ni Groq ni TEI,
#: solo se verifica qué slug/panel produce cada Settings.
VARIANTS = [
    pytest.param("mock_baseline", _settings_sin_key(), id="mock_baseline"),
    pytest.param(
        "groq_tfidf",
        Settings(groq_api_key="gsk-x", embedding_backend="offline"),
        id="groq_tfidf",
    ),
    pytest.param(
        "groq_tei_rerank",
        Settings(groq_api_key="gsk-x", embedding_backend="tei", rerank_enabled=True),
        id="groq_tei_rerank",
    ),
    pytest.param(
        "groq_tei_no_rerank",
        Settings(groq_api_key="gsk-x", embedding_backend="tei", rerank_enabled=False),
        id="groq_tei_no_rerank",
    ),
]


def test_variant_label_mock_baseline():
    etiqueta = variant_label(_settings_sin_key())
    assert etiqueta.startswith("mock_baseline")
    assert "mock (sin red)" in etiqueta and "TF-IDF" in etiqueta and "rerank: off" in etiqueta


def test_variant_label_groq_tfidf():
    etiqueta = variant_label(Settings(groq_api_key="gsk-x", embedding_backend="offline"))
    assert etiqueta.startswith("groq_tfidf")
    assert "groq/" in etiqueta and "TF-IDF" in etiqueta and "rerank: off" in etiqueta


def test_variant_label_groq_tei_rerank():
    etiqueta = variant_label(
        Settings(groq_api_key="gsk-x", embedding_backend="tei", rerank_enabled=True)
    )
    assert etiqueta.startswith("groq_tei_rerank")
    assert "tei (" in etiqueta and "rerank: on" in etiqueta


def test_variant_label_groq_tei_no_rerank():
    etiqueta = variant_label(
        Settings(groq_api_key="gsk-x", embedding_backend="tei", rerank_enabled=False)
    )
    assert etiqueta.startswith("groq_tei_no_rerank")
    assert "tei (" in etiqueta and "rerank: off" in etiqueta


def test_variant_label_fallback_sin_slug():
    etiqueta = variant_label(
        Settings(openai_api_key="sk-x", llm_model="openai/gpt-4o-mini",
                 embedding_backend="gemini")
    )
    assert not etiqueta.startswith(("mock_baseline", "groq_"))
    assert "openai/gpt-4o-mini" in etiqueta and "gemini" in etiqueta


@pytest.mark.parametrize("slug,settings", VARIANTS)
def test_render_variant_es_panel_con_tabla(slug, settings):
    # Lógica pura: dado un Settings, el panel muestra su slug. La salida se
    # aísla en un StringIO para que nada se filtre al reporte de pytest.
    panel = render_variant(settings)
    assert panel.title == "Variante activa"
    buf = io.StringIO()
    Console(file=buf, width=60).print(panel)
    texto = buf.getvalue()
    assert slug in texto
    assert "LLM" in texto and "Embeddings" in texto and "Rerank" in texto


def test_variant_short_es_compacto():
    corta = variant_short(_settings_sin_key())
    assert corta.startswith("mock_baseline")
    assert "+" in corta and "rerank off" in corta


def test_query_desde_espacio():
    assert cli._query_from_argv(["--query", "hola"]) == "hola"


def test_query_desde_igual():
    assert cli._query_from_argv(["--query=hola"]) == "hola"


def test_query_sin_valor_devuelve_none():
    assert cli._query_from_argv(["--query"]) is None
    assert cli._query_from_argv([]) is None


def test_main_query_responde_en_modo_mock(capsys, monkeypatch):
    # Sin API key: MockLLM + TfidfRetriever, determinista y sin red.
    # (se fuerza para no depender de la red ni de las keys del .env)
    monkeypatch.setattr(cli.cli_rich, "get_settings", _settings_sin_key)
    rc = cli.main(["--query", "¿Cuánto dura la garantía de una refrigeradora?"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "garantía" in out or "garantia" in out


def test_main_query_desde_igual(capsys, monkeypatch):
    monkeypatch.setattr(cli.cli_rich, "get_settings", _settings_sin_key)
    assert cli.main(["--query=¿Cuánto dura la garantía?"]) == 0
    assert capsys.readouterr().out


def test_main_plain_despacha_a_rich(monkeypatch):
    llamadas = []
    monkeypatch.setattr(cli.cli_rich, "run_rich", lambda: llamadas.append(1) or 0)
    assert cli.main(["--plain"]) == 0
    assert llamadas == [1]


def test_regresion_ui_rich_no_crash(monkeypatch):
    # Bug: UI=rich producía UnboundLocalError; ahora debe despachar a rich.
    llamadas = []
    monkeypatch.setattr(cli, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(cli.cli_rich, "run_rich", lambda: llamadas.append(1) or 0)
    assert cli.main([]) == 0
    assert llamadas == [1]


def test_textual_fallback_a_rich_si_import_falla(monkeypatch):
    llamadas = []
    monkeypatch.setattr(cli.cli_rich, "run_rich", lambda: llamadas.append(1) or 0)
    monkeypatch.setitem(sys.modules, "src.cli_textual", None)
    assert cli._run_interactive("textual") == 0
    assert llamadas == [1]


def test_tui_sin_subtitulo_redundante():
    app = SupportChatApp(settings=_settings_sin_key())
    assert app.sub_title == ""


def test_tui_muestra_panel_variante_en_chat():
    app = SupportChatApp(settings=_settings_sin_key())

    async def _leer_chat():
        async with app.run_test():
            chat = app.query_one("#chat", RichLog)
            return [linea.text for linea in chat.lines]

    lineas = asyncio.run(_leer_chat())
    assert any("mock_baseline" in linea for linea in lineas)
    assert any("Variante activa" in linea for linea in lineas)


def test_version(monkeypatch, capsys):
    monkeypatch.setattr(cli, "version", lambda *a, **k: "0.1.0-test")
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "0.1.0" in out
    assert "Variante:" in out