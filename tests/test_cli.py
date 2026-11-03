"""Tests for the command-line interface (mock LLM, offline retrieval)."""

import pytest

from tiendahogar import cli
from tiendahogar.agent import build_agent
from tiendahogar.config import Settings
from tiendahogar.domain import Action, AgentReply


def test_ask_answers_in_mock_mode(capsys):
    assert cli.main(["ask", "How long is the warranty on a refrigerator?"]) == 0
    out = capsys.readouterr().out.lower()
    assert "mock (offline)" in out
    assert "warranty" in out


def test_ask_answers_in_spanish(capsys):
    assert cli.main(["ask", "--lang", "es", "¿Cuánto dura la garantía?"]) == 0
    assert "(Fuente: Política de" in capsys.readouterr().out


def test_unknown_language_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["ask", "--lang", "fr", "Bonjour"])


def test_no_command_starts_the_chat(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "chat", lambda language: calls.append("chat") or 0)
    assert cli.main([]) == 0
    assert calls == ["chat"]


def test_chat_exits_on_exit_word(monkeypatch, capsys):
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "exit")
    assert cli.main(["chat"]) == 0
    assert "Bye!" in capsys.readouterr().out


def test_chat_remembers_earlier_messages(monkeypatch, capsys):
    answers = iter(["How long is the warranty on a refrigerator?", "And for a blender?", "exit"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: next(answers))
    assert cli.main(["chat"]) == 0
    assert capsys.readouterr().out.count("(Source: Warranty policy)") == 2


def test_ask_requires_a_question():
    with pytest.raises(SystemExit):
        cli.main(["ask"])


def test_version_prints_package_and_settings(monkeypatch, capsys):
    monkeypatch.setattr(cli, "package_version", lambda: "9.9.9")
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "tiendahogar 9.9.9" in out
    assert "retrieval: lexical (BM25)" in out


def test_describe_settings_without_key_shows_the_mock():
    settings = Settings()
    expected = "LLM: mock (offline) · retrieval: lexical (BM25) · rerank: off"
    assert cli.describe_settings(settings, build_agent(settings)) == expected


def test_describe_settings_with_key_shows_the_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    settings = Settings(embedding_model="BAAI/bge-m3", rerank_url="http://localhost:8001/rerank")
    assert cli.describe_settings(settings, build_agent(settings)) == (
        "LLM: groq:openai/gpt-oss-20b · retrieval: semantic (BAAI/bge-m3) · rerank: on"
    )


def test_guardrail_responses_are_shown_in_a_panel(capsys):
    cli.print_response(AgentReply(answer="Please contact support.", action=Action.REFER))
    assert "guardrail: refer" in capsys.readouterr().out


def test_serve_starts_uvicorn_with_the_app(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))
    assert cli.main(["serve", "--no-ui", "--port", "9000"]) == 0
    assert calls["port"] == 9000
    assert calls["host"] == "127.0.0.1"


def test_serve_prints_the_urls(monkeypatch, capsys):
    uvicorn = pytest.importorskip("uvicorn")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    assert cli.main(["serve", "--no-ui"]) == 0
    out = capsys.readouterr().out
    assert "http://localhost:7860/docs" in out
    assert "Web chat" not in out
