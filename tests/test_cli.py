"""Tests for the command-line interface (mock LLM, offline retrieval)."""

import pytest

from tiendahogar import cli
from tiendahogar.agent import AgentResponse
from tiendahogar.config import Settings


def test_ask_answers_in_mock_mode(capsys):
    assert cli.main(["ask", "How long is the warranty on a refrigerator?"]) == 0
    out = capsys.readouterr().out.lower()
    assert "mock (no api key)" in out
    assert "warranty" in out


def test_no_command_starts_the_chat(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "chat", lambda: calls.append("chat") or 0)
    assert cli.main([]) == 0
    assert calls == ["chat"]


def test_chat_exits_on_exit_word(monkeypatch, capsys):
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "exit")
    assert cli.main(["chat"]) == 0
    assert "Bye!" in capsys.readouterr().out


def test_ask_requires_a_question():
    with pytest.raises(SystemExit):
        cli.main(["ask"])


def test_version_prints_package_and_settings(monkeypatch, capsys):
    monkeypatch.setattr(cli, "package_version", lambda: "9.9.9")
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "tiendahogar 9.9.9" in out
    assert "retrieval: lexical (BM25)" in out


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "LLM: mock (no API key) · retrieval: lexical (BM25) · rerank: off"),
        (
            {"groq_api_key": "gsk-x", "embedding_model": "BAAI/bge-m3",
             "rerank_url": "http://localhost:8001/rerank"},
            "LLM: groq/openai/gpt-oss-20b · retrieval: semantic (BAAI/bge-m3) · rerank: on",
        ),
    ],
)
def test_describe_settings(overrides, expected):
    # Settings are built inside the test, after the hermetic fixture ran.
    assert cli.describe_settings(Settings(**overrides)) == expected


def test_guardrail_responses_are_shown_in_a_panel(capsys):
    cli.print_response(AgentResponse(answer="Please contact support.", action="refer"))
    assert "guardrail: refer" in capsys.readouterr().out
