"""Agent integration tests: guardrails, retrieval and tool calling.

They use the offline mock model and the BM25 retriever: no network, no keys.
"""

import logging

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from tiendahogar.agent import SupportAgent, build_model
from tiendahogar.config import Settings
from tiendahogar.mock import mock_model
from tiendahogar.retrieval import LexicalRetriever


def build_agent(model=None) -> SupportAgent:
    return SupportAgent(model=model or mock_model(), retriever=LexicalRetriever())


class TestInputGuardrail:
    def test_refund_over_500_escalates(self):
        resp = build_agent().run("I need a refund of $1200 for my washing machine")
        assert resp.action == "escalate"
        assert "support@tiendahogar.example" in resp.answer

    def test_complaint_about_an_employee_is_referred(self):
        assert build_agent().run("Your agent was rude to me").action == "refer"

    def test_legal_topic_is_referred(self):
        assert build_agent().run("I want legal advice about my contract").action == "refer"


class TestRetrieval:
    def test_warranty_answer_cites_its_source(self):
        resp = build_agent().run("How long is the warranty on a refrigerator?")
        assert resp.action == "answer"
        assert resp.sources == [0]
        assert "(Source: Warranty policy)" in resp.answer

    def test_without_context_it_does_not_invent(self):
        resp = build_agent().run("What is the weather like on Mars?")
        assert resp.sources == []
        assert "do not have that information" in resp.answer


class TestTool:
    def test_existing_order_is_answered(self):
        resp = build_agent().run("Where is my order ORD-1001?")
        assert resp.action == "answer"
        assert "in transit" in resp.answer.lower()

    def test_tool_estimate_is_not_flagged_by_the_output_guardrail(self):
        # ORD-1003 arrives in "6 business days", which is not in the corpus:
        # the output guardrail must accept it as tool data.
        resp = build_agent().run("Where is my order ORD-1003?")
        assert resp.action == "answer"
        assert "6 business days" in resp.answer

    def test_delivered_order_has_no_estimate(self):
        resp = build_agent().run("Where is my order ORD-1002?")
        assert "delivered" in resp.answer.lower()
        assert "arrive" not in resp.answer

    def test_unknown_order_is_reported_as_not_found(self):
        resp = build_agent().run("What is the status of my order ORD-9999?")
        assert resp.action == "answer"
        assert "could not find" in resp.answer

    def test_a_model_stuck_calling_tools_is_stopped(self):
        def loop(messages, info):
            return ModelResponse(parts=[ToolCallPart("get_order_status", {"order_id": "ORD-1"})])

        resp = build_agent(FunctionModel(loop)).run("Where is my order ORD-1001?")
        assert resp.action == "refer"


class TestOutputValidation:
    @staticmethod
    def replay(*answers: str) -> FunctionModel:
        queue = [ModelResponse(parts=[TextPart(a)]) for a in answers]
        return FunctionModel(lambda messages, info: queue.pop(0))

    def test_ungrounded_answer_is_retried_once(self):
        model = self.replay(
            "Refunds take 45 days.", "Refunds take 5-10 business days. (Source: Refunds)"
        )
        resp = build_agent(model).run("How long does a refund take?")
        assert resp.action == "answer"
        assert resp.answer == "Refunds take 5-10 business days. (Source: Refunds)"

    def test_self_approved_large_refund_is_replaced_by_the_escalation(self):
        model = self.replay("I approve your refund of $900 right away.")
        resp = build_agent(model).run("How long does a refund take?")
        assert resp.action == "escalate"
        assert "approve" not in resp.answer


class TestModelSelection:
    def test_mock_by_name(self):
        assert build_model("mock").model_name == "mock"

    def test_missing_api_key_falls_back_to_mock_with_a_warning(self, caplog):
        caplog.set_level(logging.WARNING)
        assert build_model("groq:openai/gpt-oss-20b").model_name == "mock"
        assert "GROQ_API_KEY" in caplog.text

    def test_provider_key_selects_the_real_model(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        assert build_model("groq:openai/gpt-oss-20b").model_name == "openai/gpt-oss-20b"

    def test_model_name_without_provider_is_rejected(self):
        with pytest.raises(ValueError, match="provider:model"):
            Settings(llm_model="groq/openai/gpt-oss-20b")
