"""Agent integration tests: guardrails, retrieval and tool calling.

They use MockLLM (no network) and the offline TF-IDF retriever.
"""

from tiendahogar.agent import SupportAgent
from tiendahogar.llm import MockLLM, ToolCall
from tiendahogar.retrieval import TfidfRetriever


def build_agent(responses: dict[str, str] | None = None) -> SupportAgent:
    return SupportAgent(
        llm=MockLLM(responses=responses),
        retriever=TfidfRetriever(),
        max_turns=2,
        top_k=3,
        relevance_threshold=0.15,
    )


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
        resp = build_agent(
            {"warranty": "Refrigerators have a 12 months warranty (Source: Warranty policy)."}
        ).run("How long is the warranty on a refrigerator?")
        assert resp.action == "answer"
        assert resp.sources
        assert "(Source: Warranty policy)" in resp.answer

    def test_without_context_it_does_not_invent(self):
        resp = build_agent().run("What is the weather like on Mars?")
        assert resp.sources == []
        assert "not" in resp.answer.lower() or "support" in resp.answer.lower()


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
        assert "6 business days" in resp.answer.lower()

    def test_delivered_order_has_no_estimate(self):
        resp = build_agent().run("Where is my order ORD-1002?")
        assert resp.action == "answer"
        assert "delivered" in resp.answer.lower()
        assert "arrive in -" not in resp.answer

    def test_unknown_order_is_reported_as_not_found(self):
        resp = build_agent().run("What is the status of my order ORD-9999?")
        assert resp.action == "answer"
        assert "could not find" in resp.answer.lower()

    def test_the_agent_runs_the_right_tool(self):
        result = build_agent()._execute_tool(
            ToolCall(id="t1", name="get_order_status", arguments={"order_id": "ORD-1001"})
        )
        assert result["order_id"] == "ORD-1001"
        assert result["status"] == "In transit"


def test_the_loop_stops_at_max_turns():
    agent = build_agent()
    assert agent.run("Where is my order ORD-1001?").raw_steps <= agent.max_turns
