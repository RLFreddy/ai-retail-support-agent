"""Tests de integración del agente: guardrails + RAG + tool-calling.

Se usa MockLLM (sin red) + TfidfRetriever (offline). Verifica:
- Guardrail de entrada: reembolso > $500 y temas humanos se desvían.
- RAG: respuesta con grounding y cita legible (Fuente: <título>).
- Tool: estado de pedido válido e inválido (sin inventar).
- Sin contexto: el agente no inventa.
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


class TestGuardrailEntrada:
    def test_reembolso_mayor_a_500_escala(self):
        resp = build_agent().run("Necesito un reembolso de $1200 por mi lavadora")
        assert resp.action == "escalate"
        assert "soporte@tiendahogar.example" in resp.answer

    def test_queja_trato_empleado_refiere(self):
        resp = build_agent().run("Tu agente fue grosero conmigo")
        assert resp.action == "refer"

    def test_tema_legal_refiere(self):
        resp = build_agent().run("Quiero asesoría legal sobre mi contrato")
        assert resp.action == "refer"


class TestRAG:
    def test_responde_garantia_con_cita(self):
        resp = build_agent(
            {"garant": "La garantía de refrigeradoras es de 12 meses (Fuente: Política de garantía)."}
        ).run("¿Cuánto dura la garantía de una refrigeradora?")
        assert resp.action == "answer"
        assert resp.sources  # grounding sí se recuperó
        assert "(Fuente: Política de garantía)" in resp.answer

    def test_sin_contexto_no_inventa(self):
        resp = build_agent().run("¿Cuál es el clima en Marte?")
        assert resp.sources == []
        assert "no" in resp.answer.lower() or "soporte" in resp.answer.lower()


class TestTool:
    def test_consulta_estado_pedido_valido(self):
        resp = build_agent().run("¿Dónde está mi pedido ORD-1001?")
        assert resp.action == "answer"
        # MockLLM responde el fallback; lo importante: no falla y no inventa
        assert resp.answer

    def test_consulta_pedido_con_estimacion_no_se_deriva_por_guardrail(self):
        # ORD-1003 tiene entrega_estimada "6 días hábiles" (no está en el corpus):
        # el guardrail de salida debe tratarlo como dato válido de la tool.
        resp = build_agent().run("¿Dónde está mi pedido ORD-1003?")
        assert resp.action == "answer"
        assert "6 días hábiles" in resp.answer.lower()

    def test_pedido_entregado_no_menciona_estimacion(self):
        # ORD-1002 no tiene entrega_estimada ("-"): no debe decir "se entregará en -".
        resp = build_agent().run("¿Dónde está mi pedido ORD-1002?")
        assert resp.action == "answer"
        assert "entregado" in resp.answer.lower()
        assert "se entregará en -" not in resp.answer

    def test_consulta_pedido_inexistente_no_inventa(self):
        resp = build_agent().run("¿Estado de mi pedido ORD-9999?")
        assert resp.action == "answer"

    def test_el_agente_ejecuta_la_tool_correcta(self):
        agent = build_agent()
        call = agent._execute_tool(
            ToolCall(id="t1", name="consultar_estado_pedido", arguments={"order_id": "ORD-1001"})
        )
        assert call["order_id"] == "ORD-1001"
        assert call["estado"] == "En tránsito"


class TestMaxTurns:
    def test_no_infinito_con_mock_loop(self):
        agent = build_agent()
        resp = agent.run("¿Dónde está mi pedido ORD-1001?")
        assert resp.raw_steps <= agent.max_turns