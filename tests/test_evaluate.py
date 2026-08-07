"""El golden set del evaluador (bonus) sigue pasando al 100% (MockLLM + TF-IDF).

``python -m src.evaluate`` no depende de red ni de API keys: usa MockLLM y el
retriever TF-IDF offline, por lo que las métricas son deterministas. Este test
protege el harness para que cualquier cambio en retrieval/agente se refleje aquí.
"""

from src.evaluate import GOLDEN_SET, evaluate


def test_golden_set_tiene_casos_de_retrieval_y_guardrail():
    assert len(GOLDEN_SET) >= 7
    assert any(case["expected_docs"] for case in GOLDEN_SET)  # casos retrieval
    assert any(not case["expected_docs"] for case in GOLDEN_SET)  # guardrail/off-topic


def test_evaluate_devuelve_metricas_dict_100():
    metrics = evaluate(verbose=False)
    assert set(metrics) == {"retrieval@1", "guardrail_accuracy", "grounding_citation"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0