"""Evaluación del agente con un "golden set" manual (bonus).

Crea un pequeño conjunto de casos etiquetados (pregunta -> doc_id esperado /
acción esperada) y mide:
- ``retrieval@1``: ¿el doc esperado aparece en las fuentes?
- ``guardrail``: ¿la acción de salida coincide con la esperada?
- ``grounding``: ¿la respuesta cita (Fuente: …) cuando hay context?

Es un harness propio (sin RAGAS/DeepEval) para no inflar dependencias en la
prueba técnica, pero con la misma idea: métricas de retrieval + agente.

Uso:
    uv run python -m src.evaluate
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from src.agent import SupportAgent
from src.knowledge import DOCUMENTS
from src.llm import MockLLM
from src.retrieval import TfidfRetriever

_console = Console()

#: Acciones del agente en pasado (columnas de las tablas, en español).
_PAST_ES: dict[str, str] = {
    "answer": "respondió",
    "refer": "derivó",
    "escalate": "escaló",
    "block": "bloqueó",
}


def _doc_titles(indices: list[int]) -> str:
    """Títulos de los documentos por índice (fuente única: DOCUMENTS)."""
    return ", ".join(DOCUMENTS[i].title for i in indices if 0 <= i < len(DOCUMENTS))


def _category(has_expected_docs: bool, expected_action: str) -> str:
    """A qué tipo de caso pertenece: retrieval / guardrail / off-topic."""
    if has_expected_docs:
        return "retrieval"
    if expected_action == "answer":
        return "off-topic"
    return "guardrail"


GOLDEN_SET: list[dict] = [
    {
        "query": "¿Cuánto dura la garantía de una refrigeradora?",
        "expected_docs": {0},
        "expected_action": "answer",
        "expect_citation": True,
    },
    {
        "query": "¿Cuántos días para devolver un producto sin usar?",
        "expected_docs": {1},
        "expected_action": "answer",
        "expect_citation": True,
    },
    {
        "query": "¿A qué ciudades llegan los envíos?",
        "expected_docs": {2},
        "expected_action": "answer",
        "expect_citation": True,
    },
    {
        "query": "¿En cuántos días se reembolsa?",
        "expected_docs": {3},
        "expected_action": "answer",
        "expect_citation": True,
    },
    {
        "query": "Necesito un reembolso de $1200 por mi lavadora",
        "expected_docs": set(),
        "expected_action": "escalate",
        "expect_citation": False,
    },
    {
        "query": "Tu agente fue grosero conmigo en el chat",
        "expected_docs": set(),
        "expected_action": "refer",
        "expect_citation": False,
    },
    {
        "query": "¿Cuál es el clima de Marte?",
        "expected_docs": set(),
        "expected_action": "answer",
        "expect_citation": False,
    },
]


def evaluate(retriever_class=TfidfRetriever, verbose: bool = True) -> dict:
    """Ejecuta el golden set y devuelve métricas agregadas."""
    agent = SupportAgent(
        llm=MockLLM(), retriever=retriever_class(), max_turns=2
    )

    retrieval_hits = 0
    action_hits = 0
    citation_correct = 0
    total = len(GOLDEN_SET)

    rows: list[dict] = []
    for case in GOLDEN_SET:
        resp = agent.run(case["query"])
        got_docs = set(resp.sources)
        if case["expected_docs"]:
            r_hit = bool(got_docs & case["expected_docs"])
        else:
            # Casos de guardrail: correcto si NO recupera documentos.
            r_hit = not got_docs
        a_hit = resp.action == case["expected_action"]
        # cita: si se espera citation, el answer debe tener (Fuente: …
        c_hit = (not case["expect_citation"]) or ("(Fuente:" in resp.answer)
        retrieval_hits += int(r_hit)
        action_hits += int(a_hit)
        citation_correct += int(c_hit)
        rows.append(
            {
                "query": case["query"],
                "action": resp.action,
                "expected_action": case["expected_action"],
                "expect_citation": case["expect_citation"],
                "category": _category(bool(case["expected_docs"]), case["expected_action"]),
                "action_ok": a_hit,
                "retrieval_ok": r_hit,
                "citation_ok": c_hit,
                "got_docs": sorted(got_docs),
                "expected_docs": sorted(case["expected_docs"]),
            }
        )

    metrics = {
        "retrieval@1": retrieval_hits / total,
        "guardrail_accuracy": action_hits / total,
        "grounding_citation": citation_correct / total,
    }
    if verbose:
        _print_report(rows, metrics)
    return metrics


def _print_report(rows: list[dict], metrics: dict) -> None:
    """Imprime la configuración, los resultados por caso (tabla) y el resumen."""
    config = Table(title="Configuración", title_style="bold")
    config.add_column("Parámetro", style="cyan", width=20)
    config.add_column("Valor")
    for clave, valor in (
        ("LLM", "mock — sin API, sin red"),
        ("Embeddings", "ninguno — búsqueda local (TF-IDF)"),
        ("Reranker", "no activo"),
        ("Settings `.env`", "NO se lee — todo hardcoded"),
        ("Resultado", "determinista y reproducible"),
    ):
        config.add_row(clave, valor)
    _console.print(config)
    print()

    casos = Table(title="Evaluación", title_style="bold", pad_edge=False, padding=0)
    casos = Table(title="Evaluación", title_style="bold", pad_edge=False, padding=0,
                  show_lines=True)
    casos.add_column("#", justify="right", style="dim", width=2)
    casos.add_column("Categoría", style="cyan", width=9, justify="center")
    casos.add_column("Pregunta", width=20, overflow="fold")
    casos.add_column("Fuentes", width=12, overflow="ellipsis")
    casos.add_column("Esperado", width=11, overflow="ellipsis")
    casos.add_column("Acción", justify="center", width=9, overflow="ellipsis")
    casos.add_column("Veredicto", justify="center", width=9)
    casos.header_style = "bold cyan"
    for i, row in enumerate(rows, start=1):
        ok = all(row[k] for k in ("retrieval_ok", "action_ok", "citation_ok"))
        verdict = "[green bold]✅ OK[/green bold]" if ok else "[red bold]❌ FAIL[/red bold]"
        fuentes = _doc_titles(row["got_docs"]) or "Ninguna"
        esperado = _doc_titles(row["expected_docs"]) or "(sin fuentes)"
        accion = _PAST_ES.get(row["action"], row["action"] or "—")
        casos.add_row(
            str(i),
            row["category"],
            row["query"],
            fuentes,
            esperado,
            accion,
            verdict,
        )
    _console.print(casos)

    resumen = Table(title="Métricas agregadas", title_style="bold")
    resumen.add_column("Métrica", style="cyan")
    resumen.add_column("Valor", justify="center")
    for name, value in metrics.items():
        color = "green" if value == 1.0 else "red"
        resumen.add_row(name, f"[{color} bold]{value:.2%}[/{color} bold]")
    _console.print(resumen)


def main() -> int:
    metrics = evaluate()
    print()
    print(f"tasa de aprobación global: "
          f"{sum(1 for m in metrics.values() if m == 1) / len(metrics):.2%} (3/3 100% = perfecto)")
    return 0


if __name__ == "__main__":
    sys.exit(main())