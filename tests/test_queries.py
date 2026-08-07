"""Recursos de los benchmarks: las queries no se desincronizan del código real.

El harness de evaluación lee sus golden sets del JSON versionado
``scripts/queries/default.json``; la base de conocimiento (documentos) se toma
directamente de ``src.knowledge.DOCUMENTS`` (fuente única, sin duplicados).
Estos tests garantizan que el JSON esté completo y consistente con el código.
"""

import json
from pathlib import Path

from src.knowledge import DOCUMENTS

QUERIES_DIR = Path(__file__).resolve().parents[1] / "scripts" / "queries"

VALID_ACTIONS_OF_QUERIES = {"answer", "refer", "escalate", "block"}

def _query_cases() -> list[dict]:
    return json.loads((QUERIES_DIR / "default.json").read_text(encoding="utf-8"))


def test_expected_sources_son_indices_validos_de_documentos():
    for query in _query_cases():
        for idx in query.get("expected_sources") or []:
            assert 0 <= idx < len(DOCUMENTS), f"{query['id']}: fuente [{idx}] no existe"


def test_default_json_tiene_expected_short_prosa_y_why():
    for query in _query_cases():
        assert query.get("expected_short"), f"{query['id']} sin expected_short (prosa)"
        assert query.get("expected_hint"), f"{query['id']} sin expected_hint (prosa)"
        assert query.get("why"), f"{query['id']} sin why"
        assert query["expected_action"] in VALID_ACTIONS_OF_QUERIES


def test_documentos_ids_y_titulos_unicos():
    titles = [doc.title for doc in DOCUMENTS]
    ids = [doc.id for doc in DOCUMENTS]
    assert len(DOCUMENTS) >= 1
    assert len(set(ids)) == len(ids), "ids de DOCUMENTS duplicados"
    assert len(set(titles)) == len(titles), "títulos de DOCUMENTS duplicados"
    assert all(doc.title and doc.id for doc in DOCUMENTS)