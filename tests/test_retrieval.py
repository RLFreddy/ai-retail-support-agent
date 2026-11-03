"""Tests de la capa RAG.

Cubre el requisito mínimo: la recuperación trae el documento correcto para una
pregunta clara (garantía), y el umbral de relevancia decide cuándo no hay contexto.
También cubre el retriever semántico y el toggle de rerank (sin red: stubs).
"""

import types

from tiendahogar.config import Settings
from tiendahogar.knowledge import Document
from tiendahogar.retrieval import SemanticRetriever, TfidfRetriever, build_retriever

_DOCS = [
    Document(id="d0", title="Garantía", content="Garantía de electrodomésticos."),
    Document(id="d1", title="Envíos", content="Envíos a todo el país."),
]


def _semantic(monkeypatch, **kwargs) -> SemanticRetriever:
    """SemanticRetriever sin red: docs fijos y ``_embed`` stub."""
    retriever = SemanticRetriever(documents=_DOCS, **kwargs)
    calls = {"n": 0}

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls["n"] += 1
        if len(texts) == 1:
            return [[0.85, 0.15]]
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(retriever, "_embed", fake_embed)
    return retriever, calls


class TestSemanticRetriever:
    def test_rerank_desactivado_no_llama_a_tei(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch, backend="tei", rerank_url="http://tei/rerank", rerank_model="bge"
        )
        called: list[bool] = []
        monkeypatch.setattr(retriever, "_rerank_tei", lambda *a, **k: called.append(True) or [])
        results = retriever.retrieve("hola", top_k=2, threshold=0.0)
        assert called == []
        assert [r.doc_id for r in results] == ["d0", "d1"]

    def test_rerank_tei_activado_ejecuta_el_crossencoder(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch, backend="tei", rerank_url="http://tei/rerank", rerank_enabled=True
        )
        called: list[bool] = []
        monkeypatch.setattr(retriever, "_rerank_tei", lambda *a, **k: called.append(True) or [(0, 0.9)])
        results = retriever.retrieve("hola", top_k=1, threshold=0.0)
        assert called == [True]
        assert results[0].doc_id == "d0"

    def test_rerank_tei_envia_texts_y_parsea_array_plano(self, monkeypatch):
        """Regresión P0: TEI espera `texts` (no `documents`) y responde un array
        plano [[index, score], ...] ya ordenado por relevancia; no se re-ordena."""
        import httpx

        retriever, _ = _semantic(
            monkeypatch,
            backend="tei",
            rerank_url="http://tei/rerank",
            rerank_enabled=True,
            rerank_model="BAAI/bge-reranker-v2-m3",
        )
        captured: dict = {}

        def fake_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return types.SimpleNamespace(
                json=lambda: [{"index": 1, "score": 0.99}, {"index": 0, "score": 0.01}],
                raise_for_status=lambda: None,
            )

        monkeypatch.setattr(httpx, "post", fake_post)
        results = retriever.retrieve("hola", top_k=2, threshold=0.0)
        assert captured["url"] == "http://tei/rerank"
        assert captured["json"]["texts"] == [d.content for d in _DOCS]
        assert "documents" not in captured["json"]
        assert captured["json"]["model"] == "BAAI/bge-reranker-v2-m3"
        assert [r.score for r in results] == [0.99, 0.01]  # respeta el orden TEI
        assert [r.doc_id for r in results] == ["d1", "d0"]

    def test_rerank_nube_usa_litellm_y_mapea_indices(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch,
            backend="openai",
            rerank_enabled=True,
            rerank_model="cohere/rerank-v3.5",
        )
        called: list[bool] = []

        def fake_rerank(model, query, documents, top_n):
            called.append(True)
            return types.SimpleNamespace(
                results=[
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.2},
                ]
            )

        monkeypatch.setattr("litellm.rerank", fake_rerank)
        results = retriever.retrieve("hola", top_k=2, threshold=0.0)
        assert called == [True]
        assert [r.doc_id for r in results] == ["d0", "d1"]

    def test_rerank_nube_pone_primero_el_doc_que_gana(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch,
            backend="openai",
            rerank_enabled=True,
            rerank_model="cohere/rerank-v3.5",
        )

        def fake_rerank(query, scored, top_n):
            return [(1, 0.99), (0, 0.01)]

        monkeypatch.setattr(retriever, "_rerank_litellm", fake_rerank)
        results = retriever.retrieve("hola", top_k=2, threshold=0.0)
        assert [r.doc_id for r in results] == ["d1", "d0"]


class TestBuildRetriever:
    def test_offline_por_defecto_devuelve_tfidf(self):
        assert isinstance(build_retriever(Settings(embedding_backend="offline")), TfidfRetriever)

    def test_umbral_por_backend_offline_usa_015(self):
        assert Settings(embedding_backend="offline").retrieval_threshold == 0.15

    def test_umbral_por_backend_embeddings_usa_056(self):
        assert Settings(embedding_backend="tei").retrieval_threshold == 0.56
        assert Settings(embedding_backend="openai").retrieval_threshold == 0.56

    def test_tei_conecta_rerank_local(self):
        r = build_retriever(
            Settings(
                embedding_backend="tei",
                tei_base_url="http://localhost:8080/v1",
                tei_rerank_base_url="",
            )
        )
        assert isinstance(r, SemanticRetriever)
        assert r.rerank_url == "http://localhost:8080/v1/rerank"
        assert not r.rerank_enabled

    def test_tei_rerank_usuario_ruta_separada_si_la_hay(self):
        r = build_retriever(
            Settings(
                embedding_backend="tei",
                tei_base_url="http://localhost:8080/v1",
                tei_rerank_base_url="http://localhost:8001",
            )
        )
        assert r.rerank_url == "http://localhost:8001/rerank"

    def test_openai_selecciona_modelo_y_key(self):
        r = build_retriever(
            Settings(
                embedding_backend="openai",
                openai_api_key="sk-test",
                rerank_enabled=True,
            )
        )
        assert isinstance(r, SemanticRetriever)
        assert r.model == "openai/text-embedding-3-small"
        assert r.api_key == "sk-test"
        assert r.rerank_url == ""
        assert r.rerank_model == "cohere/rerank-v3.5"


def test_rag_trae_documento_correcto_para_garantia():
    retriever = TfidfRetriever()
    results = retriever.retrieve("¿Cuánto dura la garantía de una refrigeradora?", top_k=1)
    assert results, "debería recuperar al menos un documento"
    assert results[0].doc_id == "garantia"


def test_rag_devuelve_no_resultados_con_query_irrelevante():
    retriever = TfidfRetriever()
    results = retriever.retrieve("zzz qwerty asdf jkl", top_k=3, threshold=0.25)
    assert results == []


def test_rag_no_devuelve_contexto_por_debajo_del_umbral():
    retriever = TfidfRetriever()
    results = retriever.retrieve("el clima de la ciudad", top_k=3, threshold=0.99)
    assert results == []


def test_rag_orden_por_relevancia_descendente():
    retriever = TfidfRetriever()
    results = retriever.retrieve("política de devoluciones de productos sin usar", top_k=3)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rag_top_k_limita_resultados():
    retriever = TfidfRetriever()
    results = retriever.retrieve("garantía de productos y devoluciones", top_k=2, threshold=0.0)
    assert len(results) == 2


def test_rag_retorna_citas_en_formato_esperado():
    retriever = TfidfRetriever()
    results = retriever.retrieve("¿a dónde debo escribir una queja sobre un empleado?", top_k=2)
    assert any(r.doc_id == "contacto" for r in results)