"""Retrieval tests: the right document for a clear question, the relevance
threshold, and the semantic retriever with its rerank toggle (stubbed, no
network)."""

import types

from tiendahogar.config import Settings
from tiendahogar.knowledge import Document
from tiendahogar.retrieval import SemanticRetriever, TfidfRetriever, build_retriever

_DOCS = [
    Document(id="d0", title="Warranty", content="Appliance warranty."),
    Document(id="d1", title="Shipping", content="Shipping nationwide."),
]


def _semantic(monkeypatch, **kwargs) -> SemanticRetriever:
    """SemanticRetriever without network: fixed docs and a stubbed ``_embed``."""
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
    def test_rerank_disabled_does_not_call_tei(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch, backend="tei", rerank_url="http://tei/rerank", rerank_model="bge"
        )
        called: list[bool] = []
        monkeypatch.setattr(retriever, "_rerank_tei", lambda *a, **k: called.append(True) or [])
        results = retriever.retrieve("hello", top_k=2, threshold=0.0)
        assert called == []
        assert [r.doc_id for r in results] == ["d0", "d1"]

    def test_tei_rerank_enabled_runs_the_cross_encoder(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch, backend="tei", rerank_url="http://tei/rerank", rerank_enabled=True
        )
        called: list[bool] = []
        monkeypatch.setattr(retriever, "_rerank_tei", lambda *a, **k: called.append(True) or [(0, 0.9)])
        results = retriever.retrieve("hello", top_k=1, threshold=0.0)
        assert called == [True]
        assert results[0].doc_id == "d0"

    def test_tei_rerank_sends_texts_and_keeps_tei_order(self, monkeypatch):
        """TEI expects `texts` (not `documents`) and returns a flat array already
        sorted by relevance, which must be kept."""
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
        results = retriever.retrieve("hello", top_k=2, threshold=0.0)
        assert captured["url"] == "http://tei/rerank"
        assert captured["json"]["texts"] == [d.content for d in _DOCS]
        assert "documents" not in captured["json"]
        assert captured["json"]["model"] == "BAAI/bge-reranker-v2-m3"
        assert [r.score for r in results] == [0.99, 0.01]
        assert [r.doc_id for r in results] == ["d1", "d0"]

    def test_cloud_rerank_uses_litellm_and_maps_indices(self, monkeypatch):
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
        results = retriever.retrieve("hello", top_k=2, threshold=0.0)
        assert called == [True]
        assert [r.doc_id for r in results] == ["d0", "d1"]

    def test_cloud_rerank_puts_the_winner_first(self, monkeypatch):
        retriever, _ = _semantic(
            monkeypatch,
            backend="openai",
            rerank_enabled=True,
            rerank_model="cohere/rerank-v3.5",
        )

        def fake_rerank(query, scored, top_n):
            return [(1, 0.99), (0, 0.01)]

        monkeypatch.setattr(retriever, "_rerank_litellm", fake_rerank)
        results = retriever.retrieve("hello", top_k=2, threshold=0.0)
        assert [r.doc_id for r in results] == ["d1", "d0"]


class TestBuildRetriever:
    def test_offline_backend_builds_tfidf(self):
        assert isinstance(build_retriever(Settings(embedding_backend="offline")), TfidfRetriever)

    def test_offline_threshold_is_015(self):
        assert Settings(embedding_backend="offline").retrieval_threshold == 0.15

    def test_semantic_threshold_is_056(self):
        assert Settings(embedding_backend="tei").retrieval_threshold == 0.56
        assert Settings(embedding_backend="openai").retrieval_threshold == 0.56

    def test_tei_wires_the_local_rerank(self):
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

    def test_tei_rerank_uses_its_own_url_when_set(self):
        r = build_retriever(
            Settings(
                embedding_backend="tei",
                tei_base_url="http://localhost:8080/v1",
                tei_rerank_base_url="http://localhost:8001",
            )
        )
        assert r.rerank_url == "http://localhost:8001/rerank"

    def test_openai_selects_model_and_key(self):
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


def test_warranty_question_retrieves_the_warranty_document():
    results = TfidfRetriever().retrieve("How long is the warranty on a refrigerator?", top_k=1)
    assert results
    assert results[0].doc_id == "warranty"


def test_irrelevant_query_returns_nothing():
    assert TfidfRetriever().retrieve("zzz qwerty asdf jkl", top_k=3, threshold=0.25) == []


def test_nothing_below_the_threshold_is_returned():
    assert TfidfRetriever().retrieve("the weather in the city", top_k=3, threshold=0.99) == []


def test_results_are_sorted_by_score():
    results = TfidfRetriever().retrieve("returns policy for unused products", top_k=3)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_results():
    results = TfidfRetriever().retrieve("warranty and returns", top_k=2, threshold=0.0)
    assert len(results) == 2


def test_employee_complaint_retrieves_the_contact_channels():
    results = TfidfRetriever().retrieve("where do I send a complaint about an employee?", top_k=2)
    assert any(r.doc_id == "contact" for r in results)
