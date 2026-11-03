"""Retrieval tests: lexical BM25 on the real corpus, and the semantic retriever
and TEI reranker with stubbed network calls."""

import types

import httpx
import pytest

from tiendahogar.config import Settings
from tiendahogar.domain import Language
from tiendahogar.knowledge import KNOWLEDGE, Document
from tiendahogar.retrieval import (
    LexicalRetriever,
    RetrievalResult,
    SemanticRetriever,
    TeiReranker,
    build_retriever,
)

DOCS = [
    Document(id="d0", title="Warranty", content="Appliance warranty."),
    Document(id="d1", title="Shipping", content="Shipping nationwide."),
]


class FakeEmbeddings:
    """OpenAI-like client: documents map to the axes, queries lean to d0."""

    def __init__(self):
        self.embeddings = self
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        vectors = [[0.85, 0.15]] if len(input) == 1 else [[1.0, 0.0], [0.0, 1.0]]
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=v) for v in vectors])


class TestLexical:
    @pytest.mark.parametrize(
        ("query", "doc_id"),
        [
            ("How long is the warranty on a refrigerator?", "warranty"),
            ("How do I make a return?", "returns"),
            ("Which cities do you ship to?", "shipping"),
            ("In how many days are refunds processed?", "refunds"),
            ("Where do I send a complaint about an employee?", "contact"),
        ],
    )
    def test_clear_question_retrieves_its_document_first(self, query, doc_id):
        assert LexicalRetriever().retrieve(query)[0].doc_id == doc_id

    @pytest.mark.parametrize(
        "query", ["What is the weather like on Mars?", "zzz qwerty", "How do I?", ""]
    )
    def test_unrelated_question_retrieves_nothing(self, query):
        assert LexicalRetriever().retrieve(query) == []

    def test_results_are_sorted_and_limited(self):
        results = LexicalRetriever(threshold=0.0).retrieve(
            "How many days do I have to return an unused product?", top_k=2
        )
        assert len(results) == 2
        assert results[0].score >= results[1].score


class TestSemantic:
    def test_ranks_by_cosine_and_embeds_documents_once(self):
        client = FakeEmbeddings()
        retriever = SemanticRetriever(client, "m", documents=DOCS, threshold=0.0)
        assert [r.doc_id for r in retriever.retrieve("q", top_k=2)] == ["d0", "d1"]
        retriever.retrieve("q")
        assert client.calls == 3  # documents once, then one call per query

    def test_threshold_filters_weak_matches(self):
        retriever = SemanticRetriever(FakeEmbeddings(), "m", documents=DOCS, threshold=0.5)
        assert [r.doc_id for r in retriever.retrieve("q")] == ["d0"]

    def test_reranker_reorders_the_candidates(self):
        class Flip:
            def rerank(self, query, results):
                return [RetrievalResult(r.document, 1.0 - r.score) for r in reversed(results)]

        retriever = SemanticRetriever(
            FakeEmbeddings(), "m", documents=DOCS, threshold=0.0, reranker=Flip()
        )
        assert [r.doc_id for r in retriever.retrieve("q", top_k=1)] == ["d1"]


def test_tei_reranker_sends_texts_and_keeps_tei_order(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json)
        return types.SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: [{"index": 1, "score": 0.99}, {"index": 0, "score": 0.01}],
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    results = TeiReranker("http://tei:8001/rerank", "bge-reranker").rerank(
        "q", [RetrievalResult(DOCS[0], 0.9), RetrievalResult(DOCS[1], 0.8)]
    )
    assert captured["url"] == "http://tei:8001/rerank"
    assert captured["json"] == {
        "query": "q",
        "texts": [d.content for d in DOCS],
        "model": "bge-reranker",
    }
    assert [(r.doc_id, r.score) for r in results] == [("d1", 0.99), ("d0", 0.01)]


class TestBuildRetriever:
    def test_lexical_by_default(self):
        retriever = build_retriever(Settings())
        assert isinstance(retriever, LexicalRetriever)
        assert retriever.threshold == 0.5

    def test_embedding_model_enables_semantic_retrieval(self):
        retriever = build_retriever(
            Settings(embedding_model="BAAI/bge-m3", embedding_base_url="http://localhost:8080/v1")
        )
        assert isinstance(retriever, SemanticRetriever)
        assert retriever.threshold == 0.56
        assert str(retriever.client.base_url).startswith("http://localhost:8080/v1")
        assert retriever.reranker is None

    def test_rerank_url_enables_the_reranker(self):
        retriever = build_retriever(
            Settings(embedding_model="BAAI/bge-m3", rerank_url="http://localhost:8001/rerank")
        )
        assert retriever.reranker.url == "http://localhost:8001/rerank"


class TestSpanishLexical:
    @pytest.fixture
    def retriever(self) -> LexicalRetriever:
        return LexicalRetriever(language=Language.ES)

    @pytest.mark.parametrize(
        ("query", "doc_id"),
        [
            ("¿Cuánto dura la garantía de una refrigeradora?", "warranty"),
            ("¿Cómo hago una devolución?", "returns"),
            ("¿Puedo devolver una licuadora?", "returns"),  # devolver ~ devoluciones
            ("¿Cuanto demora el envio a otra ciudad?", "shipping"),  # no accents
            ("¿En cuántos días se reembolsa?", "refunds"),
            ("Quiero poner una queja sobre un empleado", "contact"),
        ],
    )
    def test_spanish_question_finds_its_document(self, retriever, query, doc_id):
        results = retriever.retrieve(query)
        assert results[0].doc_id == doc_id
        assert results[0].document.title in {d.title for d in KNOWLEDGE[Language.ES]}

    def test_unrelated_spanish_question_finds_nothing(self, retriever):
        assert retriever.retrieve("¿Cuál es el clima de Marte?") == []

    def test_build_retriever_indexes_the_requested_language(self):
        retriever = build_retriever(Settings(), Language.ES)
        assert retriever.retrieve("garantía")[0].document.title == "Política de garantía"
