"""Retrieval over the knowledge base.

- ``LexicalRetriever`` (default): BM25 with the stemmer and stop words of the
  documents' language. Offline and deterministic.
- ``SemanticRetriever``: embeddings from a Pydantic AI ``Embedder`` ranked by
  cosine similarity, with an optional TEI cross-encoder reranker. The
  embedder is any Pydantic AI embedding model name (``openai:...``,
  ``cohere:...``, ``ollama:...``, ``sentence-transformers:...``) or an
  OpenAI-compatible endpoint such as a local TEI server.

Each retriever owns its relevance threshold, because BM25 and cosine scores
live on different scales. When no document reaches it, retrieval returns
nothing and the agent has no context to answer from.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import bm25s
import httpx
import numpy as np
import numpy.typing as npt
import Stemmer
from bm25s.stopwords import STOPWORDS_EN, STOPWORDS_SPANISH
from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from tiendahogar.config import Settings, get_settings
from tiendahogar.domain import Language
from tiendahogar.knowledge import KNOWLEDGE, Document

Vector = npt.NDArray[np.float64]

# bm25s sets its own logger to DEBUG on import.
logging.getLogger("bm25s").setLevel(logging.WARNING)

#: bm25s' lists plus the question words customers type, which carry no topic
#: ("how long does a refund take" is about refunds, not about "how").
STOP_WORDS_EN = sorted(
    set(STOPWORDS_EN)
    | {
        "can",
        "could",
        "did",
        "do",
        "does",
        "get",
        "has",
        "have",
        "how",
        "i",
        "long",
        "make",
        "many",
        "me",
        "much",
        "my",
        "take",
        "want",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "you",
        "your",
    }
)
STOP_WORDS_ES = sorted(
    set(STOPWORDS_SPANISH)
    | {
        "cuanto",
        "cuánto",
        "cuanta",
        "cuánta",
        "cuantos",
        "cuántos",
        "cuantas",
        "cuántas",
        "como",
        "cómo",
        "donde",
        "dónde",
        "cuando",
        "cuándo",
        "cual",
        "cuál",
        "que",
        "qué",
        "dura",
        "demora",
        "tarda",
        "puedo",
        "pueden",
        "quiero",
        "necesito",
        "tengo",
        "hacer",
        "hago",
    }
)
#: Snowball stemmer and stop words for each language.
_ANALYZERS = {Language.EN: ("english", STOP_WORDS_EN), Language.ES: ("spanish", STOP_WORDS_ES)}
#: Stems Snowball keeps apart although they are the same word family:
#: "devolver" → devolv, "devoluciones" → devolu, "devuelto" → devuelt.
_STEM_ALIASES = {
    Language.EN: {},
    Language.ES: {"devolu": "devolv", "devuelt": "devolv", "devuelv": "devolv"},
}


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document with its relevance score."""

    document: Document
    score: float

    @property
    def doc_id(self) -> str:
        return self.document.id


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]: ...


def _top(
    documents: list[Document], scores: Vector, top_k: int, threshold: float
) -> list[RetrievalResult]:
    order = np.argsort(-scores, kind="stable")
    return [
        RetrievalResult(documents[i], float(scores[i]))
        for i in order[:top_k]
        if scores[i] >= threshold
    ]


class LexicalRetriever:
    """BM25 (bm25s) over stemmed tokens of the documents' language."""

    def __init__(
        self,
        documents: list[Document] | None = None,
        threshold: float = 0.5,
        language: Language = Language.EN,
    ):
        self.documents = list(KNOWLEDGE[language] if documents is None else documents)
        self.threshold = threshold
        stemmer, self._stop_words = _ANALYZERS[language]
        self._aliases = _STEM_ALIASES[language]
        self._stemmer = Stemmer.Stemmer(stemmer)
        self._index = bm25s.BM25()
        self._index.index(self._tokenize([d.content for d in self.documents]), show_progress=False)

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        tokens: list[list[str]] = bm25s.tokenize(
            texts,
            stopwords=self._stop_words,
            stemmer=self._stemmer,
            return_ids=False,
            show_progress=False,
        )
        return [[self._aliases.get(t, t) for t in doc] for doc in tokens]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        tokens = self._tokenize([query])[0]
        if not tokens:
            return []
        return _top(self.documents, self._index.get_scores(tokens), top_k, self.threshold)


class TeiReranker:
    """Cross-encoder served by Hugging Face TEI (``POST /rerank``)."""

    def __init__(self, url: str, model: str = "", timeout: float = 30.0):
        self.url = url
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        payload: dict[str, object] = {
            "query": query,
            "texts": [r.document.content for r in results],
        }
        if self.model:
            payload["model"] = self.model
        response = httpx.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        # TEI returns [{"index": i, "score": s}, ...] sorted by relevance.
        return [
            RetrievalResult(results[item["index"]].document, float(item["score"]))
            for item in response.json()
        ]


class SemanticRetriever:
    """Cosine similarity over embeddings from an OpenAI-compatible endpoint."""

    def __init__(
        self,
        embedder: Embedder,
        documents: list[Document] | None = None,
        threshold: float = 0.56,
        reranker: TeiReranker | None = None,
    ):
        self.embedder = embedder
        self.documents = list(KNOWLEDGE[Language.EN] if documents is None else documents)
        self.threshold = threshold
        self.reranker = reranker
        self._doc_vectors: Vector | None = None

    @staticmethod
    def _normalized(embeddings: Sequence[Sequence[float]]) -> Vector:
        vectors = np.array(embeddings, dtype=np.float64)
        normalized: Vector = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return normalized

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            return []
        if self._doc_vectors is None:
            texts = [d.content for d in self.documents]
            self._doc_vectors = self._normalized(
                self.embedder.embed_documents_sync(texts).embeddings
            )
        query_vector = self._normalized(self.embedder.embed_query_sync(query).embeddings)[0]
        scores = self._doc_vectors @ query_vector
        # Over-fetch so the reranker has candidates to reorder.
        fetch = top_k * 2 if self.reranker else top_k
        results = _top(self.documents, scores, fetch, self.threshold)
        if self.reranker and len(results) > 1:
            results = self.reranker.rerank(query, results)
        return results[:top_k]


def build_embedder(settings: Settings) -> Embedder:
    """``EMBEDDING_MODEL`` with ``EMBEDDING_BASE_URL``: an OpenAI-compatible
    server such as TEI. Without a base URL: a Pydantic AI embedding model name."""
    if not settings.embedding_base_url:
        # Resolve the name now, so a typo or a missing key fails at startup.
        return Embedder(settings.embedding_model, defer_model_check=False)
    provider = OpenAIProvider(
        base_url=settings.embedding_base_url,
        # Local servers such as TEI need no key, but the client requires one.
        api_key=settings.embedding_api_key.get_secret_value() or "unused",
    )
    return Embedder(OpenAIEmbeddingModel(settings.embedding_model, provider=provider))


def build_retriever(
    settings: Settings | None = None, language: Language = Language.EN
) -> Retriever:
    """Retriever over the documents in ``language``: semantic when
    ``EMBEDDING_MODEL`` is set, lexical otherwise."""
    settings = settings or get_settings()
    if not settings.embedding_model:
        return LexicalRetriever(threshold=settings.lexical_threshold, language=language)

    reranker = (
        TeiReranker(settings.rerank_url, settings.rerank_model) if settings.rerank_url else None
    )
    return SemanticRetriever(
        build_embedder(settings),
        documents=KNOWLEDGE[language],
        threshold=settings.semantic_threshold,
        reranker=reranker,
    )
