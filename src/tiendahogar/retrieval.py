"""Retrieval over the knowledge base.

- ``LexicalRetriever`` (default): BM25 with an English stemmer. Offline and
  deterministic.
- ``SemanticRetriever``: embeddings from any OpenAI-compatible endpoint (a
  local TEI server, OpenAI, OpenRouter, Gemini...) ranked by cosine
  similarity, with an optional TEI cross-encoder reranker.

Each retriever owns its relevance threshold, because BM25 and cosine scores
live on different scales. When no document reaches it, retrieval returns
nothing and the agent has no context to answer from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import bm25s
import httpx
import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_EN
from openai import OpenAI

from tiendahogar.knowledge import DOCUMENTS, Document

# bm25s sets its own logger to DEBUG on import.
logging.getLogger("bm25s").setLevel(logging.WARNING)

#: bm25s' English list plus the question words customers type, which carry no
#: topic ("how long does a refund take" is about refunds, not about "how").
STOP_WORDS = sorted(
    set(STOPWORDS_EN)
    | {"can", "could", "did", "do", "does", "get", "has", "have", "how", "i", "long",
       "make", "many", "me", "much", "my", "take", "want", "what", "when", "where",
       "which", "who", "why", "you", "your"}
)


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
    documents: list[Document], scores: np.ndarray, top_k: int, threshold: float
) -> list[RetrievalResult]:
    order = np.argsort(-scores, kind="stable")
    return [
        RetrievalResult(documents[i], float(scores[i]))
        for i in order[:top_k]
        if scores[i] >= threshold
    ]


class LexicalRetriever:
    """BM25 (bm25s) over stemmed English tokens."""

    def __init__(self, documents: list[Document] = DOCUMENTS, threshold: float = 0.5):
        self.documents = list(documents)
        self.threshold = threshold
        self._stemmer = Stemmer.Stemmer("english")
        self._index = bm25s.BM25()
        self._index.index(self._tokenize([d.content for d in self.documents]),
                          show_progress=False)

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return bm25s.tokenize(texts, stopwords=STOP_WORDS, stemmer=self._stemmer,
                              return_ids=False, show_progress=False)

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
        payload: dict = {"query": query, "texts": [r.document.content for r in results]}
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
        client: OpenAI,
        model: str,
        documents: list[Document] = DOCUMENTS,
        threshold: float = 0.56,
        reranker: TeiReranker | None = None,
    ):
        self.client = client
        self.model = model
        self.documents = list(documents)
        self.threshold = threshold
        self.reranker = reranker
        self._doc_vectors: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            return []
        if self._doc_vectors is None:
            self._doc_vectors = self._embed([d.content for d in self.documents])
        scores = self._doc_vectors @ self._embed([query])[0]
        # Over-fetch so the reranker has candidates to reorder.
        fetch = top_k * 2 if self.reranker else top_k
        results = _top(self.documents, scores, fetch, self.threshold)
        if self.reranker and len(results) > 1:
            results = self.reranker.rerank(query, results)
        return results[:top_k]


def build_retriever(settings=None) -> Retriever:
    """Semantic retrieval when ``EMBEDDING_MODEL`` is set, lexical otherwise."""
    from tiendahogar.config import get_settings

    settings = settings or get_settings()
    if not settings.embedding_model:
        return LexicalRetriever(threshold=settings.lexical_threshold)

    client = OpenAI(
        base_url=settings.embedding_base_url or None,
        # Local servers such as TEI need no key, but the client requires one.
        api_key=settings.embedding_api_key.get_secret_value() or "unused",
    )
    reranker = (
        TeiReranker(settings.rerank_url, settings.rerank_model) if settings.rerank_url else None
    )
    return SemanticRetriever(
        client,
        settings.embedding_model,
        threshold=settings.semantic_threshold,
        reranker=reranker,
    )
