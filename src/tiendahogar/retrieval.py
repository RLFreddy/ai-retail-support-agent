"""Retrieval layer of the TiendaHogar agent.

- ``TfidfRetriever``: default, deterministic and offline (scikit-learn TF-IDF
  with an accent-folding tokenizer and English stop words).
- ``SemanticRetriever``: embeddings through LiteLLM (local TEI with bge-m3,
  OpenRouter, OpenAI or Gemini) with optional reranking.

When no document reaches the relevance threshold, retrieval returns nothing and
the agent does not answer with invented information.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from operator import itemgetter
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tiendahogar.knowledge import DOCUMENTS, Document


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document with its relevance score."""

    document: Document
    score: float

    @property
    def doc_id(self) -> str:
        return self.document.id

    @property
    def title(self) -> str:
        return self.document.title


class BaseRetriever(Protocol):
    """Common retriever interface."""

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list[RetrievalResult]:
        ...


def _tokenize(text: str) -> list[str]:
    """Lowercase, fold accents (NFKD) and split into alphanumeric tokens."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.findall(r"[a-z0-9]+", text)


#: English function words (tokenized form). Content words never go here.
STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "about", "as", "into", "is", "are", "was",
    "be", "it", "its", "this", "that", "my", "me", "i", "you", "your", "we",
    "our", "do", "does", "can", "how", "what", "when", "there",
})


class TfidfRetriever:
    """Deterministic TF-IDF retriever (scikit-learn), no network.

    The default threshold (0.15) separates relevant queries from irrelevant
    ones on this corpus; the tests and evals check it.
    """

    def __init__(self, documents: list[Document] = DOCUMENTS):
        self.documents = list(documents)
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            lowercase=False,
            norm="l2",
            stop_words=list(STOP_WORDS),
        )
        self._tfidf = self._vectorizer.fit_transform(
            [d.content for d in self.documents]
        )

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list[RetrievalResult]:
        if not query.strip():
            return []
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf)[0]
        scored = [(i, float(s)) for i, s in enumerate(sims) if s >= threshold]
        scored.sort(key=itemgetter(1), reverse=True)
        top = scored[:top_k]
        return [RetrievalResult(document=self.documents[i], score=score) for i, score in top]


class SemanticRetriever:
    """Embedding retriever through LiteLLM.

    Backends: ``tei`` (local TEI, OpenAI-compatible, bge-m3), ``openrouter``,
    ``openai`` and ``gemini``. With ``rerank_enabled`` the candidates go
    through a cross-encoder: TEI ``/rerank`` over HTTP when ``rerank_url`` is
    set, otherwise ``litellm.rerank`` with ``rerank_model``.
    """

    def __init__(
        self,
        documents: list[Document] = DOCUMENTS,
        backend: str = "tei",
        model: str = "",
        api_base: str = "",
        api_key: str = "",
        rerank_enabled: bool = False,
        rerank_url: str = "",
        rerank_model: str = "",
    ):
        self.documents = list(documents)
        self.backend = backend
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.rerank_enabled = rerank_enabled
        self.rerank_url = rerank_url
        self.rerank_model = rerank_model
        self._doc_embeddings: list[list[float]] | None = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import litellm

        kwargs: dict = {"model": self.model, "input": texts, "api_key": self.api_key or "dummy"}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        resp = litellm.embedding(**kwargs)
        return [item["embedding"] for item in resp.data]

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.56) -> list[RetrievalResult]:
        if not query.strip():
            return []
        if self._doc_embeddings is None:
            self._doc_embeddings = self._embed([d.content for d in self.documents])
        q = self._embed([query])[0]
        sims = cosine_similarity([q], self._doc_embeddings)[0]
        scored = [(i, s) for i, s in enumerate(sims) if s >= threshold]
        scored.sort(key=itemgetter(1), reverse=True)
        candidates = scored[: top_k * 2]

        if self.rerank_enabled and (self.rerank_url or self.rerank_model) and len(candidates) > 1:
            if self.rerank_url:
                candidates = self._rerank_tei(query, candidates, top_k * 2)
            else:
                candidates = self._rerank_litellm(query, candidates, top_k * 2)

        top = candidates[:top_k]
        return [RetrievalResult(document=self.documents[i], score=score) for i, score in top]

    def _rerank_tei(self, query: str, scored: list[tuple[int, float]], top_n: int) -> list[tuple[int, float]]:
        """Local TEI cross-encoder over HTTP ``/rerank``.

        TEI expects ``texts`` and returns ``[{"index", "score"}, ...]`` already
        sorted by relevance; that order is kept.
        """
        import httpx

        docs = [self.documents[i].content for i, _ in scored]
        payload = {"query": query, "texts": docs, "top_n": top_n}
        if self.rerank_model:
            payload["model"] = self.rerank_model
        resp = httpx.post(self.rerank_url, json=payload, timeout=30)
        resp.raise_for_status()
        out: list[tuple[int, float]] = []
        for item in resp.json():
            i = int(item["index"])
            out.append((scored[i][0], float(item.get("score", scored[i][1]))))
        return out

    def _rerank_litellm(self, query: str, scored: list[tuple[int, float]], top_n: int) -> list[tuple[int, float]]:
        """Cloud reranker through ``litellm.rerank``."""
        import litellm

        docs = [self.documents[i].content for i, _ in scored]
        resp = litellm.rerank(
            model=self.rerank_model,
            query=query,
            documents=docs,
            top_n=top_n,
        )
        out: list[tuple[int, float]] = []
        for r in (resp.results or []):
            item = r if isinstance(r, dict) else r.model_dump()
            idx = scored[item["index"]][0]
            out.append((idx, float(item.get("relevance_score", scored[item["index"]][1]))))
        return out


def build_retriever(settings=None) -> BaseRetriever:
    """Build the retriever for the configured embedding backend."""
    from tiendahogar.config import EMBEDDING_KEY_ENV, get_settings

    settings = settings or get_settings()
    backend = settings.embedding_backend
    if backend == "offline":
        return TfidfRetriever()

    if backend == "tei":
        model, api_key = settings.tei_embed_model, ""
        api_base = settings.tei_base_url
        base = settings.tei_rerank_base_url or settings.tei_base_url
        rerank_url = f"{base.rstrip('/')}/rerank" if base else ""
        rerank_model = settings.tei_rerank_model if base else ""
    else:
        model = {
            "openrouter": settings.openrouter_embed_model,
            "openai": settings.openai_embed_model,
            "gemini": settings.gemini_embed_model,
        }[backend]
        api_key = settings.api_keys.get(EMBEDDING_KEY_ENV[backend], "")
        api_base = ""
        rerank_url = ""
        rerank_model = settings.rerank_model

    return SemanticRetriever(
        documents=DOCUMENTS,
        backend=backend,
        model=model,
        api_base=api_base,
        api_key=api_key,
        rerank_enabled=settings.rerank_enabled,
        rerank_url=rerank_url,
        rerank_model=rerank_model,
    )
