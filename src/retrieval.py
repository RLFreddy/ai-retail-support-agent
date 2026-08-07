"""Capa de recuperación (RAG) del agente TiendaHogar.

Diseño:
- ``BaseRetriever``: protocolo común.
- ``TfidfRetriever``: retriever por defecto, determinista y sin red, basado en
  scikit-learn ``TfidfVectorizer``. El tokenizer es propio (minúsculas, folding
  de acentos) y se filtran las palabras funcionales del español
  (``SPANISH_STOP_WORDS``): esa configuración elimina los scores espurios que
  el vectorizer por defecto daba por términos de parada compartidos, el motivo
  por el que antes se implementaba el TF-IDF a mano. Ver SUBMISSION.md.
- ``SemanticRetriever``: embeddings vía LiteLLM (TEI local bge-m3, OpenRouter,
  OpenAI o Gemini), con rerank opcional: el cross-encoder local se llama por
  HTTP directo a TEI ``/rerank`` (TEI no está entre los proveedores de rerank
  de litellm); los rerankers en la nube van por ``litellm.rerank`` (Cohere,
  Jina, Together, Azure AI, ...). ``RERANK_ENABLED`` apaga el rerank por
  completo para medir su impacto (benchmark) o ahorrar latencia/costo.
- Umbral de relevancia: si ningún documento lo supera, retrieval retorna vacío
  → el agente NO responde con información inventada.

Nota de diseño (corpus de 5 docs): cada documento es un único chunk; el
chunking fino se evita a propósito, ver SUBMISSION.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from operator import itemgetter
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge import DOCUMENTS, Document


@dataclass(frozen=True)
class RetrievalResult:
    """Documento recuperado con su score de relevancia."""

    document: Document
    score: float

    @property
    def doc_id(self) -> str:
        return self.document.id

    @property
    def title(self) -> str:
        return self.document.title


class BaseRetriever(Protocol):
    """Interfaz común de los retrievers."""

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.15) -> list[RetrievalResult]:
        ...


def _tokenize(text: str) -> list[str]:
    """Tokenización simple para español (minúsculas, folding de acentos vía NFKD)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.findall(r"[a-z0-9]+", text)


#: Palabras funcionales del español (forma tokenizada: minúsculas, sin acentos).
#: Solo palabras de función (artículos, preposiciones, pronombres, verbos
#: auxiliares comunes) — NUNCA palabras de contenido como "garantia", "devolucion".
SPANISH_STOP_WORDS: frozenset[str] = frozenset({
    "a", "al", "ante", "bajo", "cabe", "con", "contra", "de", "del", "desde",
    "el", "en", "entre", "hacia", "hasta", "la", "las", "los", "lo", "su",
    "sus", "para", "por", "sin", "sobre", "tras", "un", "una", "unos", "unas",
    "y", "o", "u", "ni", "pero", "que", "si", "se", "es", "son", "esta",
    "estan", "no", "me", "te", "nos", "les", "ya", "hay",
})


# --------------------------------------------------------------------------- #
# TfidfRetriever (scikit-learn)
# --------------------------------------------------------------------------- #
class TfidfRetriever:
    """Retriever TF-IDF determinista con scikit-learn, sin red.

    ``TfidfVectorizer`` con el tokenizer propio (``_tokenize``) y las
    ``SPANISH_STOP_WORDS``: al descartar palabras funcionales se evita que
    queries irrelevantes obtengan scores espurios por términos de parada
    compartidos (era el motivo por el que se evitaba sklearn antes).
    Calibración: el umbral de relevancia (default 0.15) separa queries
    relevantes (score >= 0.2) de irrelevantes (score <= 0.14), verificado por
    tests y el golden set.
    """

    def __init__(self, documents: list[Document] = DOCUMENTS):
        self.documents = list(documents)
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            token_pattern=None,
            lowercase=False,
            norm="l2",
            stop_words=list(SPANISH_STOP_WORDS),
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


# --------------------------------------------------------------------------- #
# SemanticRetriever (embeddings vía LiteLLM + rerank TEI opcional)
# --------------------------------------------------------------------------- #
class SemanticRetriever:
    """Retriever semántico por embeddings a través de LiteLLM.

    Backends:
      - ``tei``     : servidor TEI local (endpoint OpenAI-compatible) con bge-m3.
      - ``openrouter``: qwen/qwen3-embedding-8b vía OpenRouter.
      - ``openai``    : text-embedding-3-small vía OpenAI.
      - ``gemini``    : text-embedding-004 vía Google Gemini.

    Rerank (opcional, ``rerank_enabled``):
      - Con ``rerank_url`` (TEI local): over-fetch (top_k * 2) y POST directo a
        ``/rerank`` con el cross-encoder ``bge-reranker-v2-m3`` — TEI no está
        entre los proveedores de rerank de litellm (docs.litellm.ai/docs/rerank).
      - Con ``rerank_model`` (nube): ``litellm.rerank`` (Cohere, Jina, Together,
        Azure AI, ...). Si ``rerank_enabled=False``, retrieval por coseno puro.
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
        """Cross-encoder local de TEI: HTTP directo a ``/rerank`` (litellm no
        cubre TEI como proveedor de rerank).

        API de TEI: el payload espera ``texts`` (no ``documents``) y responde un
        arreglo plano ``[{"index": int, "score": float}, ...]`` ya ordenado por
        relevancia descendente. Se respeta ese orden (no se re-ordena por index).
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
        """Reranker en la nube vía ``litellm.rerank`` (Cohere/Jina/Together/...)."""
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


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_retriever(settings=None) -> BaseRetriever:
    """Construye el retriever según la configuración (embeddings_backend).

    Rerank: ``RERANK_ENABLED`` lo activa. Con backend ``tei`` usa el cross-encoder
    local por HTTP a ``/rerank`` (``tei_rerank_model``); con backends en la nube
    (openrouter/openai/gemini) usa ``litellm.rerank`` (``rerank_model``).
    """
    from src.config import EMBEDDING_KEY_ENV, get_settings

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


#: Alias usado por tests/factory.
Retriever = TfidfRetriever