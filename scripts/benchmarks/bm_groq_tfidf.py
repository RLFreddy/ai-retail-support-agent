"""Benchmark 2: groq_tfidf — Groq (LLM real) + TF-IDF, sin embeddings.

Estado actual de entrega: LLM real vía LiteLLM con retrieval léxico offline.
Requiere GROQ_API_KEY (o la key del LLM_MODEL configurado en .env).

Uso:
    uv run python scripts/benchmarks/bm_groq_tfidf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib_harness import Variant, common_parser, run_benchmark

VARIANT = Variant(
    slug="groq_tfidf",
    llm="real",
    embedding_backend="offline",
    rerank=False,
    description="Groq + TF-IDF offline (sin embeddings).",
)


def main() -> int:
    args = common_parser().parse_args()
    run_benchmark(VARIANT, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())