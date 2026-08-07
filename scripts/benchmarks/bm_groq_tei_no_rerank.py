"""Benchmark 4: groq_tei_no_rerank — Groq + embeddings TEI bge-m3, sin rerank.

Igual que bm_groq_tei_rerank pero con RERANK_ENABLED=false: retrieval por coseno
puro. Comparado contra el benchmark 3 mide el impacto real del cross-encoder.
Requiere: GROQ_API_KEY y un servidor TEI arriba.

Uso:
    uv run python scripts/benchmarks/bm_groq_tei_no_rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib_harness import Variant, common_parser, run_benchmark

VARIANT = Variant(
    slug="groq_tei_no_rerank",
    llm="real",
    embedding_backend="tei",
    rerank=False,
    description="Groq + TEI bge-m3, coseno puro (sin cross-encoder).",
)


def main() -> int:
    args = common_parser().parse_args()
    run_benchmark(VARIANT, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())