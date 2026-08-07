"""Benchmark 1: mock_linea_base — Mock LLM · TF-IDF · sin rerank (sin red).

Línea base determinista: sin LLM real, sin embeddings, sin reranker. Corre sin
API key y sin red; es la referencia mínima que cualquier revisor puede ejecutar.

Uso:
    uv run python scripts/benchmarks/bm_mock_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib_harness import Variant, common_parser, run_benchmark

VARIANT = Variant(
    slug="mock_baseline",
    llm="mock",
    embedding_backend="offline",
    rerank=False,
    description="Mock LLM + TF-IDF offline: determinista, sin red.",
)


def main() -> int:
    args = common_parser().parse_args()
    run_benchmark(VARIANT, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())