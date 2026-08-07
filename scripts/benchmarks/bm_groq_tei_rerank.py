"""Benchmark 3: groq_tei_rerank — Groq + embeddings TEI bge-m3 + cross-encoder.

Embeddings locales (dentro del perímetro) con rerank vía el cross-encoder
BAAI/bge-reranker-v2-m3 de tu servidor TEI.
Requiere: GROQ_API_KEY y un servidor TEI arriba (default http://localhost:8080/v1
o el TEI_BASE_URL de tu .env).

Para levantar TEI:
    docker run --rm --gpus all -p 8080:80 -v $PWD/data:/data --pull always \\
      --name tei-embed-gpu \\
      ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \\
      --model-id BAAI/bge-m3 \\
      --served-model-name BAAI/bge-m3 \\
      --dtype float32 --max-batch-tokens 2048 \\
      --max-client-batch-size 32 --max-batch-requests 32 --auto-truncate

Uso:
    uv run python scripts/benchmarks/bm_groq_tei_rerank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib_harness import Variant, common_parser, run_benchmark

VARIANT = Variant(
    slug="groq_tei_rerank",
    llm="real",
    embedding_backend="tei",
    rerank=True,
    description="Groq + TEI bge-m3 + cross-encoder bge-reranker-v2-m3 (local).",
)


def main() -> int:
    args = common_parser().parse_args()
    run_benchmark(VARIANT, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())