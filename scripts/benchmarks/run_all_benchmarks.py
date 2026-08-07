"""Runner unificado: ejecuta los 4 benchmarks y genera el resumen comparativo.

Corre en secuencia las variantes por defecto y escribe, con fecha + run-id
compartidos:
  - scripts/benchmarks/reports/<fecha>_<hora>_<variante>_<runid8>.md
  - scripts/benchmarks/reports/<fecha>_<hora>_SUMMARY_<runid8>.md

Prerequisitos: benchmark 1 sin keys ni red; el resto necesita GROQ_API_KEY
y, para los de TEI, el servidor TEI arriba (si falta, ese benchmark se marca
ERROR/skip en el reporte y la corrida sigue).

Uso:
    uv run python scripts/benchmarks/run_all_benchmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bm_groq_tei_no_rerank import VARIANT as v4
from bm_groq_tei_rerank import VARIANT as v3
from bm_groq_tfidf import VARIANT as v2
from bm_mock_baseline import VARIANT as v1
from lib_harness import make_timestamp, run_benchmark, write_summary

VARIANTS = [v1, v2, v3, v4]


def main() -> int:
    from lib_harness import common_parser

    args = common_parser().parse_args()
    runid = uuid4().hex[:8]
    timestamp = make_timestamp()

    benchs = []
    for variant in VARIANTS:
        benchs.append(run_benchmark(variant, args, runid=runid, timestamp=timestamp))

    summary_path = write_summary(benchs, args.out, timestamp, runid)
    print(f"[SUMMARY] {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())