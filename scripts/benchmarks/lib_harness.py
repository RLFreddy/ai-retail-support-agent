"""Lógica común de los benchmarks de TiendaHogar.

Cada benchmark (``scripts/benchmarks/bm_*.py``) define su :class:`Variant` y
delega en :func:`run_benchmark`, que:
  1. carga las queries (``scripts/queries/*.json``),
  2. construye el agente con Settings sobre-escritos (sin tocar ``.env``),
  3. ejecuta cada query con cronómetro y captura acción/fuentes/respuesta/scores,
  4. escribe un reporte markdown en ``scripts/benchmarks/reports/``.

``run_all_benchmarks`` reutiliza el mismo harness y además genera un SUMMARY
(queries × variantes) compartiendo fecha + run-id.

Convención de nombres: prefijo ``bm_`` (benchmark), igual que el ecosistema
pyperformance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"
DEFAULT_QUERIES = SCRIPTS_DIR / "queries" / "default.json"
REPORTS_DIR = SCRIPTS_DIR / "benchmarks" / "reports"
DEFAULT_TEI_URL = "http://localhost:8080/v1"

#: Diccionario acción → texto en español (modo presente / pasado).
_ACTION_ES = {
    "answer": "responder",
    "refer": "derivar a un humano",
    "escalate": "escalar a un supervisor",
    "block": "bloquear por seguridad",
    "reject": "rechazar",
    "off_topic": "fuera de tema",
}
_PAST_ES = {
    "answer": "respondió",
    "refer": "derivó a un humano",
    "escalate": "escaló a un supervisor",
    "block": "bloqueó",
    "reject": "rechazó",
}


# --------------------------------------------------------------------------- #
# Base de conocimiento: se importa de src (fuente única, nada duplicado)
# --------------------------------------------------------------------------- #
def _knowledge() -> list[object]:
    from src.knowledge import DOCUMENTS

    return list(DOCUMENTS)


def _snippet(text: str, limit: int = 140) -> str:
    """Primera frase del contenido, truncada para el cuadro del reporte."""
    text = " ".join(text.split())
    cut = text.find(". ")
    head = (text[: cut + 1] if cut != -1 else text).strip()
    return head if len(head) <= limit else f"{text[:limit].rstrip()}…"


def _titles(indices: list[int]) -> str:
    """Títulos de los documentos por índice, en prosa (fuente única: DOCUMENTS)."""
    docs = _knowledge()
    titles = [docs[i].title for i in indices if 0 <= i < len(docs)]
    return ", ".join(titles)


def _action_es(action: str, past: bool = False) -> str:
    table = _PAST_ES if past else _ACTION_ES
    return table.get(action, action or "—")


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """Una combinación probada: LLM × embeddings × rerank."""

    slug: str
    llm: str  # "mock" | "real"
    embedding_backend: str  # offline | tei | openrouter | openai | gemini
    rerank: bool = False
    description: str = ""


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str
    expected_action: str | None = None
    expected_sources: list[int] | None = None
    expected_short: str = ""
    expected_hint: str = ""
    why: str = ""


# --------------------------------------------------------------------------- #
# Carga de queries / CLI
# --------------------------------------------------------------------------- #
def load_queries(path: Path) -> list[QueryCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        QueryCase(
            id=item["id"],
            query=item["query"],
            expected_action=item.get("expected_action"),
            expected_sources=item.get("expected_sources"),
            expected_short=item.get("expected_short", ""),
            expected_hint=item.get("expected_hint", ""),
            why=item.get("why", ""),
        )
        for item in data
    ]


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark del agente TiendaHogar")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                        help=f"archivo JSON de queries (default: {DEFAULT_QUERIES.relative_to(ROOT)})")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR,
                        help="directorio de reportes (default: scripts/benchmarks/reports)")
    parser.add_argument("--top-k", type=int, default=None, help="top_k del retriever")
    parser.add_argument("--threshold", type=float, default=None, help="umbral de relevancia")
    return parser


# --------------------------------------------------------------------------- #
# Construcción del agente
# --------------------------------------------------------------------------- #
def build_settings(variant: Variant, base=None):
    """Settings con los overrides de la variante, preservando el resto de .env."""
    from src.config import Settings, get_settings

    base = base or get_settings()
    kwargs: dict = {
        "embedding_backend": variant.embedding_backend,
        "rerank_enabled": variant.rerank,
    }
    if variant.embedding_backend == "tei" and not base.tei_base_url:
        kwargs["tei_base_url"] = DEFAULT_TEI_URL
    if variant.rerank and variant.embedding_backend != "tei":
        kwargs["rerank_model"] = base.rerank_model or "cohere/rerank-v3.5"
    return Settings(**kwargs)


def build_agent(variant: Variant, settings=None, top_k=None, threshold=None):
    from src.agent import SupportAgent
    from src.config import get_settings
    from src.llm import MockLLM, build_client
    from src.retrieval import build_retriever

    base = get_settings()
    settings = settings or build_settings(variant)
    llm = MockLLM() if variant.llm == "mock" else build_client(settings)
    retriever = build_retriever(settings)
    return SupportAgent(
        llm=llm,
        retriever=retriever,
        max_turns=base.max_turns,
        top_k=top_k or base.top_k,
        relevance_threshold=threshold if threshold is not None else settings.retrieval_threshold,
        system_prompt=base.system_prompt_text,
    )


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
def _is_infra_error(exc: Exception) -> bool:
    """¿El fallo es de infraestructura (red/TEI/API caída) y no del agente?"""
    msg = f"{type(exc).__name__}: {exc}"
    return any(
        token in msg.lower()
        for token in (
            "connection error",
            "connection refused",
            "connecttimeout",
            "timeout",
            "cannot connect",
            "network unreachable",
            "errno 111",
            "name or service not known",
        )
    )


def _status(case: QueryCase, action: str, sources: list[int]) -> str:
    if case.expected_action is not None and case.expected_action != action:
        return "FAIL"
    if case.expected_sources is not None and set(sources) != set(case.expected_sources):
        return "FAIL"
    if case.expected_action is None and case.expected_sources is None:
        return "info"
    return "OK"


def _verdict(case: QueryCase, action: str, sources: list[int]) -> str:
    """Explica el porqué del veredicto en español (solo lo que difiere)."""
    parts = []
    if case.expected_action is not None and case.expected_action != action:
        parts.append(
            f"hizo «{_action_es(action, past=True)}», pero debía «{_action_es(case.expected_action)}»"
        )
    if case.expected_sources is not None and set(sources) != set(case.expected_sources):
        if not sources:
            parts.append(
                f"no citó ningún documento, pero debía citar «{_titles(case.expected_sources)}»"
            )
        else:
            parts.append(
                f"citó «{_titles(sources)}» en lugar de «{_titles(case.expected_sources)}»"
            )
    return " · ".join(parts) or "sin criterios de juicio (caso informativo)"


def run_cases(agent, cases: list[QueryCase]) -> list[dict]:
    results: list[dict] = []
    for case in cases:
        t0 = time.monotonic()
        try:
            resp = agent.run(case.query)
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            scores: dict[str, float] = {}
            skip_reason = ""
            try:
                for r in agent.retriever.retrieve(
                    case.query, top_k=agent.top_k, threshold=agent.relevance_threshold
                ):
                    scores[r.doc_id] = round(r.score, 4)
            except Exception as exc:  # noqa: BLE001 — frontera del harness: nunca rompe el reporte
                if _is_infra_error(exc):
                    skip_reason = _infra_label(exc)
                    scores = {}
                else:
                    scores = {"<error>": 0.0}
            results.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "why": case.why,
                    "expected_hint": case.expected_hint,
                    "expected_short": case.expected_short,
                    "expected_action": case.expected_action,
                    "expected_sources": case.expected_sources,
                    "action": resp.action,
                    "sources": list(resp.sources),
                    "answer": resp.answer,
                    "scores": scores,
                    "elapsed_ms": elapsed_ms,
                    "raw_steps": resp.raw_steps,
                    "skip_reason": skip_reason,
                    "status": (
                        "SKIP" if skip_reason else _status(case, resp.action, list(resp.sources))
                    ),
                    "verdict": _verdict(case, resp.action, list(resp.sources)),
                }
            )
        except Exception as exc:  # noqa: BLE001 — frontera del harness: captura fallos de red/API
            results.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "why": case.why,
                    "expected_hint": case.expected_hint,
                    "expected_short": case.expected_short,
                    "expected_action": case.expected_action,
                    "expected_sources": case.expected_sources,
                    "action": "",
                    "sources": [],
                    "answer": "",
                    "scores": {},
                    "elapsed_ms": round((time.monotonic() - t0) * 1000),
                    "raw_steps": 0,
                    "status": "SKIP" if _is_infra_error(exc) else "ERROR",
                    "verdict": _verdict(case, "", []),
                    "skip_reason": (
                        _infra_label(exc) if _is_infra_error(exc) else ""
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return results


# --------------------------------------------------------------------------- #
# Reporte markdown
# --------------------------------------------------------------------------- #
def _documents_box() -> list[str]:
    """Cuadro «Documentos que consulta el agente» (fuente única: src.knowledge)."""
    docs = _knowledge()
    lines = [
        "| Documento | Definido en | Qué contiene (primeras palabras) |",
        "|---|---|---|",
    ]
    for doc in docs:
        lines.append(f"| {doc.title} | `src/knowledge.py` | «{_snippet(doc.content)}» |")
    return lines


def _sources_cell(r: dict) -> str:
    """Columna 'Fuentes {usadas}': títulos reales, 'Ninguna' o '—' en prosa."""
    if r["status"] in ("ERROR", "SKIP"):
        return "—"
    if r["sources"]:
        return _titles(r["sources"])
    return "Ninguna"


def _happened_cell(r: dict) -> str:
    """Columna 'Qué pasó': la acción del agente en español."""
    if r["status"] in ("ERROR", "SKIP"):
        return "—"
    return _action_es(r["action"], past=True)


def _metrics(results: list[dict]) -> dict[str, float | int]:
    """Métricas agregadas sobre las filas juzgadas (OK/FAIL), SKIP queda fuera."""
    judged = [r for r in results if r["status"] in ("OK", "FAIL")]
    ok = sum(1 for r in judged if r["status"] == "OK")
    recall_num = recall_den = 0
    guard_num = guard_den = 0
    for r in judged:
        if r["expected_sources"] is not None:
            recall_den += 1
            recall_num += int(set(r["sources"]) == set(r["expected_sources"]))
        if r["expected_action"] is not None:
            guard_den += 1
            guard_num += int(r["expected_action"] == r["action"])
    latencies = [r["elapsed_ms"] for r in judged]
    return {
        "judged": len(judged),
        "ok": ok,
        "recall": recall_num / recall_den if recall_den else None,
        "guardrail": guard_num / guard_den if guard_den else None,
        "latency_avg": round(sum(latencies) / len(latencies)) if latencies else None,
    }


def render_report(variant: Variant, cases: list[QueryCase], results: list[dict],
                  timestamp: str, runid: str, queries_path: Path) -> str:
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    err = sum(1 for r in results if r["status"] == "ERROR")
    skip = sum(1 for r in results if r["status"] == "SKIP")
    judged = ok + fail + err
    m = _metrics(results)
    acc = f"{m['ok']}/{m['judged']} · {m['ok']/m['judged']:.1%}" if m["judged"] else "— (0 filas juzgadas, todo SKIP)"

    lines = [
        f"# Benchmark — {variant.slug}",
        "",
        (
            f"- **Fecha:** {timestamp} · **Run:** `{runid}` · "
            f"**Queries:** {queries_path.name} ({len(results)})"
        ),
        f"- **Descripción:** {variant.description or '—'}",
        "",
        "## Configuración",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| LLM | `{variant.llm}` |",
        f"| Embeddings | `{variant.embedding_backend}` |",
        f"| Rerank | `{'on' if variant.rerank else 'off'}` |",
        "",
        "## Cómo leer este reporte",
        "",
        "**Veredicto** por pregunta: ✅ `OK` = hizo lo esperado · ❌ `FAIL` = algo difiere",
        "· ⏭ `SKIP` = no evaluable (servicio externo caído, ej. TEI) · ⚠ `ERROR` = falló.",
        "**Qué debía pasar** — lo que la respuesta correcta debía hacer (en palabras).",
        "**Qué pasó** — la acción real del agente: *respondió*, *escaló a un supervisor*,",
        "*derivó a un humano*, *bloqueó*.",
        "**Fuentes usadas** — el documento de la base que citó. «Ninguna» = no citó nada",
        "(el veredicto indica si eso estaba bien, ej. fuera de tema) · «—» = no aplica.",
        "",
        "## Documentos que consulta el agente",
        "",
        "> Base de la respuesta de conocimiento definida en ``src.knowledge.py``. El",
        "> retriever busca la respuesta dentro de estos documentos y se citan por su nombre:",
        "",
    ]
    lines += _documents_box()
    lines += [
        "",
        "## Métricas agregadas",
        "",
        "| Métrica | Valor |",
        "|---|---|",
        f"| Accuracy (acciones correctas) | {acc} |",
        f"| Retrieval recall@k (fuentes exactas) | {m['recall']:.1%} |" if m["recall"] is not None else "| Retrieval recall@k (fuentes exactas) | — (sin casos con fuentes) |",
        f"| Guardrail accuracy | {m['guardrail']:.1%} |" if m["guardrail"] is not None else "| Guardrail accuracy | — |",
        f"| Latencia media (juzgadas) | {m['latency_avg']} ms |" if m["latency_avg"] is not None else "| Latencia media (juzgadas) | — |",
        "",
        "## Resultados",
        "",
        "| # | Pregunta | Qué debía pasar | Qué pasó | Fuentes usadas | ms | Veredicto |",
        "|---|----------|-----------------|----------|----------------|----|-----------|",
    ]
    for r in results:
        mark = {"OK": "✅", "FAIL": "❌", "ERROR": "⚠", "SKIP": "⏭", "info": "▪"}[r["status"]]
        lines.append(
            f"| {r['id']} | {r['query'][:40]}{'…' if len(r['query']) > 40 else ''} "
            f"| {r['expected_short'] or '—'} | {_happened_cell(r)} "
            f"| {_sources_cell(r)} | {r['elapsed_ms']} | {mark} {r['status']} |"
        )
    total_note = f"**Totales:** {ok}/{judged} OK · {fail} FAIL · {err} ERROR · {skip} SKIP"
    if skip:
        total_note += " (SKIP = infraestructura no disponible, ver detalle)"
    lines += [
        "",
        total_note,
        "",
        "## Detalle por query",
        "",
    ]
    for r in results:
        lines.append(f"### {r['id']} ({r['status']}) — {r['query']}")
        if r.get("why"):
            lines.append(f"- **Por qué importa:** {r['why']}")
        if r.get("expected_hint"):
            lines.append(f"- **Qué debía pasar:** {r['expected_hint']}")
        if r["status"] == "ERROR":
            lines.append(f"- ❌ `{r['error']}`")
            lines.append("")
            continue
        if r["status"] == "SKIP":
            lines.append(f"- ⏭ {r['skip_reason']}. No se evaluó esta query en esta variante.")
            lines.append("")
            continue
        lines.append(
            f"- **Qué pasó:** {_happened_cell(r)} · **Fuentes usadas:** {_sources_cell(r)} "
            f"· **Duración:** {r['elapsed_ms']} ms"
        )
        if r["status"] == "FAIL":
            lines.append(f"- **Veredicto:** ❌ {r['verdict']}.")
        elif r["status"] in ("OK", "info"):
            lines.append("- **Veredicto:** ✅ coincide con lo esperado.")
        if r["status"] == "info":
            lines.append("- *(Caso informativo: no había criterio esperado para comparar.)*")
        if r["scores"]:
            score_cells = []
            for doc_id, score in sorted(r["scores"].items(), key=lambda kv: kv[1], reverse=True):
                score_cells.append(f"{doc_id} = {score}")
            lines.append(
                f"- **Puntajes de relevancia del buscador** (detalle técnico): "
                f"{' · '.join(score_cells)}"
            )
        lines.append("- Respuesta:")
        for para in r["answer"].strip().splitlines() or ["*(vacía)*"]:
            lines.append(f"  > {para}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _run_dir(out_dir: Path, timestamp: str, runid: str) -> Path:
    """Carpeta única por corrida: scripts/benchmarks/reports/<timestamp>_<runid>/."""
    d = out_dir / f"{timestamp}_{runid}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_report(variant: Variant, results: list[dict], out_dir: Path,
                 timestamp: str, runid: str, queries_path: Path) -> Path:
    fname = _run_dir(out_dir, timestamp, runid) / f"{variant.slug}.md"
    fname.write_text(
        render_report(variant, [], results, timestamp, runid, queries_path),
        encoding="utf-8",
    )
    return fname


def make_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC).astimezone()).strftime("%Y-%m-%d_%H-%M")


def _suppress_litellm_noise() -> None:
    """Silencia el spam de errores/footer de litellm en stdout (harness limpio)."""
    try:
        import litellm

        litellm.suppress_debug_info = True
        litellm.suppress_possible_openai_completions_param_warning = True
    except ImportError:
        pass


def _infra_label(exc: Exception) -> str:
    msg = " ".join(str(exc).split())
    return f"infraestructura no disponible: {type(exc).__name__}{(': ' + msg[:90]) if msg else ''}"


def run_benchmark(variant: Variant, args: argparse.Namespace | None = None,
                  runid: str | None = None, timestamp: str | None = None) -> dict:
    """Ejecuta el benchmark de una variante y escribe su reporte."""
    _suppress_litellm_noise()
    args = args or argparse.Namespace(
        queries=DEFAULT_QUERIES, out=REPORTS_DIR, top_k=None, threshold=None
    )
    runid = runid or uuid4().hex[:8]
    timestamp = timestamp or make_timestamp()
    cases = load_queries(args.queries)
    agent = build_agent(variant, top_k=args.top_k, threshold=args.threshold)
    results = run_cases(agent, cases)
    path = write_report(variant, results, args.out, timestamp, runid, args.queries)

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    err = sum(1 for r in results if r["status"] == "ERROR")
    skip = sum(1 for r in results if r["status"] == "SKIP")
    judged = ok + fail + err
    pct = f"{ok/judged:.1%}" if judged else "—"
    if judged:
        base = f"OK {ok}/{judged} ({pct}) · FAIL {fail}"
        extra = f"SKIP {skip} (infra no disponible)" if skip else f"ERROR {err}"
        print(f"[{variant.slug}] {base} · {extra} → {path}")
    else:
        print(f"[{variant.slug}] SKIP {skip}/{len(results)} · infra no disponible → {path}")
    return {"variant": variant, "results": results, "path": path,
            "timestamp": timestamp, "runid": runid}


# --------------------------------------------------------------------------- #
# SUMMARY comparativo
# --------------------------------------------------------------------------- #
def render_summary(benchs: list[dict], timestamp: str, runid: str) -> str:
    variants = [b["variant"] for b in benchs]
    all_results: dict[str, dict] = {}
    for b in benchs:
        for r in b["results"]:
            all_results.setdefault(r["id"], {})[b["variant"].slug] = r

    lines = [
        f"# Resumen comparativo — {timestamp} (run `{runid}`)",
        "",
        "| Variante | LLM | Embeddings | Rerank |",
        "|---|---|---|---|",
    ]
    for v in variants:
        lines.append(f"| `{v.slug}` | {v.llm} | {v.embedding_backend} | {v.rerank} |")
    lines += [
        "",
        "## Matriz de resultados (OK / FAIL / ERROR / info)",
        "",
    ]
    header = "| # | Query | " + " | ".join(f"`{v.slug}`" for v in variants) + " |"
    sep = "|---|-------|" + "".join("---|" for _ in variants)
    lines += [header, sep]
    totals = {v.slug: 0 for v in variants}
    judged = {v.slug: 0 for v in variants}
    skipped = {v.slug: 0 for v in variants}
    for qid, by_variant in all_results.items():
        cells = []
        for v in variants:
            r = by_variant.get(v.slug)
            if r is None:
                cells.append("—")
            else:
                cells.append({"OK": "✅", "FAIL": "❌", "ERROR": "⚠", "SKIP": "⏭", "info": "▪"}[r["status"]])
                if r["status"] in ("OK", "FAIL", "ERROR"):
                    judged[v.slug] += 1
                    totals[v.slug] += int(r["status"] == "OK")
                elif r["status"] == "SKIP":
                    skipped[v.slug] += 1
        q = by_variant[variants[0].slug]["query"]
        lines.append(f"| {qid} | {q[:44]}{'…' if len(q) > 44 else ''} | " + " | ".join(cells) + " |")
    lines.append("| **Totales** | | " + " | ".join(
        (
            f"**{totals[v.slug]}/{judged[v.slug]}** ({skipped[v.slug]} ⏭)"
            if skipped[v.slug] and judged[v.slug]
            else (
                f"**{totals[v.slug]}/{judged[v.slug]}**"
                if judged[v.slug]
                else f"**{skipped[v.slug]} ⏭ (infra)**"
            )
        )
        for v in variants
    ) + " |")
    lines += [
        "",
        "## Latencia por query (ms)",
        "",
        header,
        sep,
    ]
    for qid, by_variant in all_results.items():
        cells = []
        for v in variants:
            r = by_variant.get(v.slug)
            cells.append(str(r["elapsed_ms"]) if r and r["status"] not in ("ERROR", "SKIP") else "—")
        q = by_variant[variants[0].slug]["query"]
        lines.append(f"| {qid} | {q[:44]}{'…' if len(q) > 44 else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines).rstrip() + "\n"


def write_summary(benchs: list[dict], out_dir: Path, timestamp: str, runid: str) -> Path:
    path = _run_dir(out_dir, timestamp, runid) / "SUMMARY.md"
    path.write_text(render_summary(benchs, timestamp, runid), encoding="utf-8")
    return path
