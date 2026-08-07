# Benchmarks — cómo correrlos y cómo leerlos

Guía del harness en `scripts/benchmarks/`: qué mide cada variante, cómo correrlo
y cómo leer los reportes (con un **ejemplo real anotado** `≡≡`). Se generó con:

```bash
uv run python scripts/benchmarks/run_all_benchmarks.py   # 4 variantes + SUMMARY
# o individualmente, p. ej.:
uv run python scripts/benchmarks/bm_mock_baseline.py
```

Cada corrida crea una carpeta `scripts/benchmarks/reports/<fecha>_<runid>/` con un `.md`
por variante + `SUMMARY.md`. El ejemplo es la variante `mock_baseline`
(Mock LLM + TF-IDF, sin red), que siempre corre y es el "patrón de referencia".

---

## Reporte (formato real — 100% prosa)

```
# Benchmark — mock_baseline
≡≡  Nombre de la variante probada.

- Fecha: 2026-08-05 · Run: 684ae2f5 · Queries: default.json (8)
≡≡  run-id único por corrida → nunca se mezclan dos ejecuciones.

## Configuración
| Parámetro | Valor |
| LLM | mock |        ≡≡ mock = sin red (determinista) · real = Groq
| Embeddings | offline | ≡≡ TF-IDF (default, sin red)
| Rerank | off |

## Cómo leer este reporte          ← leyenda autosuficiente (en cada reporte)
Veredicto: ✅ OK = hizo lo esperado · ❌ FAIL = algo difiere
· ⏭ SKIP = servicio externo caído (ej. TEI) · ⚠ ERROR = falló.
Qué debía pasar / Qué pasó (respondió, escaló, derivó, bloqueó)
Fuentes usadas = el documento citado por su nombre.

## Documentos que consulta el agente   ← cuadro de referencia (que pediste)
> Base definida en src/knowledge.py · se citan por su nombre, sin [0]/[4].
| Documento | Definido en | Qué contiene (primeras palabras) |
| Política de garantía | src/knowledge.py | "Todos los electrodomésticos grandes…"
| Política de devoluciones | src/knowledge.py | "Los productos pueden devolverse dentro de 30 días…"
| Tiempos de envío | src/knowledge.py | "Envíos a la capital: 2-3 días hábiles…"
| Reembolsos | src/knowledge.py | "Los reembolsos se procesan en 5-10 días hábiles…"
| Canales de contacto | src/knowledge.py | "Para quejas sobre el trato de un empleado…"
≡≡  Se genera automáticamente desde src.knowledge.DOCUMENTS (fuente única).

## Métricas agregadas
Accuracy (acciones correctas)   7/8 · 87.5%    ← ¿respeta el contrato?
Retrieval recall@k (fuentes)    66.7%         ← ¿recuperó los docs correctos?
Guardrail accuracy              100.0%        ← ¿escaló/derivió cuando debía?
Latencia media (juzgadas)          2 ms

## Resultados                       ← columna prosa, sin tokens `[N]`
| # | Pregunta | Qué debía pasar | Qué pasó | Fuentes usadas | ms | Veredicto |
| q01 | ¿Cuánto dura la garantía de una refrigeradora? | responder citando la garantía (12 meses) | respondió | Política de garantía | 3 | ✅ OK |
| q02 | Reembolso de $1200 | escalar a un supervisor (> $500) | escaló a un supervisor | — | 0 | ✅ OK |
| q05 | ¿Me llegan compras al interior del país? | responder que sí, citando Tiempos de envío | respondió | Ninguna | 1 | ❌ FAIL |
| q08 | ¿Cuál es el clima de Marte? | responder "no tengo esa información" | respondió | Ninguna | 1 | ✅ OK |

**Totales:** 7/8 OK · 1 FAIL · 0 ERROR · 0 SKIP

q05 se lee directo: "esperaba que citara Tiempos de envío, respondió sin citar
nada" → FAIL real del retriever, no del script. El detalle lo dice aún más claro:

### q05 (FAIL) — ¿Me llegan compras al interior del país?
- **Por qué importa:** RAG con sinónimo ('interior del país' ≈ envíos)…
- **Qué debía pasar:** Debe responder que sí se hacen envíos al interior con
  los tiempos, citando el documento de envíos.
- **Qué pasó:** respondió · **Fuentes usadas:** Ninguna · 1 ms
- **Veredicto:** ❌ no citó ningún documento, pero debía citar «Tiempos de envío».
- Respuesta: "Lo siento, no tengo esa información."
≡≡  El lector entiende el juicio sin código: esperaba cita de Envíos y no hubo.
```

---

## Si una variante no puede correr (infra estructural no disponible)

Con TEI apagado las variantes `groq_tei_*` salen `⏭ SKIP`, no FAIL:

```
[groq_tei_rerank] SKIP 8/8 · infra no disponible → reports/<runid>/groq_tei_rerank.md
```

Eso **no** es un fallo del agente: el servidor de los embeddings no está levantado.

## Cómo llegar a datos reales de las variantes TEI

Levanta los **2 contenedores** TEI (bge-m3 es solo embeddings; el cross-encoder
`bge-reranker-v2-m3` solo sirve `/rerank`):

> **Modelos en `data/`:** TEI sirve los pesos desde el volumen que montas (`-v $PWD/data:/data`); por defecto `--model-id` apunta al Hub de HuggingFace y TEI descarga los pesos ahí la primera vez (bge-m3 ≈ 2.3 GB, reranker ≈ 2.6 GB). Para usar una copia local pre-descargada (sin re-descargas ni acceso al Hub), pon los pesos en `./data` y apunta `--model-id` a esa ruta local.

```bash
# 1) Embeddings (bge-m3) — puerto 8080
docker run --rm --gpus all -p 8080:80 -v $PWD/data:/data --pull always \
  --name tei-embed-gpu \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id BAAI/bge-m3 \
  --served-model-name BAAI/bge-m3 \
  --dtype float32 \
  --max-batch-tokens 2048 \
  --max-client-batch-size 32 \
  --max-batch-requests 32 \
  --auto-truncate

# 2) Reranker (bge-reranker-v2-m3) — puerto 8001
docker run --rm --gpus all -p 8001:80 -v $PWD/data:/data --pull always \
  --name tei-rerank-gpu \
  ghcr.io/huggingface/text-embeddings-inference:turing-1.9 \
  --model-id BAAI/bge-reranker-v2-m3 \
  --max-batch-tokens 4096 --max-client-batch-size 16
```

El umbral por backend está en tu `.env`: `RELEVANCE_THRESHOLD=0.15` (TF-IDF) y
`SEMANTIC_THRESHOLD=0.56` (embeddings, otra escala de cosines). Con el harness
puedes probar otro umbral sin editar nada: `--threshold 0.56`.

## Cómo validar escenarios propios

```bash
uv run python -m src.cli --query='¿Cuánta garantía tiene una lavadora?'
```