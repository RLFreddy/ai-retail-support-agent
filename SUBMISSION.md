## Arquitectura propuesta y justificación

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Usuario / CLI                            │
│              (Textual TUI · rich REPL · --query=...)                │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ mensaje
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        GUARDRAILS DE ENTRADA                        │
│   reglas puras (sin red): inyección de prompt / PII → block         │
│   · trato empleado · disputa facturación · legal → refer            │
│   · reembolso > $500 → escalate                                     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ proceed
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         RAG (retriever pluggable)                   │
│   TfidfRetriever (default, offline, sin red)                        │
│               └  SemanticRetriever (TEI bge-m3 / OpenRouter)        │
│   corpus = 5 documentos de política (chunk = documento completo)    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ contexto citado
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   LOOP DE AGENTE (propio, ~60 líneas)               │
│   system + contexto RAG → LiteLLM (proveedor configurable; mock sin key) │
│            │ tool_calls                                             │
│            ▼                                                        │
│   ejecuta consultar_estado_pedido(order_id) → tool result           │
│            │ (max_turns = 2) ←── vuelve al modelo                   │
│            ▼                                                        │
│   respuesta final con citas (Fuente: <título>)                     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     GUARDRAILS DE SALIDA                             │
│   · grounding: verifica números/plazos contra el corpus             │
│   · auto-aprobación de reembolso > $500 → escalate                  │
│   · respuesta vacía → block                                          │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                          Respuesta al usuario
```

**Decisiones clave (trade-offs por límite de tiempo):**

| Decisión | Origen | Trade-off |
|---|---|---|
| **Loop de agente propio** (~60 líneas) en vez de un SDK (OpenAI Agents SDK, LangChain, pydantic-ai) | una sola tool + grounding estricto → un SDK solo añade abstracción | perdemos *structured output* validado de pydantic-ai, pero ganamos legibilidad, tests y control de guardrails/citas |
| **LiteLLM** como capa de modelos | un único contrato `completion` (tool-calling) para Groq, OpenRouter, OpenAI, Anthropic, Azure, Ollama | introduce una abstracción, pero a cambio ningún vendor lock-in y portabilidad directa a Foundry |
| **Guardrails como funciones puras** (`src/guardrails.py`) | bloqueo *duro* (inyección/PII/tema humano/monto) no debe depender del LLM | la capa semántica fina la hace el LLM, pero nunca se expone una respuesta que viole política |
| **Retriever pluggable `BaseRetriever`** | corpus de 5 docs (~700 tokens) | TF-IDF offline basta y evita el costo/latencia de embeddings; el semántico queda listo para crecer |
| **Sin vector DB en el POC** | 5 documentos en memoria | introducir (Chroma/Qdrant/FAISS) sería deuda técnica; se sustituye por Azure AI Search / Databricks sin tocar el agente |

La tool se expone además como **servidor MCP** (`src/mcp_server.py`, FastMCP), estándar de integración de agentes de Microsoft.

## Decisiones técnicas de RAG

| Tema | Decisión | Motivo / cómo escalaría |
|---|---|---|
| **Chunking** | No se aplica: 1 doc = 1 párrafo (~100-150 tokens), una idea cada uno | Partirlos haría que "garantía" recuperase un fragmento sin el contexto del umbral de monto. Con miles de docs: chunking por párrafo (~300 tokens, solapamiento 50) + *hybrid search* (BM25 + embeddings) + reranker + Azure AI Search / Databricks Vector Search por tenant/producto. La interface `BaseRetriever` ya permite enchufarlo. |
| **Embeddings** | Offline: **TF-IDF** (`TfidfVectorizer`, tokenizer propio + stop-words en español). Semántico (opcional): **bge-m3** (TEI local / OpenRouter `qwen/qwen3-embedding-8b` / OpenAI 3-small / Gemini) | bge-m3: buen soporte de español y se sirve local (sin costo por query, sin salida de datos del perímetro). En producción Foundry: embeddings/reranker Azure OpenAI o Databricks Foundation Models. Rerank opcional (`RERANK_ENABLED`) por cross-encoder TEI `bge-reranker-v2-m3` o `litellm.rerank` (nube). |
| **Threshold** | **Por backend** en `.env`: TF-IDF `RELEVANCE_THRESHOLD=0.15` · embeddings `SEMANTIC_THRESHOLD=0.56` | TF-IDF: relevantes ≥ ~0.24, irrelevantes ≤ ~0.14 (calibrado con tests y golden). bge-m3 vive en ~0.3-0.8; con 0.15 pasaba todo y citaba de más; 0.56 separa correcto (0.74) del 2º mejor (0.54). Si **ningún doc supera el umbral**: retriever devuelve vacío y el agente responde que no tiene esa info y remite a `soporte@tiendahogar.example` — **nunca inventa**. |

## Pruebas automatizadas

`uv run pytest tests/` → **toda la suite pasa** (100% mock, determinista, sin red).

| Caso | Archivo | Qué verifica |
|---|---|---|
| RAG → doc correcto | `tests/test_retrieval.py` | garantía recupera el doc `garantia`; irrelevante/baja umbral no devuelve contexto |
| Tool válida / inválida | `tests/test_tools.py` | `ORD-1001` ok; id inexistente → `encontrado: False`, estado `no encontrado` (sin inventar) |
| Guardrail activado | `tests/test_guardrails.py` · `test_agent.py` | reembolso > $500 escala; trato/facturación/legal → referir; salida no soportada → bloqueada; frontera de $500 (exacto no escala) y datos de la tool como fuente autorizada |
| Integración del agente | `tests/test_agent.py` | loop de tool-calling con límite de iteraciones, grounding con citas, fallback "no tengo esa info" |
| Servidor MCP | `tests/test_mcp.py` | tool expuesta con FastMCP y respuesta desde tabla mock (sin red) |
| Firma exacta | `tests/test_tools.py` | `inspect`: `consultar_estado_pedido(order_id: str) -> dict` (como pide el enunciado) |

Golden set: `uv run python -m src.evaluate` → **retrieval@1, guardrails y grounding al 100%**.

## Cómo mapearías esto a producción

| Prototipo (este repo) | Producción (Grupo Mariposa) |
|---|---|
| Loop de agente propio | **Microsoft Foundry** (Azure AI Agent Service) con trazas y telemetría |
| LiteLLM + Groq gpt-oss-20b | Foundation model Azure AI (GPT-4o / phi) o Foundry, *managed identity* |
| 5 docs en memoria | **Databricks/Unity Catalog**: tablas delta + índice vectorial con *hybrid search* |
| Retriever offline | **Azure AI Search** + **Databricks Vector Search** por tenant/producto |
| Tool mock (dict) | Servicio HTTP (contrato OpenAPI) detrás de **Apigee** (auth, rate-limit) |
| Llamada síncrona CLI | Eventos asíncronos por **Kafka** (ingesta políticas, logs, derivaciones a humano) |
| Guardrails locales | bundle de guardrails en la plataforma + **Azure AI Content Safety** (PII, content), monitorización |
| Secrets `.env` | Azure Key Vault / secrets Foundry, *zero-trust* (identidad por servicio) |
| CLI Textual/rich | **Copilot Studio / Power Automate** (mismas reglas en el orquestador) |
| Canal chat (CLI) | **Teams (Bot Framework)** + canales web (reusa tool + guardrails + RAG) |

Además: auth por `order_id` con OAuth2 (Apigee), idempotencia en Kafka (`conversation_id + event_type`), observabilidad por llamada (retrieval_score, latencia, token_usage, FinOps) y gobernanza **NIST AI RMF / ISO 42001**. Si el agente creciera: orquestación con **Pydantic AI/LangGraph**, evaluación con **RAGAS/DeepEval** (pre-release), guardrails semánticos con **Guardrails AI / Content Safety**.

## Limitaciones conocidas

- La suite usa `MockLLM` (determinista, sin red, sin keys). Validar con un LLM **real** es manual: `LLM_MODEL` + key y correr `--query`/CLI antes de desplegar.
- Guardrail de salida "no inventar" = heurística de números/plazos (contrasta con corpus y con `entrega_estimada` de la tool); es la última línea defensiva, no un evaluador semántico completo (en producción: grounding formal en la plataforma).
- Retriever semántico y rerank están testeados sin red (stubs) y **validados en vivo** con TEI local (bge-m3 :8080, reranker :8001): `groq_tei_rerank` / `groq_tei_no_rerank` → **8/8 (100%)** con los 2 contenedores; TF-IDF 7/8 (falla solo en q05, que no dice "envío").
- TUI Textual para terminales interactivas; la prueba CLI automatizada usa `--query` (rich). Sin persistencia de conversación (stateless, dentro del límite del enunciado).

## Tiempo invertido

**~7 horas** (~6 núcleo: agente RAG + tool + guardrails + tests · ~1 h: MCP, evaluador golden y documentación).
