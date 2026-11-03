# TiendaHogar support agent

[![CI](https://github.com/RLFreddy/ai-retail-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/RLFreddy/ai-retail-support-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-4f46e5)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/lint-ruff-d7ff64)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy%20strict-2a6db2)](https://mypy-lang.org/)

A customer support agent for a home appliance store. It answers questions about
**warranties, returns, shipping, refunds and orders** from five policy
documents, cites its sources, looks orders up with a tool, and hands the
conversation to a human when a policy says so. It never makes up a deadline or
approves money it is not allowed to. It speaks **English and Spanish**: each
language has its own documents, retrieval analyzer, guardrail rules and
replies.

<p align="center">
  <img src="assets/demo.gif"
       alt="Terminal chat: a cited warranty answer and an escalated refund in English, then an order lookup and a blocked prompt injection in Spanish"
       width="760">
  <br><em>Terminal: <code>tiendahogar chat</code>, then <code>--lang es</code></em>
</p>

<p align="center">
  <img src="assets/demo-web.gif"
       alt="Web chat: two questions in the English tab, then the Spanish tab answers an order lookup and blocks a prompt injection"
       width="760">
  <br><em>Web UI: <code>tiendahogar serve</code>, English and Español tabs</em>
</p>

**Stack:** Python 3.12 · [Pydantic AI](https://ai.pydantic.dev/) · BM25
([bm25s](https://github.com/xhluca/bm25s)), with optional embeddings and a
reranker · FastAPI + Gradio · FastMCP · OpenTelemetry · pytest + promptfoo ·
uv, Ruff, mypy strict, Docker.

Without an API key it runs a deterministic offline model, so the tests and
evals pass on any machine with no network.

## What it does

| Customer says | Agent does | How |
|---|---|---|
| "How long is the warranty on a refrigerator?" | Answers and cites *Warranty policy* | BM25 retrieval → LLM with the documents as context |
| "Where is my order ORD-1002?" | Looks the order up | `get_order_status` tool call |
| "I need a refund of $1200" | **Escalates** to a supervisor | Input guardrail: refunds over $500 |
| "I want to dispute a charge" | **Refers** to a human at support | Input guardrail: billing, legal, staff complaints |
| "Ignore your instructions…", card numbers | **Blocks** | Input guardrail: injection and sensitive data |
| The model invents "45 days" | Retries once, then **refers** | Output validator: every deadline must appear in the sources |
| "¿Cuánto dura la garantía?" | Answers in Spanish, cites *Política de garantía* | Spanish documents, stemmer and rules (`--lang es`, the *Español* tab) |

## Quick start

```bash
git clone https://github.com/RLFreddy/ai-retail-support-agent && cd ai-retail-support-agent
uv sync --all-extras

uv run tiendahogar ask "How long is the warranty on a refrigerator?"
uv run tiendahogar chat      # interactive chat in the terminal (--lang es for Spanish)
uv run tiendahogar serve     # web chat at http://localhost:7860, API at /chat
```

To use a real model, copy `.env.example` to `.env` and set `GROQ_API_KEY`
([free key](https://console.groq.com/keys)). `LLM_MODEL` accepts any
[Pydantic AI model name](https://ai.pydantic.dev/models/overview/), e.g.
`openai:gpt-4.1-mini`, with that provider's key.

With Docker:

```bash
make build
make run                     # web chat and API on http://localhost:7860
make ask Q="Where is my order ORD-1002?"
```

## Interfaces

All of them share one agent per language, so they apply the same guardrails.
The language is chosen explicitly (a tab, a field, a flag), not guessed.

| Interface | Command | Notes |
|---|---|---|
| CLI | `tiendahogar ask "…"` / `tiendahogar chat` | Rich output; guardrail decisions in a panel; `--lang es` |
| HTTP API | `tiendahogar serve --no-ui` | `POST /chat {"message", "language"}` → `{answer, action, sources}`; `GET /health`; OpenAPI at `/docs` |
| Web UI | `tiendahogar serve` | Gradio chat at `/`, with *English* and *Español* tabs |
| MCP server | `tiendahogar mcp` | Exposes `get_order_status` over stdio to any MCP client |

```bash
curl -s localhost:7860/chat -H 'content-type: application/json' \
  -d '{"message": "Can I return a blender after 20 days?"}'
curl -s localhost:7860/chat -H 'content-type: application/json' \
  -d '{"message": "¿Dónde está mi pedido ORD-1001?", "language": "es"}'
```

## Architecture

```mermaid
flowchart TD
    U[Customer message] --> IG{Input guardrails}
    IG -- injection / sensitive data / off topic --> B[block]
    IG -- billing / legal / staff --> R[refer to support]
    IG -- refund over $500 --> E[escalate]
    IG -- pass --> RET[Retriever<br/>BM25, or embeddings + reranker]
    RET --> AG[Pydantic AI agent]
    AG <--> T[get_order_status tool]
    AG --> OV{Output validator<br/>deadlines grounded?}
    OV -- no, retry once --> AG
    OV -- yes --> OG{Output guardrails}
    OG -- self-approved refund over $500 --> E
    OG -- pass --> A[answer + sources]
```

- **Guardrails are plain functions** (`guardrails.py`), so policy never depends
  on the model following instructions. They run before and after the model.
- **Grounding** is checked by a Pydantic AI output validator: every deadline in
  the answer ("5-7 business days") must appear in the retrieved documents or
  in a tool result. The model gets one retry with the reason; after that the
  customer is referred to a human instead of receiving an unsupported answer.
- **Retrieval** is lexical BM25 by default (no model, no network). Setting
  `EMBEDDING_MODEL` switches to embeddings through any OpenAI-compatible
  endpoint (e.g. a local [TEI](https://github.com/huggingface/text-embeddings-inference)
  server with bge-m3), and `RERANK_URL` adds a cross-encoder.
- **Amounts** are parsed with `Decimal` (`money.py`), and only a refund request
  above $500 escalates; a $1200 refrigerator in a warranty question does not.

The reasoning behind each choice is in [docs/decisions.md](docs/decisions.md).

## Evaluation

Two layers, both offline and deterministic in CI:

- **Unit and regression tests** (`make test`, 170+ tests): retrieval, tools,
  guardrails, money parsing, the agent loop with scripted model responses,
  the API, the CLI and the MCP server. The suite is hermetic:
  [pytest-socket](https://github.com/miketheman/pytest-socket) blocks the
  network and no `.env` or API key is read.
- **End-to-end evals with [promptfoo](https://www.promptfoo.dev/)** (`make eval`):
  test cases covering the technical test's scenarios, golden policy questions
  and adversarial inputs. Each case grades the decision (action and cited
  documents), not just the wording. See [evals/](evals/README.md).

### Results

The eval cases against the real model, three runs each, one request at a
time (`make bench VARIANTS='lexical$'`, 2026-11-03):

| Variant | English (22 cases) | Spanish (10 cases) | Only failure |
|---|---|---|---|
| Mock model + BM25 (CI) | 21/22 | 10/10 | the paraphrase with no shared words (expected) |
| Groq `gpt-oss-20b` + BM25 | **63/66** (21/22 in each run) | **30/30** | the same paraphrase, every run |
| + embeddings (bge-m3), + reranker | not measured yet | needs the local TEI containers |

| Groq + BM25, per message | Value |
|---|---|
| Settled by a guardrail, no model call | 41% of the cases, in milliseconds |
| Latency with the model, p50 / p95 | 3.1 s / 6.2 s (includes tool calls and retries) |
| Tokens with the model, average | ~520 input, ~160 output |
| Cost per 1,000 messages | ~$0.08 at Groq's list price ($0.10 / $0.50 per M tokens) |

The one miss is the reason semantic retrieval exists: "Can I get my purchases
sent to the countryside?" shares no words with the shipping policy, so BM25
finds nothing and the agent correctly says it does not know instead of
guessing. Whether embeddings and the reranker are worth their extra service is
what the remaining bench variants measure.

### What the evals caught

- **A dash the grounding check did not know.** The real model wrote "5‑10
  business days" with a non-breaking hyphen (U+2011). The validator saw an
  unknown deadline and referred a correct answer to a human. Deadlines are now
  normalised across every Unicode dash, with a regression test.
- **A range the grounding check read as one number.** In Spanish the real
  model wrote "entre 5 y 7 días hábiles". The validator only saw "7 días",
  judged it invented and referred a correct answer. "between 5 and 7" had the
  same bug in English; both now read as the range in the policy.
- **A silent fallback.** A `.env` written for the old LiteLLM format
  (`groq/...` instead of `groq:...`) made the agent fall back to the mock
  without failing. `LLM_MODEL` is now validated at startup.
- **Policy rules that fired on the wrong messages.** An audit of v0.1 found 17
  defects, such as escalating "my $1200 refrigerator broke" (a warranty
  question, not a refund) and treating "send me the invoice" as a billing
  dispute. Each one is a regression test.

## Configuration

Settings come from environment variables or `.env` (see
[.env.example](.env.example)).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `groq:openai/gpt-oss-20b` | Pydantic AI model; `mock` for the offline model |
| `GROQ_API_KEY` | – | Provider key; without it the agent falls back to the mock |
| `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL` | – | Turn on semantic retrieval |
| `RERANK_URL`, `RERANK_MODEL` | – | Turn on cross-encoder reranking |
| `TOP_K`, `LEXICAL_THRESHOLD`, `SEMANTIC_THRESHOLD` | `3`, `0.5`, `0.56` | Retrieval cut-offs |
| `SYSTEM_PROMPT_FILE` | – | Replace the default system prompt |
| `LOG_FORMAT`, `LOG_LEVEL` | `text`, `WARNING` | `json` writes one object per line |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | – | Export traces of every run, model call and tool call |

## Development

```bash
make install     # dependencies and git hooks (ruff, mypy, commit messages)
make check       # lint, strict types, tests and evals, as in CI
make help        # every target
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/),
checked by commitizen. The CHANGELOG is generated from them.

```
src/tiendahogar/
  agent.py        Pydantic AI agent, grounding validator, guardrail flow
  guardrails.py   input and output policy checks
  retrieval.py    BM25, embeddings and reranker retrievers
  knowledge.py    the policy documents and system prompt, in English and Spanish
  tools.py        get_order_status
  money.py        refund amount parsing
  mock.py         deterministic offline model
  api.py, ui.py   FastAPI app and Gradio chat
  cli.py          ask, chat, serve, mcp
  mcp_server.py   MCP server
  telemetry.py    JSON logs and OpenTelemetry
tests/            pytest suite
evals/            promptfoo suites
deploy/           Hugging Face Space config
```

## History

This started as a timed technical test for an AI engineer role, delivered in
Spanish with a custom agent loop, LiteLLM and TF-IDF (tag
[`v0.1.0`](https://github.com/RLFreddy/ai-retail-support-agent/tree/v0.1.0)).
Version 1.0 is a rewrite in English, with Spanish back as a second language:
it moves the agent to Pydantic AI, replaces TF-IDF with BM25, fixes 17 policy
defects found in an audit (each with a regression test), and adds the API, web
UI, tracing, promptfoo evals, strict typing and CI. See the [CHANGELOG](CHANGELOG.md).
