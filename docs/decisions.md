# Design decisions

Why the agent is built the way it is, what each choice trades away and how it
would change at a larger scale. The original technical-test write-up (Spanish)
is in tag `v0.1.0`, file `SUBMISSION.md`.

## Agent framework: Pydantic AI

**Choice.** The agent runs on [Pydantic AI](https://ai.pydantic.dev/): one
`Agent` with the `get_order_status` tool, the retrieved documents as
per-run instructions, a request limit of three (tool call, answer, one retry)
and an output validator for grounding.

**Why.** v0.1 had a hand-written 60-line loop over LiteLLM. It worked, but it
re-implemented what a small framework already gives: tool schemas from type
hints, message history, retries with feedback to the model, usage limits,
provider-agnostic model names, a test model (`FunctionModel`) and
OpenTelemetry spans. Pydantic AI does this without the graph/chain
abstractions of larger frameworks, which a single-tool agent does not need.

**Trade-off.** A dependency on a young, fast-moving library. The agent is kept
behind `SupportAgent.run() -> AgentReply`, so the interfaces never touch the
framework.

## Guardrails as plain functions, before and after the model

**Choice.** `guardrails.py` decides with regular expressions and `Decimal`
arithmetic: block (prompt injection, card numbers and passwords, off-topic),
refer (billing disputes, legal threats, staff complaints) and escalate (refund
requests over $500). The output check blocks empty answers, escalates an
answer that approves a refund over $500, and refers ungrounded deadlines.

**Why.** These are business rules with legal and financial weight. They must
hold even if the model ignores its instructions, and they must be testable
without a model. Running them first also saves a model call on every message
they settle.

**Trade-off.** Rules miss paraphrases a classifier would catch, and catch some
false positives. The 17 defects found in the v0.1 audit were mostly of this
kind (e.g. "invoice" flagged as a billing dispute, "$1200 refrigerator"
escalated in a warranty question); each fix has a regression test. At scale a
moderation model (e.g. Azure AI Content Safety, Llama Guard) would sit in
front of the rules, not replace them.

## Grounding: an output validator with one retry

**Choice.** Every numeric deadline in the answer ("5-7 business days",
"12 months") must appear in the retrieved documents or in a tool result
(`estimated_delivery`). If not, the validator raises `ModelRetry` with the
offending claims; if the retry still fails, the customer is referred to
support.

**Why.** Inventing a return window or a delivery date is the most harmful
mistake this agent can make, and it is cheap to check exactly. Spellings are
normalised first (`5–7`, `5 to 7`, the non-breaking hyphen some models emit).

**Trade-off.** It checks deadlines only, not every factual claim. A full
groundedness evaluator (LLM-as-judge or NLI) would cover more, at the cost of
latency and a second model per answer.

## Retrieval: BM25 by default, embeddings optional

**Choice.** Lexical BM25 (`bm25s` with an English stemmer and stop words) over
the five documents, one document per chunk, with a score threshold. Setting
`EMBEDDING_MODEL` switches to embeddings through any OpenAI-compatible
endpoint; `RERANK_URL` adds a cross-encoder (TEI `/rerank`).

**Why.** Five short documents, one idea each: BM25 retrieves them correctly
offline, in microseconds, with no model to host. Splitting them would separate,
for example, a refund window from its amount limit. v0.1 used TF-IDF; BM25
handles term frequency and document length better and needs no scikit-learn.

**Trade-off.** Lexical retrieval misses paraphrases with no shared terms
("Can I get my purchases sent to the countryside?" for shipping). Those cases live in
`evals/cases/semantic.yaml` and only run in the bench, where embeddings
(bge-m3) are expected to pass them. Thresholds are per backend
(`LEXICAL_THRESHOLD`, `SEMANTIC_THRESHOLD`) because BM25 and cosine scores live
on different scales. When nothing clears the threshold the agent says it does
not have the information and gives the support email; it does not guess.

**At scale.** Paragraph chunking (~300 tokens with overlap), hybrid search
(BM25 + embeddings, reciprocal rank fusion) and a reranker in a managed index
(Azure AI Search, Databricks Vector Search), filtered by product or tenant.
The `Retriever` protocol is the only seam that changes.

## No vector database

Five documents fit in memory. A vector database would add a service to run and
nothing to measure. The semantic retriever keeps the vectors in a NumPy array.

## Offline mock model

**Choice.** `mock.py` is a Pydantic AI `FunctionModel` that calls the tool for
order ids and otherwise answers from the retrieved document, with a citation.
The agent falls back to it, with a warning, when the provider key is missing.

**Why.** Tests, evals, the Docker smoke test and CI then exercise the real
agent, guardrails, retrieval and validator with no key and no network, and
give the same result every time.

**Trade-off.** The mock proves the plumbing and the policies, not the model's
judgement. `make bench` runs the same cases against the real model.

## Evaluation: promptfoo for end-to-end cases

Each case is one customer message with the expected action and cited
documents, graded by a Python assertion. promptfoo provides the runner, the
variant matrix (`promptfooconfig.bench.yaml`), caching and a results viewer,
replacing the custom evaluator, benchmark runner and report scripts of v0.1
(about 1,100 lines).

## Interfaces share one agent

CLI, HTTP API, Gradio UI and the MCP server all build the same `SupportAgent`,
so no interface can skip a guardrail. The API endpoint is synchronous on
purpose: FastAPI runs it in a thread pool, which keeps the agent code simple
and does not block the event loop.

## Conversation memory sent by the client

The agent takes the earlier turns as an argument instead of storing sessions:
the web chat already keeps the history, the CLI chat accumulates it, and API
callers send it with each message. No session store, no expiry, and any
replica can answer any message. Pydantic AI's `message_history` carries the
turns to the model; no memory library is needed for a single conversation.

Three rules keep the history from weakening the guardrails:

- Only the last six turns are sent, which bounds tokens and cost.
- Earlier customer messages that a guardrail would stop are dropped with their
  reply, so a blocked injection is never sent to the model in a later turn.
- The input rules also run over the previous and the current message
  together, but only an escalation counts there: "I want a refund" followed
  by "it cost $1200" escalates, while an old refused topic does not block
  every later message.

Retrieval uses the current message, and adds the previous question only when
the message matches nothing alone ("and for a blender?").

A client could forge assistant turns. The output guardrails and the grounding
check still run on every answer, and assistant turns never count as sources.

## Two languages, chosen explicitly

Spanish is a first-class language, not a translation layer: each language has
its own documents, BM25 stemmer and stop words, system prompt and fixed
replies, and the guardrail rules are the union of both languages, so a message
cannot dodge a rule by switching language. Deadlines in either language are
normalized to the same canonical form ("2 a 3 días hábiles" → "2-3 business
day") before the grounding check.

The customer picks the language (a UI tab, the API's `language` field, the
CLI's `--lang`) instead of the agent detecting it: detection is unreliable on
short messages such as "ORD-1002?", and a wrong guess would cite documents in
the wrong language. With embeddings, a multilingual model such as bge-m3
could serve both languages from one index; BM25 needs one per language.

## Observability

Logs are text or JSON lines with the action, sources and latency of each chat
request. With `OTEL_EXPORTER_OTLP_ENDPOINT` set and the `otel` extra installed,
Pydantic AI emits OpenTelemetry spans for each run, model request (with token
usage) and tool call, viewable in Jaeger, Phoenix or Logfire. Tracing is off
by default and costs nothing when off.

## From prototype to production

| Prototype | Production |
|---|---|
| Pydantic AI agent in-process | Same agent behind the API, or a managed agent service (e.g. Azure AI Foundry Agent Service) |
| Groq `gpt-oss-20b` | Any provider through the model name; managed identity instead of keys |
| Five documents in code | Policies in a governed store (e.g. Unity Catalog), indexed for hybrid search |
| Order dict | Order service behind an API gateway (auth, rate limits); the customer must own the order |
| Regex guardrails | Rules plus a moderation/PII service and human review queues |
| `.env` secrets | Key Vault or the platform's secret store |
| Gradio / CLI | Web and Teams channels calling `POST /chat` |
| JSON logs, optional traces | Traces, cost and quality dashboards, alerts on escalation and refer rates |

## Known limitations

- Memory lasts one conversation and lives in the client; nothing is stored
  between visits.
- Order lookup does not authenticate the customer.
- Two languages only; adding one means its documents, stop words, rule
  alternations and replies.
- Grounding covers deadlines, not every claim.
- The real-model bench depends on a third-party API and is not run in CI.
