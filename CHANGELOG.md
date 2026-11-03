## v1.0.0 (2026-11-03)

### Feat

- **ui**: use the Soft theme and re-record the web demo in dark mode
- **api**: serve the agent over HTTP with a Gradio chat UI
- add JSON logs and opt-in OpenTelemetry tracing
- translate the agent to English

### BREAKING CHANGE

- The agent, knowledge base, tool (`get_order_status`) and CLI are now in
  English; the package moved to `tiendahogar` with a single CLI.

### Fix

- **agent**: verify grounding in an output validator with one retry
- **guardrails**: catch paraphrased injections and stop flagging invoice requests
- **guardrails**: parse refund amounts with Decimal and require a refund request

### Refactor

- add domain types for actions and replies
- **agent**: run the agent on Pydantic AI and drop LiteLLM
- **retrieval**: replace TF-IDF with BM25 and drop the Cohere rerank
- **cli**: unify the CLIs and drop Textual
- move the package to src/tiendahogar

## v0.1.0 (2026-08-06)

Original delivery of the technical test, in Spanish: custom agent loop over
LiteLLM, TF-IDF retrieval, Textual TUI and a custom evaluation harness.
