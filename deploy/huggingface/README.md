---
title: TiendaHogar Support Agent
emoji: 🛋️
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: RAG, tool calling and guardrails for retail support
---

# TiendaHogar support agent

A customer support agent for a home appliance store: it answers from the
store policies (with citations), looks up orders with a tool, and refers,
escalates or blocks when a guardrail fires.

- Web chat at `/`, HTTP API at `POST /chat` and `GET /health`.
- Add `GROQ_API_KEY` as a Space secret to use the live model; without it the
  Space runs the deterministic offline mock.

Source code: https://github.com/RLFreddy/ai-retail-support-agent
