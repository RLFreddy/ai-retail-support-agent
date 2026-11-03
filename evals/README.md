# Evals

End-to-end evaluation of the support agent with [promptfoo](https://www.promptfoo.dev/).
Each case sends one customer message to the agent and grades its **decision**
(action and cited documents), not just its wording.

| File | Purpose |
|---|---|
| `promptfooconfig.yaml` | Fast suite: mock LLM + offline retrieval. No network, no keys. |
| `promptfooconfig.bench.yaml` | Variant matrix with the real model, with and without TEI and reranking. |
| `provider.py` | promptfoo provider that builds the agent for a variant and returns `{answer, action, sources}`. |
| `assertions.py` | Default assertion: `expected_action`, `expected_sources`, `includes_sources`, `forbidden`. |
| `cases/exam.yaml` | The cases of the technical test. |
| `cases/golden.yaml` | Retrieval and policy cases. |
| `cases/hard.yaml` | Adversarial and edge cases. |
| `cases/semantic.yaml` | Paraphrases only semantic retrieval can match (bench only; lexical variants are expected to miss them). |

## Running

```bash
make eval    # deterministic suite; exit code 100 if any case fails
make bench   # real-model matrix, 3 runs each; needs the API key in .env and the TEI containers
make bench VARIANTS='lexical$'   # only the variants that need no TEI container
npx promptfoo@0.124.1 view   # browse the last results
```

Both targets need [uv](https://docs.astral.sh/uv/) and Node.js; promptfoo runs
the provider with the project's virtualenv (`PROMPTFOO_PYTHON=.venv/bin/python`).

## Adding a case

```yaml
- description: "hard · what this case protects"
  vars:
    query: "customer message"
    expected_action: answer        # answer | refer | escalate | block
    includes_sources: "1"          # documents that must be cited
    forbidden: "45 days"           # text that must not appear in the answer
```

Documents are referenced by index: 0 warranty, 1 returns, 2 shipping,
3 refunds, 4 contact channels.
