ENV_FILE := $(if $(wildcard .env),--env-file=.env,)
IMAGE := tiendahogar
PROMPTFOO := PROMPTFOO_PYTHON=$(CURDIR)/.venv/bin/python PROMPTFOO_DISABLE_TELEMETRY=1 \
	npx -y promptfoo@0.124.1

.DEFAULT_GOAL := help
.PHONY: help install test lint format typecheck eval bench mcp-smoke check \
	build run chat ask version shell clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

# --- Development (host, needs uv; evals also need Node) ---------------------

install: ## Install the dependencies and the git hooks
	uv sync --locked
	uv run pre-commit install

test: ## Run the test suite (hermetic: no network, no keys)
	uv run pytest

lint: ## Check lint and formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix lint and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Strict type check
	uv run mypy

eval: ## Deterministic evals: mock model, lexical retrieval (fails below 100%)
	uv sync --locked --quiet
	$(PROMPTFOO) eval -c evals/promptfooconfig.yaml --no-cache

bench: ## Real-model variant matrix (needs .env with the API key and the TEI containers)
	uv sync --locked --quiet
	$(PROMPTFOO) eval -c evals/promptfooconfig.bench.yaml --no-cache --env-file .env

mcp-smoke: ## Start the MCP server and call its tool over stdio
	uv run python scripts/mcp_smoke_test.py

check: lint typecheck test eval ## Everything CI runs

# --- Docker -------------------------------------------------------------------

build: ## Build the runtime image
	docker build -t $(IMAGE) .

run: ## Answer the default demo question
	docker run --rm $(ENV_FILE) $(IMAGE)

chat: ## Interactive chat (needs a TTY)
	docker run --rm -it $(ENV_FILE) $(IMAGE) chat

ask: ## Answer one question: make ask Q="Where is my order ORD-1002?"
	docker run --rm $(ENV_FILE) $(IMAGE) ask "$(Q)"

version: ## Print the version and the active configuration
	docker run --rm $(ENV_FILE) $(IMAGE) --version

shell: ## Shell inside the image
	docker run --rm -it $(ENV_FILE) --entrypoint sh $(IMAGE)

clean: ## Remove the image
	docker rmi $(IMAGE)
