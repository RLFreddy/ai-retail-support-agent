ENV_FILE := $(if $(wildcard .env),--env-file=.env,)

.PHONY: build run repl query eval bench version shell test lint mcp-smoke clean

build: ## Construir la imagen: docker build -t tiendahogar .
	docker build -t tiendahogar .

run: ## Demo por defecto (modo mock, sin red)
	docker run --rm $(ENV_FILE) tiendahogar

repl: ## Interactive chat (needs a TTY)
	docker run --rm -it $(ENV_FILE) tiendahogar tiendahogar chat

query: ## Respuesta a una consulta: make query Q="¿Dónde está mi pedido ORD-1002?"
	docker run --rm $(ENV_FILE) tiendahogar tiendahogar ask "$$Q"

PROMPTFOO := PROMPTFOO_PYTHON=$(CURDIR)/.venv/bin/python PROMPTFOO_DISABLE_TELEMETRY=1 \
	npx -y promptfoo@0.124.1

eval: ## Deterministic evals on the host (mock LLM, offline retrieval; needs uv + Node)
	uv sync --quiet
	$(PROMPTFOO) eval -c evals/promptfooconfig.yaml --no-cache

bench: ## Variant matrix with the real model (needs the API key and the TEI containers)
	uv sync --quiet
	$(PROMPTFOO) eval -c evals/promptfooconfig.bench.yaml --no-cache --env-file .env

version: ## Versión y variante del agente
	docker run --rm tiendahogar tiendahogar --version

shell: ## Shell interactivo dentro del contenedor
	docker run --rm -it $(ENV_FILE) tiendahogar sh

test: ## Tests dentro del contenedor (pytest)
	docker run --rm tiendahogar pytest

lint: ## Lint dentro del contenedor (ruff)
	docker run --rm tiendahogar ruff check src tests scripts

mcp-smoke: ## Probar el servidor MCP (exit 0/1)
	docker run --rm tiendahogar python scripts/mcp_smoke_test.py

clean: ## Eliminar la imagen
	docker rmi tiendahogar