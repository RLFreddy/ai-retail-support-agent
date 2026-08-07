ENV_FILE := $(if $(wildcard .env),--env-file=.env,)

.PHONY: build run repl tui demo query evaluate version shell test lint benchmark mcp-smoke clean

build: ## Construir la imagen: docker build -t tiendahogar .
	docker build -t tiendahogar .

run: ## Demo por defecto (modo mock, sin red)
	docker run --rm $(ENV_FILE) tiendahogar

repl: ## REPL interactiva de rich (requiere TTY)
	docker run --rm -it $(ENV_FILE) -e UI=rich tiendahogar python -m src.cli --plain

tui: ## TUI interactiva de Textual (requiere TTY)
	docker run --rm -it $(ENV_FILE) -e UI=textual tiendahogar python -m src.cli

query: ## Respuesta a una consulta: make query Q="¿Dónde está mi pedido ORD-1002?"
	docker run --rm $(ENV_FILE) tiendahogar python -m src.cli --query "$$Q"

evaluate: ## Golden set (retrieval@1, guardrails, grounding)
	docker run --rm $(ENV_FILE) tiendahogar python -m src.evaluate

version: ## Versión y variante del agente
	docker run --rm tiendahogar python -m src.cli --version

shell: ## Shell interactivo dentro del contenedor
	docker run --rm -it $(ENV_FILE) -e UI=rich tiendahogar sh

test: ## Tests dentro del contenedor (pytest)
	docker run --rm tiendahogar pytest

lint: ## Lint dentro del contenedor (ruff)
	docker run --rm tiendahogar ruff check src tests scripts

benchmark: ## Correr las 4 variantes del benchmark + SUMMARY (reportes persistidos en el host)
	mkdir -p scripts/benchmarks/reports
	docker run --rm $(ENV_FILE) \
		-v "$(CURDIR)/scripts/benchmarks/reports:/app/scripts/benchmarks/reports" \
		tiendahogar python scripts/benchmarks/run_all_benchmarks.py

mcp-smoke: ## Probar el servidor MCP (exit 0/1)
	docker run --rm tiendahogar python scripts/mcp_smoke_test.py

clean: ## Eliminar la imagen
	docker rmi tiendahogar