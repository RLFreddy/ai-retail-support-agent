FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so code changes reuse this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra web --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --extra web --no-editable


FROM python:3.12-slim

RUN useradd --create-home --uid 10001 app
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False

USER app
WORKDIR /home/app

EXPOSE 7860
ENTRYPOINT ["tiendahogar"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7860"]
