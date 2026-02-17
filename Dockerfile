FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN uv sync --no-dev

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

WORKDIR /app
COPY --from=builder /app /app

ENTRYPOINT ["uv", "run", "python", "-m", "mermaid_ascii"]
