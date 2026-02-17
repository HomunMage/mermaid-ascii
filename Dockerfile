FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY . .
RUN uv sync --frozen --no-dev
RUN uv pip install pyinstaller
RUN uv run pyinstaller --onefile --name mermaid-ascii src/mermaid_ascii/__main__.py

FROM scratch AS export
COPY --from=builder /app/dist/mermaid-ascii* /
