# Stage 1: Build Python binary with PyInstaller
FROM python:3.12-slim-bookworm AS python-builder

RUN apt-get update && apt-get install -y \
    binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY . .
RUN uv sync --frozen --no-dev
RUN uv pip install pyinstaller
RUN uv run pyinstaller --onefile --name mermaid-ascii src/mermaid_ascii/__main__.py

# Stage 2: Build Rust binary
FROM rust:1.82-slim-bookworm AS rust-builder

WORKDIR /app

COPY src/rust/ src/rust/

RUN cd src/rust && cargo build --release

# Stage 3: Export both binaries via scratch image
FROM scratch AS export
COPY --from=python-builder /app/dist/mermaid-ascii /mermaid-ascii-python
COPY --from=rust-builder /app/src/rust/target/release/mermaid-ascii /mermaid-ascii
