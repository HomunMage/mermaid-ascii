FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ARG TARGET=x86_64-unknown-linux-gnu

WORKDIR /app
COPY . .
RUN uv build

FROM scratch AS export
ARG TARGET=x86_64-unknown-linux-gnu
COPY --from=builder /app/dist/* /
