FROM rust:latest AS builder

ARG TARGET
RUN rustup target add ${TARGET}
RUN apt-get update && apt-get install -y \
    gcc-mingw-w64-x86-64 \
    gcc-aarch64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

ARG VERSION=dev

WORKDIR /app
COPY . .
ENV MERMAID_ASCII_VERSION=${VERSION}

# Download released homunc to compile .hom files (detect host arch)
RUN ARCH=$(dpkg --print-architecture) && \
    case "$ARCH" in \
      amd64) HOMUNC_ARCH="x86_64" ;; \
      arm64) HOMUNC_ARCH="aarch64" ;; \
      *) HOMUNC_ARCH="x86_64" ;; \
    esac && \
    wget -q "https://github.com/HomunMage/Homun-Lang/releases/latest/download/homunc-linux-${HOMUNC_ARCH}" -O /usr/local/bin/homunc \
    && chmod +x /usr/local/bin/homunc \
    || true
# build.rs will use homunc from PATH to compile .hom → OUT_DIR/*.rs
RUN cargo build --release --target ${TARGET}

FROM scratch AS export
ARG TARGET
COPY --from=builder /app/target/${TARGET}/release/mermaid-ascii* /
