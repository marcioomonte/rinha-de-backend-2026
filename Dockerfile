# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
FROM --platform=linux/amd64 rust:1.95-slim-bookworm AS builder
WORKDIR /build

# Pre-fetch and compile dependencies (heavy: tokio, axum, hyper, ...)
# with dummy source files so the cached layer survives source edits.
COPY Cargo.toml ./
RUN mkdir -p src src/bin \
 && echo "fn main() {}" > src/main.rs \
 && echo "fn main() {}" > src/bin/preprocess.rs \
 && cargo build --release --bin server --bin preprocess \
 && rm -rf src target/release/deps/server* target/release/deps/preprocess*

# Real source
COPY src ./src

# Force a rebuild of the project itself (deps stay cached)
RUN touch src/main.rs src/bin/preprocess.rs \
 && cargo build --release --bin server --bin preprocess

# Reference dataset + build IVF index
COPY resources ./resources
RUN ./target/release/preprocess \
 && ls -lh data/ivf.bin

# ---------- Stage 2: runtime ----------
FROM --platform=linux/amd64 debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/server /app/server
COPY --from=builder /build/data/ivf.bin /app/data/ivf.bin
COPY --from=builder /build/resources/mcc_risk.json /app/resources/mcc_risk.json
COPY --from=builder /build/resources/normalization.json /app/resources/normalization.json

EXPOSE 3000
CMD ["/app/server"]
