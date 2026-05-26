# syntax=docker/dockerfile:1.7

# ---------- Stage 1: builder ----------
FROM --platform=linux/amd64 node:22-alpine AS builder
WORKDIR /build

# Build deps for native modules (mmap-io)
RUN apk add --no-cache python3 make g++

# Deps first (cache-friendly)
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src
COPY scripts ./scripts
COPY resources ./resources

# Compile and pre-process the dataset
RUN npx tsc \
 && node --import tsx scripts/preprocess.ts \
 && ls -lh data/refs.bin

# ---------- Stage 2: runtime ----------
FROM --platform=linux/amd64 node:22-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production

COPY package.json package-lock.json ./
# Native modules need build tools to compile during install too
RUN apk add --no-cache --virtual .build-deps python3 make g++ \
 && npm ci --omit=dev \
 && apk del .build-deps \
 && npm cache clean --force

COPY --from=builder /build/dist ./dist
COPY --from=builder /build/data ./data
COPY --from=builder /build/resources/mcc_risk.json ./resources/mcc_risk.json
COPY --from=builder /build/resources/normalization.json ./resources/normalization.json

EXPOSE 3000
CMD ["node", "dist/src/server.js"]
