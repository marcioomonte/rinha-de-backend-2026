# Fraud Detection Backend — Design Document

**Status:** Approved
**Date:** 2026-05-25
**Author:** Marcio Monte (com assistência didática)
**Context:** Rinha de Backend 2026 — Fraud Detection (busca vetorial)

---

## 1. Objetivo

Construir um backend que receba transações de cartão de crédito via HTTP, vetorize cada uma em 14 dimensões, encontre os 5 vetores mais próximos em um dataset de 3 milhões de referências rotuladas e responda se a transação deve ser aprovada ou negada.

A submissão é regida pelas regras da Rinha de Backend 2026 (ver `docs/br/`):

- Endpoints `GET /ready` e `POST /fraud-score` na porta 9999.
- Topologia mínima: load balancer round-robin + 2 instâncias da API.
- Limite total: 1 CPU e 350 MB de RAM somando todos os serviços.
- `docker-compose.yml` com imagens públicas linux/amd64, rede bridge, sem privileged.

### Objetivo deste projeto especificamente

- Aprender, na prática, o ciclo completo: vetorização, busca vetorial (KNN brute force), Docker, load balancing.
- Submeter uma solução **correta e funcional** com pontuação positiva.
- Estabelecer uma base sobre a qual otimizações de performance possam ser iteradas depois (worker threads, índices ANN, etc.).

### Não-objetivos (desta primeira versão)

- Ganhar a Rinha. Não vamos para Rust/Zig/SIMD nesta iteração.
- Implementar índice ANN (HNSW, IVF) na primeira versão — fica para iteração futura.
- Otimização agressiva por SIMD/threads — também fica para iteração futura.

---

## 2. Stack

| Camada | Escolha | Justificativa |
|---|---|---|
| Linguagem | TypeScript (Node 22) | Familiaridade do autor; runtime adequado; ecosistema de bibliotecas |
| Servidor HTTP | Fastify | Um dos servidores Node mais rápidos com boa ergonomia |
| Load balancer | nginx 1.27 alpine | Padrão de mercado; configuração mínima; consumo baixo de RAM |
| Container | Docker multi-stage | Pré-processamento no build, runtime enxuto |
| Estratégia de busca | Brute force KNN (k=5) **como ponto de partida** | Maior valor didático; evolução para ANN fica na seção 9 |
| Distância | Euclidiana ao quadrado | Mesma ordenação que euclidiana, sem `sqrt` no hot path |
| Acesso ao dataset | `mmap` via `mmap-io` | Compartilhamento de páginas físicas entre containers |

---

## 3. Arquitetura

```
                    ┌─────────────────┐
   cliente ──9999──▶│  nginx          │ (round-robin, sem lógica de negócio)
                    └────────┬────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
            ┌──────────┐          ┌──────────┐
            │  api-1   │          │  api-2   │
            │ (Node)   │          │ (Node)   │
            └─────┬────┘          └─────┬────┘
                  │ mmap                │ mmap
                  └──────────┬──────────┘
                             ▼
                   ┌─────────────────────┐
                   │ /app/data/refs.bin  │ (171 MB, dentro da imagem)
                   │  3M × (56B + 1B)    │
                   └─────────────────────┘
```

### Serviços no `docker-compose.yml`

| Serviço | Imagem | Limit CPU | Limit RAM | Papel |
|---|---|---|---|---|
| `nginx` | `nginx:1.27-alpine` | 0.15 | 20 MB | Load balancer round-robin |
| `api-1` | `rinha2026-api:vX` (nossa) | 0.425 | 160 MB | API + KNN + mmap |
| `api-2` | `rinha2026-api:vX` (nossa) | 0.425 | 160 MB | API + KNN + mmap |
| **Total** | — | **1.00** | **340 MB** | dentro do limite de 1 CPU / 350 MB |

### Orçamento de RAM por API (estimativa)

| Item | RAM |
|---|---|
| Node runtime + Fastify + heap inicial | ~50 MB |
| Estruturas auxiliares (mccRisk, normalization, top-5) | ~5 MB |
| Mmap "visível" do refs.bin (RSS shared) | ~100 MB |
| Margem de segurança | ~5 MB |
| **Total por instância** | **~160 MB** |

A RAM física **real** **deve** ser menor porque o `mmap` com `MAP_SHARED` faz com que as páginas do `refs.bin` sejam compartilhadas entre as duas instâncias pelo kernel Linux. Como exatamente o cgroup v2 contabiliza essas páginas compartilhadas é uma incógnita prática — vamos validar com `docker stats` durante os primeiros testes. Se o RSS reportado por cada container ficar próximo do limite de 160 MB e estourar, recorremos ao plano B (quantização int8, ver seção 10).

---

## 4. Componentes (módulos)

### `src/dataset.ts` — carregamento via mmap

Responsabilidade: abrir `/app/data/refs.bin`, mapear em memória, expor:

- `vectors: Float32Array` — view sobre os primeiros 168 MB (42 milhões de floats).
- `labels: Uint8Array` — view sobre os últimos 3 MB (3 milhões de bytes, 0 = legit, 1 = fraud).
- `totalRecords: number` — constante, 3 000 000.

Carregamento acontece **uma vez** no startup. Função `loadDataset()` retorna apenas quando tudo estiver pronto.

### `src/vectorize.ts` — payload → vetor de 14 floats

Função pura:

```ts
function vectorize(
  payload: FraudScoreRequest,
  mccRisk: Record<string, number>,
  norm: Normalization
): Float32Array
```

Implementa exatamente as 14 dimensões descritas em `docs/br/REGRAS_DE_DETECCAO.md`. Cuidados:

- `getUTCDay()` em JS retorna dom=0; converter para seg=0 com `(getUTCDay() + 6) % 7`.
- `last_transaction === null` → `v[5] = -1` e `v[6] = -1` (sentinela).
- MCC ausente em `mccRisk` → `0.5` padrão.
- `clamp(x)` mantém em `[0, 1]`; o `-1` sentinela **não** passa pelo clamp.

### `src/knn.ts` — brute force KNN

Função:

```ts
function knnSearch(query: Float32Array): number
```

Retorna o número de fraudes nos 5 vetores mais próximos (0 a 5).

Estratégia detalhada:

1. Iterar de `i = 0` até `totalRecords - 1`.
2. Para cada `i`, calcular `dist²` somando 14 termos `(query[d] - vectors[i*14 + d])²`.
3. Se `dist² < topDists[4]`, inserir no array `topDists`/`topIdx` mantendo ordenado.
4. **Early termination:** durante a soma dos 14 termos, se a soma parcial já passou de `topDists[4]`, abandonar o registro.
5. Ao final, contar quantos `labels[topIdx[k]] === 1` para `k = 0..4`.

Otimizações já incluídas:

- Squared euclidean (sem `Math.sqrt`).
- Float32Array views diretamente sobre o buffer mmap (sem `readFloatLE`).
- Loop unrolled das 14 dimensões.
- Array de top-5 (em vez de min-heap, mais simples e rápido para k=5).
- Early termination.

### `src/server.ts` — HTTP layer (Fastify)

Rotas:

- `GET /ready` — 200 se `isReady === true`, 503 caso contrário.
- `POST /fraud-score` — chama `vectorize` → `knnSearch` → retorna `{ approved, fraud_score }`.

Configuração:

- Logger desligado.
- Validação de schema desligada na primeira versão (Fastify ainda valida JSON parse).
- `host: '0.0.0.0'`, `port: 3000`.
- `isReady` vira `true` somente após `loadDataset()` ter retornado com sucesso.

### `scripts/preprocess.ts` — gerador do `refs.bin`

Roda **dentro do `docker build`**, não no runtime. Recebe `resources/references.json.gz`, produz `/build/data/refs.bin`.

Layout do arquivo:

```
Bytes 0..(168_000_000 - 1):     vetores (3M × 14 × float32, little-endian)
Bytes 168_000_000..(171_000_000 - 1): labels (3M × uint8)
```

Implementação: stream de descompactação (`zlib.createGunzip`) + parser JSON streaming (ex.: `stream-json`) para evitar carregar 284 MB em RAM mesmo no build.

### `nginx.conf` — load balancer

Configuração mínima:

- `upstream apis` com `server api-1:3000` e `server api-2:3000`.
- Algoritmo: round-robin (default do nginx, **sem** `least_conn` ou outros).
- `keepalive 256` no upstream para reaproveitar conexões TCP.
- `access_log off`, `error_log /dev/null crit`.
- `proxy_http_version 1.1`, `proxy_set_header Connection ""`.

Nenhuma lógica de negócio: sem `if`, sem `map`, sem rewrite condicional do body.

### `Dockerfile` — build multi-stage

**Stage 1 (`builder`):**
- Base: `node:22-alpine`
- Instala deps (`npm ci`)
- Copia src, scripts, resources
- Roda `tsc` para transpilar
- Roda `preprocess.ts` para gerar `data/refs.bin`

**Stage 2 (`runtime`):**
- Base: `node:22-alpine`
- Copia apenas `dist/` e `data/refs.bin` do builder
- Instala apenas deps de runtime (`npm ci --omit=dev`)
- `CMD ["node", "dist/server.js"]`

Build deve ser feito sempre com `--platform linux/amd64` (autor usa Mac M4, teste roda em amd64).

---

## 5. Fluxo de dados

### Startup (cada instância da API)

```
1. Container inicia
2. server.ts → loadDataset()
3. mmap-io abre /app/data/refs.bin
4. Cria Float32Array sobre offset 0
5. Cria Uint8Array sobre offset 168_000_000
6. Carrega mcc_risk.json e normalization.json em memória (~3 KB)
7. isReady = true
8. Fastify começa a aceitar requisições
```

Tempo total esperado: 1–3 segundos.

### Por requisição (`POST /fraud-score`)

```
1. nginx recebe na porta 9999
2. nginx encaminha para api-1 ou api-2 (round-robin)
3. Fastify parse do JSON
4. vectorize(payload, mccRisk, norm) → Float32Array(14)
5. knnSearch(vector) → número de fraudes (0..5)
6. fraud_score = fraudes / 5
7. approved = fraud_score < 0.6
8. Resposta JSON
```

---

## 6. Tratamento de erros

A pontuação penaliza erros HTTP com peso 5 (versus FP=1, FN=3). Pra evitar HTTP 500 a todo custo, qualquer erro inesperado no `vectorize` ou `knnSearch` retorna `{ approved: true, fraud_score: 0.0 }` com status 200 (um FN/TN no pior caso, melhor que um Err).

```ts
app.setErrorHandler((err, req, reply) => {
  // Log opcional aqui apenas em modo debug
  reply.code(200).send({ approved: true, fraud_score: 0.0 })
})
```

Trade-off explícito: aceitamos virar um FN ocasional em vez de devolver 500. Para a Rinha, isso é vantajoso.

---

## 7. Testes e validação

### Testes locais (desenvolvimento)

- **Unit tests** (vitest ou node:test) para `vectorize`:
  - Caso com `last_transaction = null` → posições 5 e 6 = `-1`.
  - Clamp em `amount > max_amount`.
  - MCC ausente → `0.5`.
  - `day_of_week` para uma segunda e um domingo.
- **Unit tests** para `knnSearch` com dataset de fixture pequeno (10 vetores) com top-5 esperado calculado manualmente.

### Smoke test

Usar `test/smoke.js` do repo oficial para validação rápida do contrato HTTP.

### Load test

Usar `test/test.js` do repo oficial:
```bash
cd test && docker compose --profile test up
```

Validar `results.json`:
- `failure_rate` deve ficar bem abaixo de 15%.
- `final_score` positivo é o critério mínimo de sucesso para esta versão.

---

## 8. Estrutura final do projeto (branch `main`)

```
rinha-de-backend-2026/
├── docs/
│   └── superpowers/specs/2026-05-25-fraud-detection-design.md
├── src/
│   ├── server.ts
│   ├── vectorize.ts
│   ├── knn.ts
│   ├── dataset.ts
│   └── types.ts
├── scripts/
│   └── preprocess.ts
├── tests/
│   ├── vectorize.test.ts
│   └── knn.test.ts
├── data/                       (gerado no build; .gitignore)
│   └── refs.bin
├── nginx.conf
├── Dockerfile
├── docker-compose.yml
├── package.json
├── tsconfig.json
└── README.md
```

### Branch `submission`

Apenas:

```
submission/
├── docker-compose.yml          (referenciando imagem JÁ no Docker Hub/GHCR)
├── nginx.conf
├── info.json
└── participants/MarcioMonte.json    (no fork do repo da Rinha, não aqui)
```

---

## 9. Caminhos de evolução (para depois do MVP)

Em ordem sugerida, uma mudança por vez, sempre medindo `final_score` antes e depois:

1. **Microoptimization do hot path** — desenrolamento mais agressivo, sem branches no early termination.
2. **Worker threads** — particionar a busca em N chunks; com 0.425 CPU pode não compensar, mas vale medir.
3. **Pré-filtro grosso por bucket** — ex.: separar vetores por `hour_of_day` e só buscar nos buckets próximos.
4. **Migrar para `hnswlib-node`** — abandonar brute force; salto grande de performance.
5. **Trocar Node por Go/Rust** — se quiser brigar pelo topo do ranking.

Cada um desses passos pode (e deve) virar um novo design + plano de implementação separados.

---

## 10. Riscos conhecidos

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| `mmap-io` instável em alpine | Baixa | Alto (impede a estratégia de RAM) | Plano B: quantização int8 (45 MB total, cabe em RAM normal) |
| p99 estoura 2 s sob carga | Média | Alto (corte de -3000 pontos) | Early termination + medir antes de submeter; ter HNSW como evolução |
| Diferença de cálculo de data em UTC | Baixa | Médio (FP/FN sistemáticos) | Testes unitários cobrindo UTC, segunda/domingo |
| Build amd64 lento no M4 (emulação) | Alta | Baixo (lentidão de DX) | Aceitar; usar cache do Docker; rodar build em CI quando possível |
| RAM compartilhada via mmap não baixar RSS reportado | Média | Médio (cgroups matam container) | Medir com `docker stats`; quantização int8 como plano B |

---

## 11. Definição de "pronto" para esta versão

- [ ] Build com `docker buildx build --platform linux/amd64` finaliza sem erro.
- [ ] `docker compose up` traz os 3 serviços; `GET /ready` em 9999 retorna 200 em até 30 s.
- [ ] Smoke test do repo passa.
- [ ] Load test do repo termina sem `failure_rate > 15%`.
- [ ] `final_score` positivo em `test/results.json`.
- [ ] Imagem publicada num registry público.
- [ ] Branch `submission` criada com `docker-compose.yml` apontando para a imagem publicada.
- [ ] `participants/MarcioMonte.json` adicionado no fork do repo da Rinha via PR.
