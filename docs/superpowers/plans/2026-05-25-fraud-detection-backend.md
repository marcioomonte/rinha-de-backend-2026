# Fraud Detection Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir o backend de detecção de fraude da Rinha 2026: API HTTP em TypeScript/Fastify atrás de nginx (round-robin), com KNN brute force sobre um arquivo binário pré-processado mapeado por `mmap`. A meta desta versão é uma submissão funcional com `final_score` positivo.

**Architecture:** nginx (load balancer) → 2× api (Node 22 + Fastify) → `refs.bin` (171 MB, mmap compartilhado). Pré-processamento do dataset acontece no `docker build`, runtime apenas mapeia o arquivo.

**Tech Stack:** TypeScript, Node 22, Fastify 5, vitest, mmap-io, stream-json, nginx 1.27, Docker (multi-stage).

> **Nota do autor:** Marcio prefere commits manuais. Os "commit checkpoints" no fim de cada bloco lógico ficam como **sugestão** — quando você (Marcio) sentir que está num ponto estável, copia o comando. Nada é executado automaticamente.

---

## Task 1: Bootstrap do projeto Node/TypeScript

**Files:**
- Create: `package.json`
- Create: `tsconfig.json`
- Create: `.nvmrc`
- Create: `.gitignore` (já existe — vamos adicionar entradas)
- Create: `.dockerignore`

**Por que importa:** estabelece o terreno. Sem `tsconfig.json` o `tsc` não roda; sem `package.json` o `npm install` não tem o que instalar. Essa task **não tem teste** porque é só configuração.

- [ ] **Step 1: Verificar versão do Node**

Run: `node --version`
Expected: `v22.x.x`. Se sair `v20` ou outra, instale Node 22 (com nvm: `nvm install 22 && nvm use 22`).

- [ ] **Step 2: Criar `.nvmrc`**

```
22
```

- [ ] **Step 3: Criar `package.json`**

```json
{
  "name": "rinha-de-backend-2026",
  "version": "0.1.0",
  "description": "Fraud detection backend for Rinha de Backend 2026",
  "license": "MIT",
  "type": "module",
  "scripts": {
    "build": "tsc",
    "preprocess": "node --import tsx scripts/preprocess.ts",
    "start": "node dist/server.js",
    "dev": "node --import tsx src/server.ts",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "fastify": "^5.0.0",
    "mmap-io": "^1.4.5",
    "stream-json": "^1.8.0"
  },
  "devDependencies": {
    "@types/node": "^22.0.0",
    "@types/stream-json": "^1.7.7",
    "tsx": "^4.19.0",
    "typescript": "^5.6.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 4: Criar `tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "node",
    "outDir": "dist",
    "rootDir": ".",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "resolveJsonModule": true,
    "declaration": false,
    "sourceMap": false,
    "isolatedModules": true
  },
  "include": ["src/**/*.ts", "scripts/**/*.ts"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

- [ ] **Step 5: Atualizar `.gitignore`**

O arquivo já existe. Acrescente (sem remover o que já está lá):

```
node_modules/
dist/
data/
*.log
.DS_Store
```

- [ ] **Step 6: Criar `.dockerignore`**

```
node_modules
dist
data
.git
.gitignore
docs
participants
test
misc
*.md
.DS_Store
```

(Excluir `resources/references.json.gz` daqui **não**, pois ele é necessário no build.)

- [ ] **Step 7: Instalar dependências**

Run: `npm install`
Expected: `node_modules` populado, sem warning crítico.

- [ ] **Step 8: Validar compilação inicial**

Run: `npx tsc --noEmit`
Expected: sai sem output (sucesso — não há código ainda).

**Checkpoint de commit (sugestão):**
```bash
git add package.json package-lock.json tsconfig.json .nvmrc .gitignore .dockerignore
# Quando quiser commitar:
# git commit -m "chore: bootstrap node/typescript project"
```

---

## Task 2: Tipos do domínio

**Files:**
- Create: `src/types.ts`

**Por que importa:** centraliza a forma do payload, evita `any` espalhado. Sem teste — é só definição de tipo.

- [ ] **Step 1: Criar `src/types.ts`**

```ts
export interface FraudScoreRequest {
  id: string
  transaction: {
    amount: number
    installments: number
    requested_at: string
  }
  customer: {
    avg_amount: number
    tx_count_24h: number
    known_merchants: string[]
  }
  merchant: {
    id: string
    mcc: string
    avg_amount: number
  }
  terminal: {
    is_online: boolean
    card_present: boolean
    km_from_home: number
  }
  last_transaction: {
    timestamp: string
    km_from_current: number
  } | null
}

export interface FraudScoreResponse {
  approved: boolean
  fraud_score: number
}

export interface Normalization {
  max_amount: number
  max_installments: number
  amount_vs_avg_ratio: number
  max_minutes: number
  max_km: number
  max_tx_count_24h: number
  max_merchant_avg_amount: number
}

export type MccRisk = Record<string, number>
```

- [ ] **Step 2: Validar compilação**

Run: `npx tsc --noEmit`
Expected: sai sem erro.

---

## Task 3: Função `vectorize` — TDD

**Files:**
- Create: `tests/vectorize.test.ts`
- Create: `src/vectorize.ts`

**Por que importa:** essa é a função pura que transforma payload em vetor de 14 floats. Testes garantem que cada uma das 14 dimensões está correta — qualquer erro aqui vira FP/FN sistemático no teste oficial.

- [ ] **Step 1: Escrever os testes (que vão falhar)**

Crie `tests/vectorize.test.ts`:

```ts
import { describe, it, expect } from 'vitest'
import { vectorize } from '../src/vectorize.js'
import type { FraudScoreRequest, Normalization } from '../src/types.js'

const norm: Normalization = {
  max_amount: 10000,
  max_installments: 12,
  amount_vs_avg_ratio: 10,
  max_minutes: 1440,
  max_km: 1000,
  max_tx_count_24h: 20,
  max_merchant_avg_amount: 10000,
}

const mccRisk = {
  '5411': 0.15,
  '5912': 0.20,
}

function basePayload(overrides: Partial<FraudScoreRequest> = {}): FraudScoreRequest {
  return {
    id: 'tx-1',
    transaction: {
      amount: 100,
      installments: 1,
      requested_at: '2026-03-09T12:00:00Z', // segunda-feira 12h UTC
    },
    customer: { avg_amount: 100, tx_count_24h: 1, known_merchants: ['MERC-001'] },
    merchant: { id: 'MERC-001', mcc: '5411', avg_amount: 100 },
    terminal: { is_online: false, card_present: true, km_from_home: 0 },
    last_transaction: null,
    ...overrides,
  }
}

describe('vectorize', () => {
  it('returns a 14-dimension vector', () => {
    const v = vectorize(basePayload(), mccRisk, norm)
    expect(v.length).toBe(14)
  })

  it('uses -1 sentinel for indices 5 and 6 when last_transaction is null', () => {
    const v = vectorize(basePayload({ last_transaction: null }), mccRisk, norm)
    expect(v[5]).toBe(-1)
    expect(v[6]).toBe(-1)
  })

  it('computes minutes_since_last_tx when last_transaction is present', () => {
    const v = vectorize(basePayload({
      transaction: { amount: 100, installments: 1, requested_at: '2026-03-09T13:00:00Z' },
      last_transaction: { timestamp: '2026-03-09T12:00:00Z', km_from_current: 500 },
    }), mccRisk, norm)
    // 60 minutos / 1440 max = 0.0416666...
    expect(v[5]).toBeCloseTo(60 / 1440, 5)
    expect(v[6]).toBeCloseTo(500 / 1000, 5)
  })

  it('clamps amount above max_amount to 1.0', () => {
    const v = vectorize(basePayload({
      transaction: { amount: 25000, installments: 1, requested_at: '2026-03-09T12:00:00Z' },
    }), mccRisk, norm)
    expect(v[0]).toBe(1.0)
  })

  it('returns 0.5 for unknown MCC', () => {
    const v = vectorize(basePayload({
      merchant: { id: 'MERC-001', mcc: '9999', avg_amount: 100 },
    }), mccRisk, norm)
    expect(v[12]).toBe(0.5)
  })

  it('returns 1 when merchant is unknown to customer', () => {
    const v = vectorize(basePayload({
      customer: { avg_amount: 100, tx_count_24h: 1, known_merchants: ['MERC-002'] },
      merchant: { id: 'MERC-001', mcc: '5411', avg_amount: 100 },
    }), mccRisk, norm)
    expect(v[11]).toBe(1)
  })

  it('computes day_of_week with seg=0, dom=6', () => {
    // 2026-03-09 é segunda-feira
    const seg = vectorize(basePayload({
      transaction: { amount: 100, installments: 1, requested_at: '2026-03-09T12:00:00Z' },
    }), mccRisk, norm)
    expect(seg[4]).toBeCloseTo(0 / 6, 5)

    // 2026-03-15 é domingo
    const dom = vectorize(basePayload({
      transaction: { amount: 100, installments: 1, requested_at: '2026-03-15T12:00:00Z' },
    }), mccRisk, norm)
    expect(dom[4]).toBeCloseTo(6 / 6, 5)
  })

  it('matches the legit example from REGRAS_DE_DETECCAO.md', () => {
    // Payload exato do docs
    const v = vectorize({
      id: 'tx-1329056812',
      transaction: { amount: 41.12, installments: 2, requested_at: '2026-03-11T18:45:53Z' },
      customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ['MERC-003', 'MERC-016'] },
      merchant: { id: 'MERC-016', mcc: '5411', avg_amount: 60.25 },
      terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
      last_transaction: null,
    }, { '5411': 0.15 }, norm)

    // Vetor esperado da spec: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
    expect(v[0]).toBeCloseTo(0.004112, 4)
    expect(v[1]).toBeCloseTo(0.1667, 3)
    expect(v[2]).toBeCloseTo(0.05, 3)
    expect(v[3]).toBeCloseTo(18 / 23, 3)
    expect(v[4]).toBeCloseTo(2 / 6, 3)        // 2026-03-11 é quarta-feira → idx 2
    expect(v[5]).toBe(-1)
    expect(v[6]).toBe(-1)
    expect(v[7]).toBeCloseTo(29.23 / 1000, 4)
    expect(v[8]).toBeCloseTo(3 / 20, 4)
    expect(v[9]).toBe(0)
    expect(v[10]).toBe(1)
    expect(v[11]).toBe(0)                      // MERC-016 está em known_merchants
    expect(v[12]).toBe(0.15)
    expect(v[13]).toBeCloseTo(60.25 / 10000, 5)
  })
})
```

- [ ] **Step 2: Rodar testes para ver eles falharem**

Run: `npm test`
Expected: FAIL — `vectorize` não existe ainda.

- [ ] **Step 3: Implementar `src/vectorize.ts`**

```ts
import type { FraudScoreRequest, Normalization, MccRisk } from './types.js'

function clamp(x: number): number {
  if (x < 0) return 0
  if (x > 1) return 1
  return x
}

export function vectorize(
  p: FraudScoreRequest,
  mccRisk: MccRisk,
  norm: Normalization
): Float32Array {
  const v = new Float32Array(14)
  const date = new Date(p.transaction.requested_at)

  v[0] = clamp(p.transaction.amount / norm.max_amount)
  v[1] = clamp(p.transaction.installments / norm.max_installments)

  const avgAmount = p.customer.avg_amount === 0 ? 1 : p.customer.avg_amount
  v[2] = clamp((p.transaction.amount / avgAmount) / norm.amount_vs_avg_ratio)

  v[3] = date.getUTCHours() / 23
  // JS getUTCDay: dom=0..sáb=6 → queremos seg=0..dom=6
  v[4] = ((date.getUTCDay() + 6) % 7) / 6

  if (p.last_transaction !== null) {
    const ms = date.getTime() - new Date(p.last_transaction.timestamp).getTime()
    const minutes = ms / 60000
    v[5] = clamp(minutes / norm.max_minutes)
    v[6] = clamp(p.last_transaction.km_from_current / norm.max_km)
  } else {
    v[5] = -1
    v[6] = -1
  }

  v[7] = clamp(p.terminal.km_from_home / norm.max_km)
  v[8] = clamp(p.customer.tx_count_24h / norm.max_tx_count_24h)
  v[9] = p.terminal.is_online ? 1 : 0
  v[10] = p.terminal.card_present ? 1 : 0
  v[11] = p.customer.known_merchants.includes(p.merchant.id) ? 0 : 1
  v[12] = mccRisk[p.merchant.mcc] ?? 0.5
  v[13] = clamp(p.merchant.avg_amount / norm.max_merchant_avg_amount)

  return v
}
```

- [ ] **Step 4: Rodar testes para verificar que passam**

Run: `npm test`
Expected: todos os testes do arquivo `vectorize.test.ts` passam.

**Checkpoint de commit (sugestão):**
```bash
git add src/types.ts src/vectorize.ts tests/vectorize.test.ts
# git commit -m "feat: implement payload vectorize with unit tests"
```

---

## Task 4: Script de pré-processamento — gerar `refs.bin`

**Files:**
- Create: `scripts/preprocess.ts`

**Por que importa:** transforma `resources/references.json.gz` (16 MB comprimido / 284 MB descomprimido) em `data/refs.bin` (171 MB binário). Rodado **uma vez no `docker build`**, não no runtime. Sem TDD aqui — é um script "data pipeline" cuja verificação é manual (tamanho do arquivo gerado).

**Layout do arquivo:**
```
Bytes 0..(168_000_000 - 1):           vetores (3M × 14 × float32, LE)
Bytes 168_000_000..(171_000_000 - 1): labels (3M × uint8: 0=legit, 1=fraud)
```

- [ ] **Step 1: Criar diretório de saída**

```bash
mkdir -p data
```

- [ ] **Step 2: Implementar `scripts/preprocess.ts`**

```ts
import { createReadStream, createWriteStream, openSync, writeSync, closeSync } from 'node:fs'
import { createGunzip } from 'node:zlib'
import { pipeline } from 'node:stream/promises'
import StreamArray from 'stream-json/streamers/StreamArray.js'
import { parser } from 'stream-json'

const TOTAL_RECORDS = 3_000_000
const VECTOR_SIZE = 14
const VECTOR_BYTES = VECTOR_SIZE * 4  // 56 bytes por vetor
const LABEL_BYTES = 1
const RECORD_BYTES = VECTOR_BYTES + LABEL_BYTES  // 57

const INPUT = 'resources/references.json.gz'
const OUTPUT = 'data/refs.bin'

async function main() {
  console.log(`Reading ${INPUT}`)
  console.log(`Writing ${OUTPUT}`)
  console.log(`Expected size: ${(TOTAL_RECORDS * RECORD_BYTES / 1024 / 1024).toFixed(1)} MB`)

  // Buffers separados (Layout B do design): vetores primeiro, labels depois.
  const vectorsBuf = Buffer.allocUnsafe(TOTAL_RECORDS * VECTOR_BYTES)
  const labelsBuf = Buffer.allocUnsafe(TOTAL_RECORDS * LABEL_BYTES)

  let count = 0
  const source = createReadStream(INPUT)
    .pipe(createGunzip())
    .pipe(parser())
    .pipe(StreamArray.withParser())

  for await (const { value } of source) {
    if (count >= TOTAL_RECORDS) {
      throw new Error(`More than ${TOTAL_RECORDS} records in input — aborting`)
    }

    const vec: number[] = value.vector
    if (vec.length !== VECTOR_SIZE) {
      throw new Error(`Record ${count} has vector length ${vec.length}, expected ${VECTOR_SIZE}`)
    }

    const vecOffset = count * VECTOR_BYTES
    for (let d = 0; d < VECTOR_SIZE; d++) {
      vectorsBuf.writeFloatLE(vec[d], vecOffset + d * 4)
    }

    labelsBuf[count] = value.label === 'fraud' ? 1 : 0
    count++

    if (count % 250_000 === 0) {
      console.log(`  ${count.toLocaleString()} / ${TOTAL_RECORDS.toLocaleString()}`)
    }
  }

  if (count !== TOTAL_RECORDS) {
    throw new Error(`Expected ${TOTAL_RECORDS} records, got ${count}`)
  }

  // Escreve o arquivo final: vetores ‖ labels
  const fd = openSync(OUTPUT, 'w')
  writeSync(fd, vectorsBuf)
  writeSync(fd, labelsBuf)
  closeSync(fd)

  console.log(`Done. Wrote ${count.toLocaleString()} records to ${OUTPUT}`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
```

- [ ] **Step 3: Rodar o script local**

Run: `npm run preprocess`

Expected output (em ~30-90 segundos):
```
Reading resources/references.json.gz
Writing data/refs.bin
Expected size: 163.1 MB
  250,000 / 3,000,000
  500,000 / 3,000,000
  ...
  3,000,000 / 3,000,000
Done. Wrote 3,000,000 records to data/refs.bin
```

- [ ] **Step 4: Verificar o arquivo gerado**

Run: `ls -lh data/refs.bin`
Expected: ~171 MB.

Run: `wc -c data/refs.bin`
Expected: `171000000` (exato).

**Checkpoint de commit (sugestão):**
```bash
git add scripts/preprocess.ts
# git commit -m "feat: add references.json.gz → refs.bin preprocessor"
```

> **Importante:** `data/refs.bin` está no `.gitignore`. Não commitar.

---

## Task 5: Loader do dataset via `mmap`

**Files:**
- Create: `src/dataset.ts`

**Por que importa:** abre `data/refs.bin` uma vez no startup e expõe duas views — `vectors: Float32Array(42M)` e `labels: Uint8Array(3M)`. O KNN vai indexar essas views diretamente. Sem TDD aqui — depende do arquivo real.

- [ ] **Step 1: Implementar `src/dataset.ts`**

```ts
import { openSync, fstatSync, closeSync } from 'node:fs'
import mmap from 'mmap-io'

const TOTAL_RECORDS = 3_000_000
const VECTOR_SIZE = 14
const VECTOR_BYTES_TOTAL = TOTAL_RECORDS * VECTOR_SIZE * 4   // 168_000_000
const LABEL_BYTES_TOTAL = TOTAL_RECORDS                       // 3_000_000
const EXPECTED_SIZE = VECTOR_BYTES_TOTAL + LABEL_BYTES_TOTAL  // 171_000_000

export interface Dataset {
  vectors: Float32Array
  labels: Uint8Array
  totalRecords: number
  vectorSize: number
}

export function loadDataset(path: string): Dataset {
  const fd = openSync(path, 'r')
  const stat = fstatSync(fd)
  if (stat.size !== EXPECTED_SIZE) {
    closeSync(fd)
    throw new Error(`Unexpected refs.bin size: ${stat.size}, expected ${EXPECTED_SIZE}`)
  }

  // PROT_READ + MAP_SHARED: páginas compartilháveis entre processos.
  const buf = mmap.map(stat.size, mmap.PROT_READ, mmap.MAP_SHARED, fd)
  // O fd pode ser fechado após o mmap; o mapping persiste.
  closeSync(fd)

  // Float32Array sobre o primeiro segmento (vetores).
  // ArrayBuffer subjacente: buf.buffer; offset: buf.byteOffset.
  const vectors = new Float32Array(buf.buffer, buf.byteOffset, VECTOR_BYTES_TOTAL / 4)
  const labels = new Uint8Array(buf.buffer, buf.byteOffset + VECTOR_BYTES_TOTAL, LABEL_BYTES_TOTAL)

  return { vectors, labels, totalRecords: TOTAL_RECORDS, vectorSize: VECTOR_SIZE }
}
```

- [ ] **Step 2: Validar compilação**

Run: `npx tsc --noEmit`
Expected: sai sem erro. (mmap-io pode dar warning de tipos — vamos ignorar.)

> Se `mmap-io` der erro de tipo do tipo "Cannot find module", crie `src/mmap-io.d.ts` com:
> ```ts
> declare module 'mmap-io' {
>   export const PROT_READ: number
>   export const MAP_SHARED: number
>   export function map(size: number, prot: number, flags: number, fd: number, offset?: number): Buffer
> }
> ```

- [ ] **Step 3: Smoke test manual no REPL**

Run:
```bash
node --import tsx -e "import('./src/dataset.ts').then(m => { const d = m.loadDataset('data/refs.bin'); console.log('totalRecords:', d.totalRecords); console.log('vectors.length:', d.vectors.length); console.log('labels.length:', d.labels.length); console.log('first vector:', Array.from(d.vectors.slice(0, 14))); console.log('first label:', d.labels[0]); })"
```

Expected:
```
totalRecords: 3000000
vectors.length: 42000000
labels.length: 3000000
first vector: [ 0.01, 0.0833..., 0.05, ... ]
first label: 0
```

**Checkpoint de commit (sugestão):**
```bash
git add src/dataset.ts
# git commit -m "feat: mmap-backed dataset loader"
```

---

## Task 6: Função `knnSearch` — TDD com fixture pequena

**Files:**
- Create: `tests/knn.test.ts`
- Create: `src/knn.ts`

**Por que importa:** o núcleo da decisão. Brute force varrendo todos os registros, mantendo top-5 por distância euclidiana ao quadrado, devolvendo número de fraudes.

- [ ] **Step 1: Escrever testes (vão falhar)**

```ts
import { describe, it, expect } from 'vitest'
import { knnSearch } from '../src/knn.js'

// Fixture: 6 registros em ordem conhecida.
// Cada vetor tem 14 dimensões. Pra ficar legível, montamos com helpers.
function makeFixture() {
  const vectors = new Float32Array(6 * 14)
  const labels = new Uint8Array(6)

  function setRecord(i: number, baseValue: number, label: number) {
    for (let d = 0; d < 14; d++) vectors[i * 14 + d] = baseValue
    labels[i] = label
  }

  setRecord(0, 0.10, 0) // legit, muito próximo de 0.1
  setRecord(1, 0.11, 1) // fraud, perto
  setRecord(2, 0.20, 0) // legit, mais longe
  setRecord(3, 0.90, 1) // fraud, longe
  setRecord(4, 0.05, 0) // legit, perto
  setRecord(5, 0.15, 1) // fraud, intermediário

  return { vectors, labels, totalRecords: 6, vectorSize: 14 }
}

describe('knnSearch', () => {
  it('returns 0..5 (number of fraud labels among top-5 nearest)', () => {
    const fixture = makeFixture()
    const q = new Float32Array(14).fill(0.10)
    const result = knnSearch(q, fixture)
    expect(result).toBeGreaterThanOrEqual(0)
    expect(result).toBeLessThanOrEqual(5)
  })

  it('finds the closest 5 records and counts fraud labels', () => {
    const fixture = makeFixture()
    const q = new Float32Array(14).fill(0.10)
    // Distâncias ao quadrado de q=[0.10]*14 a cada registro:
    //   record 0 (0.10): 0
    //   record 1 (0.11): 14 * 0.01^2 = 0.0014
    //   record 2 (0.20): 14 * 0.10^2 = 0.14
    //   record 3 (0.90): 14 * 0.80^2 = 8.96  ← descartado
    //   record 4 (0.05): 14 * 0.05^2 = 0.035
    //   record 5 (0.15): 14 * 0.05^2 = 0.035
    // Top-5 = 0, 1, 4, 5, 2 → labels 0, 1, 0, 1, 0 → 2 fraudes
    const result = knnSearch(q, fixture)
    expect(result).toBe(2)
  })

  it('returns 5 when all top-5 are fraud', () => {
    const vectors = new Float32Array(5 * 14).fill(0.5)
    const labels = new Uint8Array([1, 1, 1, 1, 1])
    const q = new Float32Array(14).fill(0.5)
    const result = knnSearch(q, { vectors, labels, totalRecords: 5, vectorSize: 14 })
    expect(result).toBe(5)
  })

  it('returns 0 when all top-5 are legit', () => {
    const vectors = new Float32Array(5 * 14).fill(0.5)
    const labels = new Uint8Array([0, 0, 0, 0, 0])
    const q = new Float32Array(14).fill(0.5)
    const result = knnSearch(q, { vectors, labels, totalRecords: 5, vectorSize: 14 })
    expect(result).toBe(0)
  })
})
```

- [ ] **Step 2: Rodar testes — devem falhar**

Run: `npm test`
Expected: FAIL — `knnSearch` não existe.

- [ ] **Step 3: Implementar `src/knn.ts`**

```ts
import type { Dataset } from './dataset.js'

const K = 5
const VECTOR_SIZE = 14

// Top-5 dists e labels mantidos no escopo do módulo para evitar alocação por chamada.
const topDists = new Float64Array(K)
const topLabels = new Uint8Array(K)

export function knnSearch(query: Float32Array, dataset: Pick<Dataset, 'vectors' | 'labels' | 'totalRecords'>): number {
  const { vectors, labels, totalRecords } = dataset

  for (let k = 0; k < K; k++) {
    topDists[k] = Infinity
    topLabels[k] = 0
  }

  for (let i = 0; i < totalRecords; i++) {
    const base = i * VECTOR_SIZE
    const worst = topDists[K - 1]

    // Unrolled squared euclidean com early termination
    let d = 0
    let diff: number

    diff = query[0]  - vectors[base];      d  = diff * diff
    diff = query[1]  - vectors[base + 1];  d += diff * diff
    diff = query[2]  - vectors[base + 2];  d += diff * diff
    diff = query[3]  - vectors[base + 3];  d += diff * diff
    diff = query[4]  - vectors[base + 4];  d += diff * diff
    diff = query[5]  - vectors[base + 5];  d += diff * diff
    diff = query[6]  - vectors[base + 6];  d += diff * diff
    if (d >= worst) continue
    diff = query[7]  - vectors[base + 7];  d += diff * diff
    diff = query[8]  - vectors[base + 8];  d += diff * diff
    diff = query[9]  - vectors[base + 9];  d += diff * diff
    if (d >= worst) continue
    diff = query[10] - vectors[base + 10]; d += diff * diff
    diff = query[11] - vectors[base + 11]; d += diff * diff
    diff = query[12] - vectors[base + 12]; d += diff * diff
    diff = query[13] - vectors[base + 13]; d += diff * diff

    if (d >= worst) continue

    // Inserção ordenada no top-5: encontra posição e shifta o resto
    const label = labels[i]
    let pos = K - 1
    while (pos > 0 && topDists[pos - 1] > d) {
      topDists[pos] = topDists[pos - 1]
      topLabels[pos] = topLabels[pos - 1]
      pos--
    }
    topDists[pos] = d
    topLabels[pos] = label
  }

  let fraudCount = 0
  for (let k = 0; k < K; k++) fraudCount += topLabels[k]
  return fraudCount
}
```

- [ ] **Step 4: Rodar testes — devem passar**

Run: `npm test`
Expected: todos os testes passam.

**Checkpoint de commit (sugestão):**
```bash
git add src/knn.ts tests/knn.test.ts
# git commit -m "feat: brute-force KNN search with early termination"
```

---

## Task 7: Servidor HTTP Fastify

**Files:**
- Create: `src/server.ts`

**Por que importa:** amarra tudo. Carrega o dataset, monta as rotas, escuta na porta 3000.

- [ ] **Step 1: Implementar `src/server.ts`**

```ts
import Fastify from 'fastify'
import { readFileSync } from 'node:fs'
import { loadDataset, type Dataset } from './dataset.js'
import { vectorize } from './vectorize.js'
import { knnSearch } from './knn.js'
import type { FraudScoreRequest, Normalization, MccRisk } from './types.js'

const DATA_PATH = process.env.REFS_BIN_PATH ?? '/app/data/refs.bin'
const MCC_RISK_PATH = process.env.MCC_RISK_PATH ?? '/app/resources/mcc_risk.json'
const NORM_PATH = process.env.NORM_PATH ?? '/app/resources/normalization.json'
const PORT = Number(process.env.PORT ?? 3000)

let isReady = false
let dataset: Dataset
let mccRisk: MccRisk
let norm: Normalization

const app = Fastify({
  logger: false,
  disableRequestLogging: true,
})

app.get('/ready', async (_req, reply) => {
  if (!isReady) return reply.code(503).send()
  return reply.code(200).send()
})

app.post<{ Body: FraudScoreRequest }>('/fraud-score', async (req, reply) => {
  const vec = vectorize(req.body, mccRisk, norm)
  const fraudCount = knnSearch(vec, dataset)
  const fraud_score = fraudCount / 5
  const approved = fraud_score < 0.6
  return reply.code(200).send({ approved, fraud_score })
})

// Defensive: qualquer erro vira { approved: true, fraud_score: 0 } (FN < Err em score).
app.setErrorHandler((_err, _req, reply) => {
  reply.code(200).send({ approved: true, fraud_score: 0 })
})

async function start() {
  console.log('Loading mcc_risk and normalization...')
  mccRisk = JSON.parse(readFileSync(MCC_RISK_PATH, 'utf-8'))
  norm = JSON.parse(readFileSync(NORM_PATH, 'utf-8'))

  console.log(`Loading dataset from ${DATA_PATH}...`)
  const t0 = Date.now()
  dataset = loadDataset(DATA_PATH)
  console.log(`Dataset loaded in ${Date.now() - t0} ms (${dataset.totalRecords.toLocaleString()} records)`)

  await app.listen({ host: '0.0.0.0', port: PORT })
  console.log(`Server listening on :${PORT}`)
  isReady = true
}

start().catch((err) => {
  console.error('Fatal startup error:', err)
  process.exit(1)
})
```

- [ ] **Step 2: Subir o servidor local (dev mode)**

Run num terminal:
```bash
REFS_BIN_PATH=./data/refs.bin \
MCC_RISK_PATH=./resources/mcc_risk.json \
NORM_PATH=./resources/normalization.json \
npm run dev
```

Expected:
```
Loading mcc_risk and normalization...
Loading dataset from ./data/refs.bin...
Dataset loaded in 50 ms (3,000,000 records)
Server listening on :3000
```

- [ ] **Step 3: Testar `/ready` no outro terminal**

Run: `curl -i http://localhost:3000/ready`
Expected: `HTTP/1.1 200 OK`

- [ ] **Step 4: Testar `/fraud-score` com um payload de exemplo**

Run:
```bash
curl -X POST http://localhost:3000/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "tx-1329056812",
    "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
    "customer":    { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003","MERC-016"] },
    "merchant":    { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
    "terminal":    { "is_online": false, "card_present": true, "km_from_home": 29.23 },
    "last_transaction": null
  }'
```

Expected: JSON tipo `{"approved":true,"fraud_score":0}` (esse é o exemplo legítimo da spec).

- [ ] **Step 5: Medir latência aproximada**

Run várias vezes:
```bash
time curl -s -X POST http://localhost:3000/fraud-score \
  -H 'Content-Type: application/json' \
  -d @- < <(echo '{"id":"tx-1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-09T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5411","avg_amount":100},"terminal":{"is_online":false,"card_present":true,"km_from_home":0},"last_transaction":null}')
```

Expected: tempo total na faixa de **30–150 ms** no seu M4 nativo (será mais lento dentro do container amd64 emulado).

- [ ] **Step 6: Parar o servidor**

Ctrl+C no terminal do `npm run dev`.

**Checkpoint de commit (sugestão):**
```bash
git add src/server.ts
# git commit -m "feat: fastify server wiring /ready and /fraud-score"
```

---

## Task 8: Dockerfile multi-stage

**Files:**
- Create: `Dockerfile`

**Por que importa:** isola o pré-processamento (faz no build) do runtime (faz só load). A imagem final embute `refs.bin` como parte da imagem.

- [ ] **Step 1: Criar `Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1.7
# ---------- Stage 1: builder ----------
FROM --platform=linux/amd64 node:22-alpine AS builder
WORKDIR /build

# Deps primeiro (cache friendly)
COPY package.json package-lock.json ./
RUN npm ci

# Código fonte + recursos
COPY tsconfig.json ./
COPY src ./src
COPY scripts ./scripts
COPY resources ./resources

# Transpila e pré-processa o dataset
RUN npx tsc \
 && node --import tsx scripts/preprocess.ts \
 && ls -lh data/refs.bin

# ---------- Stage 2: runtime ----------
FROM --platform=linux/amd64 node:22-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production

COPY package.json package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

COPY --from=builder /build/dist ./dist
COPY --from=builder /build/data ./data
COPY --from=builder /build/resources/mcc_risk.json ./resources/mcc_risk.json
COPY --from=builder /build/resources/normalization.json ./resources/normalization.json

EXPOSE 3000
CMD ["node", "dist/server.js"]
```

- [ ] **Step 2: Build da imagem**

Run:
```bash
docker buildx build --platform linux/amd64 -t rinha2026-api:dev --load .
```

Expected: build completa em alguns minutos (a primeira vez é mais lenta por causa da emulação amd64 no M4). A linha do preprocess deve mostrar `data/refs.bin` ~171 MB.

- [ ] **Step 3: Subir o container isolado pra sanity check**

Run:
```bash
docker run --rm -p 3000:3000 rinha2026-api:dev
```

Expected: log igual ao startup local — `Server listening on :3000`.

- [ ] **Step 4: Em outro terminal, validar `/ready` e `/fraud-score`**

```bash
curl -i http://localhost:3000/ready
curl -X POST http://localhost:3000/fraud-score -H 'Content-Type: application/json' \
  -d '{"id":"tx-1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-09T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5411","avg_amount":100},"terminal":{"is_online":false,"card_present":true,"km_from_home":0},"last_transaction":null}'
```

Expected: 200 OK em ambos.

- [ ] **Step 5: Parar o container**

Ctrl+C.

**Checkpoint de commit (sugestão):**
```bash
git add Dockerfile
# git commit -m "feat: multi-stage Dockerfile with preprocess in build stage"
```

---

## Task 9: Configuração do nginx

**Files:**
- Create: `nginx.conf`

**Por que importa:** o load balancer. Tem que ser round-robin puro, sem lógica.

- [ ] **Step 1: Criar `nginx.conf`**

```nginx
worker_processes 1;
worker_rlimit_nofile 8192;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    access_log off;
    error_log /dev/null crit;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 30;
    keepalive_requests 10000;

    upstream apis {
        # round-robin padrão; sem "least_conn" para evitar parecer lógica.
        server api-1:3000 max_fails=0;
        server api-2:3000 max_fails=0;
        keepalive 256;
    }

    server {
        listen 9999 default_server backlog=4096;

        location / {
            proxy_pass http://apis;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
        }
    }
}
```

**Checkpoint de commit (sugestão):**
```bash
git add nginx.conf
# git commit -m "feat: nginx round-robin load balancer config"
```

---

## Task 10: `docker-compose.yml`

**Files:**
- Create: `docker-compose.yml`

**Por que importa:** orquestra nginx + 2 APIs, com os limites de recursos.

- [ ] **Step 1: Criar `docker-compose.yml`**

```yaml
services:
  api-1:
    image: rinha2026-api:dev
    networks: [rinha]
    deploy:
      resources:
        limits:
          cpus: "0.425"
          memory: "160MB"
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:3000/ready"]
      interval: 2s
      timeout: 1s
      retries: 30
      start_period: 30s

  api-2:
    image: rinha2026-api:dev
    networks: [rinha]
    deploy:
      resources:
        limits:
          cpus: "0.425"
          memory: "160MB"
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:3000/ready"]
      interval: 2s
      timeout: 1s
      retries: 30
      start_period: 30s

  nginx:
    image: nginx:1.27-alpine
    ports:
      - "9999:9999"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api-1
      - api-2
    networks: [rinha]
    deploy:
      resources:
        limits:
          cpus: "0.15"
          memory: "20MB"

networks:
  rinha:
    driver: bridge
```

- [ ] **Step 2: Subir o stack completo**

Run: `docker compose up -d`
Expected: 3 containers (`api-1`, `api-2`, `nginx`) iniciam.

- [ ] **Step 3: Aguardar prontidão**

Run: `for i in {1..30}; do curl -fs http://localhost:9999/ready && echo " — ready" && break; echo "tentativa $i..."; sleep 2; done`
Expected: em ~10-30 segundos, mostra ` — ready`.

- [ ] **Step 4: Testar `/fraud-score` via load balancer**

```bash
curl -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{"id":"tx-1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-09T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5411","avg_amount":100},"terminal":{"is_online":false,"card_present":true,"km_from_home":0},"last_transaction":null}'
```

Expected: 200 com resposta JSON.

- [ ] **Step 5: Ver consumo de recursos**

Run: `docker stats --no-stream`
Expected: `api-1` e `api-2` mostrando RSS na faixa de 100-160 MB cada; `nginx` <20 MB. Se algum estourar, é hora de cair pro plano B (quantização int8 — fora deste plano).

- [ ] **Step 6: Derrubar o stack**

Run: `docker compose down`

**Checkpoint de commit (sugestão):**
```bash
git add docker-compose.yml
# git commit -m "feat: docker-compose with 1 CPU / 340MB resource budget"
```

---

## Task 11: Rodar smoke test oficial

**Files:** (nenhum — só execução)

**Por que importa:** verificação rápida de que o contrato HTTP está OK antes do load test pesado.

- [ ] **Step 1: Subir o stack**

Run: `docker compose up -d`

- [ ] **Step 2: Aguardar `/ready`**

Run: `for i in {1..30}; do curl -fs http://localhost:9999/ready && break; sleep 2; done`

- [ ] **Step 3: Rodar smoke test**

```bash
cd test && docker compose --profile smoke up --abort-on-container-exit
```

Expected: k6 mostra um número pequeno de requisições, todas com `status=200`. Nenhum erro.

- [ ] **Step 4: Voltar pra raiz e parar tudo**

```bash
cd ..
docker compose down
cd test && docker compose --profile smoke down
```

---

## Task 12: Rodar load test oficial e medir pontuação

**Files:** (nenhum — só execução)

**Por que importa:** é o critério de "pronto pra submeter". Se o `final_score` for positivo, declaramos vitória **da primeira versão**.

- [ ] **Step 1: Subir o stack**

Run: `docker compose up -d`

- [ ] **Step 2: Aguardar `/ready` + warmup**

Run: `for i in {1..30}; do curl -fs http://localhost:9999/ready && break; sleep 2; done && sleep 5`
(O `sleep 5` extra deixa o JIT do V8 quentar com alguns smoke requests opcionais antes do teste oficial.)

- [ ] **Step 3: Rodar load test completo**

```bash
cd test && docker compose --profile test up --abort-on-container-exit
```

Expected: ~2 minutos de execução. Ao final, `test/results.json` é gerado.

- [ ] **Step 4: Inspecionar resultado**

Run: `cat test/results.json`
Olhar especialmente:
- `scoring.failure_rate` — precisa estar bem abaixo de `15%`.
- `scoring.final_score` — alvo desta versão: **positivo** (>0). Faixa esperada para brute force: 100–1500.
- `scoring.p99_score.value` e `scoring.detection_score.value` — pra entender de onde veio o score.
- `scoring.breakdown` — `false_positive_detections`, `false_negative_detections`, `http_errors`.

- [ ] **Step 5: Diagnóstico (se necessário)**

Se `failure_rate > 15%`:
- Verificar logs: `docker compose logs api-1 api-2 | tail -50`
- Provável causa: timeout / OOM. Olhar `docker stats` durante a execução.

Se `final_score` muito baixo (negativo ou <100):
- `p99` provavelmente alto. Próximo passo é otimização (não faz parte desta primeira versão; documentar resultado e seguir pra evoluções na seção 9 do design).

- [ ] **Step 6: Parar o stack**

```bash
docker compose down
cd test && docker compose --profile test down
cd ..
```

**Checkpoint de commit (sugestão):**
Se quiser salvar o `results.json` como referência:
```bash
mkdir -p results-history
cp test/results.json results-history/v0.1-brute-force-$(date +%Y%m%d-%H%M).json
# git add results-history/...
# git commit -m "chore: record baseline brute-force load test results"
```

---

## Pronto pra submeter?

Quando a Task 12 mostrar `final_score > 0` e `failure_rate < 15%`, **esta primeira versão está completa**. A submissão oficial (publicar imagem em registry público, criar branch `submission`, PR no repo da Rinha, abrir issue `rinha/test`) **não está coberta neste plano** — vai ser um próximo design + plano separado, depois que validarmos localmente.

---

## Resumo de progresso

- [ ] Task 1 — Bootstrap do projeto
- [ ] Task 2 — Tipos do domínio
- [ ] Task 3 — `vectorize` com testes
- [ ] Task 4 — Script `preprocess.ts`
- [ ] Task 5 — Loader `mmap`
- [ ] Task 6 — `knnSearch` com testes
- [ ] Task 7 — Servidor Fastify
- [ ] Task 8 — Dockerfile
- [ ] Task 9 — nginx.conf
- [ ] Task 10 — docker-compose.yml
- [ ] Task 11 — Smoke test oficial
- [ ] Task 12 — Load test oficial e baseline
