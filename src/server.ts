import Fastify from 'fastify'
import { readFileSync } from 'node:fs'
import { loadDataset, type Dataset } from './dataset.js'
import { vectorize } from './vectorize.js'
import { knnSearch } from './knn.js'
import type { FraudScoreRequest, Normalization, MccRisk } from './types.js'

const INDEX_PATH = process.env.INDEX_BIN_PATH ?? '/app/data/index.bin'
const LABELS_PATH = process.env.LABELS_BIN_PATH ?? '/app/data/labels.bin'
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

app.setErrorHandler((_err, _req, reply) => {
  reply.code(200).send({ approved: true, fraud_score: 0 })
})

async function start() {
  console.log('Loading mcc_risk and normalization...')
  mccRisk = JSON.parse(readFileSync(MCC_RISK_PATH, 'utf-8'))
  norm = JSON.parse(readFileSync(NORM_PATH, 'utf-8'))

  console.log(`Loading HNSW index from ${INDEX_PATH} and labels from ${LABELS_PATH}...`)
  const t0 = Date.now()
  dataset = loadDataset(INDEX_PATH, LABELS_PATH)
  console.log(`Dataset loaded in ${Date.now() - t0} ms (${dataset.totalRecords.toLocaleString()} records)`)

  await app.listen({ host: '0.0.0.0', port: PORT })
  console.log(`Server listening on :${PORT}`)
  isReady = true
}

start().catch((err) => {
  console.error('Fatal startup error:', err)
  process.exit(1)
})
