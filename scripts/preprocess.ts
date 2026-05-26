import { createReadStream, openSync, writeSync, closeSync, mkdirSync } from 'node:fs'
import { createGunzip } from 'node:zlib'
import streamArray from 'stream-json/streamers/StreamArray.js'
import hnswlib from 'hnswlib-node'

const StreamArray = streamArray
const { HierarchicalNSW } = hnswlib

const TOTAL_RECORDS = 3_000_000
const VECTOR_SIZE = 14

// Uniform downsampling to fit the HNSW index within the 160 MB per-instance
// RAM budget. SAMPLE_RATE=6 keeps one in every six records -> ~500K samples.
const SAMPLE_RATE = 12
const SAMPLED_RECORDS = Math.ceil(TOTAL_RECORDS / SAMPLE_RATE)

const INPUT = 'resources/references.json.gz'
const OUTPUT_DIR = 'data'
const INDEX_OUT = `${OUTPUT_DIR}/index.bin`
const LABELS_OUT = `${OUTPUT_DIR}/labels.bin`

// HNSW build parameters
const HNSW_M = 4
const HNSW_EF_CONSTRUCTION = 200

async function main(): Promise<void> {
  console.log(`Reading ${INPUT}`)
  console.log(`Writing ${INDEX_OUT} and ${LABELS_OUT}`)
  console.log(`HNSW params: M=${HNSW_M}, efConstruction=${HNSW_EF_CONSTRUCTION}`)

  mkdirSync(OUTPUT_DIR, { recursive: true })

  const labelsBuf = Buffer.allocUnsafe(SAMPLED_RECORDS)

  const index = new HierarchicalNSW('l2', VECTOR_SIZE)
  index.initIndex(SAMPLED_RECORDS, HNSW_M, HNSW_EF_CONSTRUCTION, 42)

  let recordIdx = 0
  let sampleCount = 0
  const tStart = Date.now()
  const source = createReadStream(INPUT)
    .pipe(createGunzip())
    .pipe(StreamArray.withParser())

  for await (const { value } of source) {
    if (recordIdx % SAMPLE_RATE === 0) {
      const vec: number[] = value.vector
      if (vec.length !== VECTOR_SIZE) {
        throw new Error(`Record ${recordIdx} has vector length ${vec.length}, expected ${VECTOR_SIZE}`)
      }
      if (sampleCount >= SAMPLED_RECORDS) {
        throw new Error(`Sampled more than ${SAMPLED_RECORDS} records — aborting`)
      }

      index.addPoint(vec, sampleCount)
      labelsBuf[sampleCount] = value.label === 'fraud' ? 1 : 0
      sampleCount++

      if (sampleCount % 50_000 === 0) {
        const elapsed = ((Date.now() - tStart) / 1000).toFixed(1)
        console.log(`  ${sampleCount.toLocaleString()} / ${SAMPLED_RECORDS.toLocaleString()}  (${elapsed}s)`)
      }
    }
    recordIdx++
  }

  if (recordIdx !== TOTAL_RECORDS) {
    throw new Error(`Expected ${TOTAL_RECORDS} input records, got ${recordIdx}`)
  }

  console.log(`Writing HNSW index...`)
  index.writeIndexSync(INDEX_OUT)

  console.log(`Writing labels...`)
  const fd = openSync(LABELS_OUT, 'w')
  writeSync(fd, labelsBuf)
  closeSync(fd)

  const totalSec = ((Date.now() - tStart) / 1000).toFixed(1)
  console.log(`Done in ${totalSec}s. Wrote ${sampleCount.toLocaleString()} samples (1 in every ${SAMPLE_RATE}).`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
