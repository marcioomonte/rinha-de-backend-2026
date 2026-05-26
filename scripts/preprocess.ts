import { createReadStream, openSync, writeSync, closeSync, mkdirSync } from 'node:fs'
import { createGunzip } from 'node:zlib'
import streamArray from 'stream-json/streamers/StreamArray.js'

const StreamArray = streamArray

const TOTAL_RECORDS = 3_000_000
const VECTOR_SIZE = 14
const VECTOR_BYTES = VECTOR_SIZE * 4
const LABEL_BYTES = 1

const INPUT = 'resources/references.json.gz'
const OUTPUT_DIR = 'data'
const OUTPUT = `${OUTPUT_DIR}/refs.bin`

async function main(): Promise<void> {
  console.log(`Reading ${INPUT}`)
  console.log(`Writing ${OUTPUT}`)
  const totalBytes = TOTAL_RECORDS * (VECTOR_BYTES + LABEL_BYTES)
  console.log(`Expected size: ${(totalBytes / 1024 / 1024).toFixed(1)} MB`)

  mkdirSync(OUTPUT_DIR, { recursive: true })

  const vectorsBuf = Buffer.allocUnsafe(TOTAL_RECORDS * VECTOR_BYTES)
  const labelsBuf = Buffer.allocUnsafe(TOTAL_RECORDS * LABEL_BYTES)

  let count = 0
  const source = createReadStream(INPUT)
    .pipe(createGunzip())
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
