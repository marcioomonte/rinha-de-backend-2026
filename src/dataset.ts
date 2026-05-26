import { openSync, fstatSync, closeSync } from 'node:fs'
import mmapModule from '@fayzanx/mmap-io'

const mmap = mmapModule as unknown as {
  PROT_READ: number
  MAP_SHARED: number
  map: (size: number, protection: number, flags: number, fd: number, offset?: number) => Buffer
}

const TOTAL_RECORDS = 3_000_000
const VECTOR_SIZE = 14
const VECTOR_BYTES_TOTAL = TOTAL_RECORDS * VECTOR_SIZE * 4
const LABEL_BYTES_TOTAL = TOTAL_RECORDS
const EXPECTED_SIZE = VECTOR_BYTES_TOTAL + LABEL_BYTES_TOTAL

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

  const buf = mmap.map(stat.size, mmap.PROT_READ, mmap.MAP_SHARED, fd)
  closeSync(fd)

  const vectors = new Float32Array(buf.buffer, buf.byteOffset, VECTOR_BYTES_TOTAL / 4)
  const labels = new Uint8Array(buf.buffer, buf.byteOffset + VECTOR_BYTES_TOTAL, LABEL_BYTES_TOTAL)

  return { vectors, labels, totalRecords: TOTAL_RECORDS, vectorSize: VECTOR_SIZE }
}
