import { openSync, fstatSync, closeSync } from 'node:fs'
import mmapModule from '@fayzanx/mmap-io'
import hnswlib from 'hnswlib-node'

const { HierarchicalNSW } = hnswlib

const mmap = mmapModule as unknown as {
  PROT_READ: number
  MAP_SHARED: number
  map: (size: number, protection: number, flags: number, fd: number, offset?: number) => Buffer
}

const VECTOR_SIZE = 14

// efSearch controls the recall/speed tradeoff at query time.
// Higher = better recall, slower. Tune later.
const DEFAULT_EF_SEARCH = 64

export interface Dataset {
  index: InstanceType<typeof HierarchicalNSW>
  labels: Uint8Array
  totalRecords: number
  vectorSize: number
}

export function loadDataset(indexPath: string, labelsPath: string): Dataset {
  // Load HNSW index (it lives entirely in process RAM — no mmap)
  const index = new HierarchicalNSW('l2', VECTOR_SIZE)
  index.readIndexSync(indexPath)
  index.setEf(DEFAULT_EF_SEARCH)

  // mmap labels (shared between containers via kernel page cache).
  // Size is derived from the file (matches the sampled record count
  // produced by scripts/preprocess.ts).
  const fd = openSync(labelsPath, 'r')
  const stat = fstatSync(fd)
  const totalRecords = stat.size

  const buf = mmap.map(stat.size, mmap.PROT_READ, mmap.MAP_SHARED, fd)
  closeSync(fd)
  const labels = new Uint8Array(buf.buffer, buf.byteOffset, totalRecords)

  return { index, labels, totalRecords, vectorSize: VECTOR_SIZE }
}
