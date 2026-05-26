import type { Dataset } from './dataset.js'

const K = 5
const VECTOR_SIZE = 14

const topDists = new Float64Array(K)
const topLabels = new Uint8Array(K)

export function knnSearch(
  query: Float32Array,
  dataset: Pick<Dataset, 'vectors' | 'labels' | 'totalRecords'>
): number {
  const { vectors, labels, totalRecords } = dataset

  for (let k = 0; k < K; k++) {
    topDists[k] = Infinity
    topLabels[k] = 0
  }

  for (let i = 0; i < totalRecords; i++) {
    const base = i * VECTOR_SIZE
    const worst = topDists[K - 1]

    let d = 0
    let diff: number

    diff = query[0] - vectors[base];      d  = diff * diff
    diff = query[1] - vectors[base + 1];  d += diff * diff
    diff = query[2] - vectors[base + 2];  d += diff * diff
    diff = query[3] - vectors[base + 3];  d += diff * diff
    diff = query[4] - vectors[base + 4];  d += diff * diff
    diff = query[5] - vectors[base + 5];  d += diff * diff
    diff = query[6] - vectors[base + 6];  d += diff * diff
    if (d >= worst) continue
    diff = query[7] - vectors[base + 7];  d += diff * diff
    diff = query[8] - vectors[base + 8];  d += diff * diff
    diff = query[9] - vectors[base + 9];  d += diff * diff
    if (d >= worst) continue
    diff = query[10] - vectors[base + 10]; d += diff * diff
    diff = query[11] - vectors[base + 11]; d += diff * diff
    diff = query[12] - vectors[base + 12]; d += diff * diff
    diff = query[13] - vectors[base + 13]; d += diff * diff

    if (d >= worst) continue

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
