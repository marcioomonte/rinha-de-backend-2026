import type { Dataset } from './dataset.js'

const K = 5

export function knnSearch(
  query: Float32Array,
  dataset: Pick<Dataset, 'index' | 'labels'>
): number {
  // hnswlib-node accepts a plain number[] — convert from Float32Array
  const queryArr = Array.from(query)
  const result = dataset.index.searchKnn(queryArr, K)
  const neighbors = result.neighbors

  let fraudCount = 0
  for (let i = 0; i < neighbors.length; i++) {
    fraudCount += dataset.labels[neighbors[i]]
  }
  return fraudCount
}
