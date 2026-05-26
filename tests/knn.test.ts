import { describe, it, expect } from 'vitest'
import { knnSearch } from '../src/knn.js'
import hnswlib from 'hnswlib-node'

const { HierarchicalNSW } = hnswlib

function makeFixture(records: { vec: number[]; label: number }[]) {
  const idx = new HierarchicalNSW('l2', 14)
  idx.initIndex(records.length, 16, 200, 42)
  const labels = new Uint8Array(records.length)
  records.forEach((r, i) => {
    idx.addPoint(r.vec, i)
    labels[i] = r.label
  })
  idx.setEf(64)
  return { index: idx, labels }
}

describe('knnSearch (HNSW)', () => {
  it('returns 0..5 number of fraud labels among the K nearest', () => {
    const fixture = makeFixture([
      { vec: new Array(14).fill(0.10), label: 0 },
      { vec: new Array(14).fill(0.11), label: 1 },
      { vec: new Array(14).fill(0.20), label: 0 },
      { vec: new Array(14).fill(0.90), label: 1 },
      { vec: new Array(14).fill(0.05), label: 0 },
      { vec: new Array(14).fill(0.15), label: 1 },
    ])
    const q = new Float32Array(14).fill(0.10)
    const result = knnSearch(q, fixture)
    expect(result).toBeGreaterThanOrEqual(0)
    expect(result).toBeLessThanOrEqual(5)
  })

  it('returns 5 when all neighbors are fraud', () => {
    const fixture = makeFixture([
      { vec: new Array(14).fill(0.5), label: 1 },
      { vec: new Array(14).fill(0.5), label: 1 },
      { vec: new Array(14).fill(0.5), label: 1 },
      { vec: new Array(14).fill(0.5), label: 1 },
      { vec: new Array(14).fill(0.5), label: 1 },
    ])
    const q = new Float32Array(14).fill(0.5)
    expect(knnSearch(q, fixture)).toBe(5)
  })

  it('returns 0 when all neighbors are legit', () => {
    const fixture = makeFixture([
      { vec: new Array(14).fill(0.5), label: 0 },
      { vec: new Array(14).fill(0.5), label: 0 },
      { vec: new Array(14).fill(0.5), label: 0 },
      { vec: new Array(14).fill(0.5), label: 0 },
      { vec: new Array(14).fill(0.5), label: 0 },
    ])
    const q = new Float32Array(14).fill(0.5)
    expect(knnSearch(q, fixture)).toBe(0)
  })
})
