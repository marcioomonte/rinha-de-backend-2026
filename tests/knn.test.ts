import { describe, it, expect } from 'vitest'
import { knnSearch } from '../src/knn.js'

function makeFixture() {
  const vectors = new Float32Array(6 * 14)
  const labels = new Uint8Array(6)

  function setRecord(i: number, baseValue: number, label: number) {
    for (let d = 0; d < 14; d++) vectors[i * 14 + d] = baseValue
    labels[i] = label
  }

  setRecord(0, 0.10, 0)
  setRecord(1, 0.11, 1)
  setRecord(2, 0.20, 0)
  setRecord(3, 0.90, 1)
  setRecord(4, 0.05, 0)
  setRecord(5, 0.15, 1)

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
    // Top-5 nearest to q=0.10: records 0(0.10), 1(0.11), 4(0.05), 5(0.15), 2(0.20)
    // Labels: 0, 1, 0, 1, 0 → 2 fraudes
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
