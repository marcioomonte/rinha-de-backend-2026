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
      requested_at: '2026-03-09T12:00:00Z',
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
    const seg = vectorize(basePayload({
      transaction: { amount: 100, installments: 1, requested_at: '2026-03-09T12:00:00Z' },
    }), mccRisk, norm)
    expect(seg[4]).toBeCloseTo(0 / 6, 5)

    const dom = vectorize(basePayload({
      transaction: { amount: 100, installments: 1, requested_at: '2026-03-15T12:00:00Z' },
    }), mccRisk, norm)
    expect(dom[4]).toBeCloseTo(6 / 6, 5)
  })

  it('matches the legit example from REGRAS_DE_DETECCAO.md', () => {
    const v = vectorize({
      id: 'tx-1329056812',
      transaction: { amount: 41.12, installments: 2, requested_at: '2026-03-11T18:45:53Z' },
      customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ['MERC-003', 'MERC-016'] },
      merchant: { id: 'MERC-016', mcc: '5411', avg_amount: 60.25 },
      terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
      last_transaction: null,
    }, { '5411': 0.15 }, norm)

    expect(v[0]).toBeCloseTo(0.004112, 4)
    expect(v[1]).toBeCloseTo(0.1667, 3)
    expect(v[2]).toBeCloseTo(0.05, 3)
    expect(v[3]).toBeCloseTo(18 / 23, 3)
    expect(v[4]).toBeCloseTo(2 / 6, 3)
    expect(v[5]).toBe(-1)
    expect(v[6]).toBe(-1)
    expect(v[7]).toBeCloseTo(29.23 / 1000, 4)
    expect(v[8]).toBeCloseTo(3 / 20, 4)
    expect(v[9]).toBe(0)
    expect(v[10]).toBe(1)
    expect(v[11]).toBe(0)
    expect(v[12]).toBeCloseTo(0.15, 5)
    expect(v[13]).toBeCloseTo(60.25 / 10000, 5)
  })
})
