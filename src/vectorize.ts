import type { FraudScoreRequest, Normalization, MccRisk } from './types.js'

function clamp(x: number): number {
  if (x < 0) return 0
  if (x > 1) return 1
  return x
}

export function vectorize(
  p: FraudScoreRequest,
  mccRisk: MccRisk,
  norm: Normalization
): Float32Array {
  const v = new Float32Array(14)
  const date = new Date(p.transaction.requested_at)

  v[0] = clamp(p.transaction.amount / norm.max_amount)
  v[1] = clamp(p.transaction.installments / norm.max_installments)

  const avgAmount = p.customer.avg_amount === 0 ? 1 : p.customer.avg_amount
  v[2] = clamp((p.transaction.amount / avgAmount) / norm.amount_vs_avg_ratio)

  v[3] = date.getUTCHours() / 23
  v[4] = ((date.getUTCDay() + 6) % 7) / 6

  if (p.last_transaction !== null) {
    const ms = date.getTime() - new Date(p.last_transaction.timestamp).getTime()
    const minutes = ms / 60000
    v[5] = clamp(minutes / norm.max_minutes)
    v[6] = clamp(p.last_transaction.km_from_current / norm.max_km)
  } else {
    v[5] = -1
    v[6] = -1
  }

  v[7] = clamp(p.terminal.km_from_home / norm.max_km)
  v[8] = clamp(p.customer.tx_count_24h / norm.max_tx_count_24h)
  v[9] = p.terminal.is_online ? 1 : 0
  v[10] = p.terminal.card_present ? 1 : 0
  v[11] = p.customer.known_merchants.includes(p.merchant.id) ? 0 : 1
  v[12] = mccRisk[p.merchant.mcc] ?? 0.5
  v[13] = clamp(p.merchant.avg_amount / norm.max_merchant_avg_amount)

  return v
}
