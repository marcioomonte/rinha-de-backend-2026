export interface FraudScoreRequest {
  id: string
  transaction: {
    amount: number
    installments: number
    requested_at: string
  }
  customer: {
    avg_amount: number
    tx_count_24h: number
    known_merchants: string[]
  }
  merchant: {
    id: string
    mcc: string
    avg_amount: number
  }
  terminal: {
    is_online: boolean
    card_present: boolean
    km_from_home: number
  }
  last_transaction: {
    timestamp: string
    km_from_current: number
  } | null
}

export interface FraudScoreResponse {
  approved: boolean
  fraud_score: number
}

export interface Normalization {
  max_amount: number
  max_installments: number
  amount_vs_avg_ratio: number
  max_minutes: number
  max_km: number
  max_tx_count_24h: number
  max_merchant_avg_amount: number
}

export type MccRisk = Record<string, number>
