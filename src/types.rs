use serde::{Deserialize, Serialize};

/// Inbound HTTP payload for POST /fraud-score.
///
/// We mirror the JSON shape exactly. Serde will reject any payload whose
/// shape doesn't match (missing fields, wrong types) with a 400 — that's
/// fine for this challenge.
#[derive(Deserialize)]
pub struct FraudScoreRequest {
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    /// `null` in the JSON becomes `None` here — the type system forces us
    /// to handle the "no previous transaction" case explicitly.
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: u32,
    /// ISO-8601 string. Parsed at vectorize time with chrono.
    pub requested_at: String,
}

#[derive(Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f32,
}

/// Outbound HTTP payload.
#[derive(Serialize)]
pub struct FraudScoreResponse {
    pub approved: bool,
    pub fraud_score: f32,
}

/// resources/normalization.json — constants that scale the raw payload
/// values into [0, 1].
#[derive(Deserialize, Clone)]
pub struct Normalization {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

/// resources/mcc_risk.json — string → float lookup.
pub type MccRisk = std::collections::HashMap<String, f32>;
