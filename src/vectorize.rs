use chrono::{DateTime, Datelike, Timelike, Utc};

use crate::types::{FraudScoreRequest, MccRisk, Normalization};

/// Each query is reduced to exactly 14 dimensions, all in [0, 1] except
/// for indices 5 and 6 which can be the sentinel -1 when
/// `last_transaction` is null.
pub const DIM: usize = 14;

#[inline]
fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

/// Map a payload into the 14-dimension vector. Mirrors the spec in
/// docs/br/REGRAS_DE_DETECCAO.md exactly.
///
/// Returns `Some(vec)` on success, `None` if a date can't be parsed —
/// in the caller (the HTTP layer) we turn `None` into a defensive
/// "approved=true, fraud_score=0.0" response so we never emit a 500.
pub fn vectorize(
    p: &FraudScoreRequest,
    mcc_risk: &MccRisk,
    norm: &Normalization,
) -> Option<[f32; DIM]> {
    let requested_at = DateTime::parse_from_rfc3339(&p.transaction.requested_at)
        .ok()?
        .with_timezone(&Utc);

    let mut v = [0.0_f32; DIM];

    // 0: amount / max_amount
    v[0] = clamp01(p.transaction.amount / norm.max_amount);

    // 1: installments / max_installments
    v[1] = clamp01(p.transaction.installments as f32 / norm.max_installments);

    // 2: (amount / avg_amount) / amount_vs_avg_ratio  — guard against avg=0
    let avg = if p.customer.avg_amount == 0.0 {
        1.0
    } else {
        p.customer.avg_amount
    };
    v[2] = clamp01((p.transaction.amount / avg) / norm.amount_vs_avg_ratio);

    // 3: hour / 23
    v[3] = requested_at.hour() as f32 / 23.0;

    // 4: day_of_week / 6  (Mon=0..Sun=6 — chrono already does this)
    v[4] = requested_at.weekday().num_days_from_monday() as f32 / 6.0;

    // 5, 6: depend on last_transaction (-1 sentinel when null)
    match &p.last_transaction {
        None => {
            v[5] = -1.0;
            v[6] = -1.0;
        }
        Some(lt) => {
            let prev = DateTime::parse_from_rfc3339(&lt.timestamp)
                .ok()?
                .with_timezone(&Utc);
            let minutes =
                (requested_at.timestamp_millis() - prev.timestamp_millis()) as f32 / 60_000.0;
            v[5] = clamp01(minutes / norm.max_minutes);
            v[6] = clamp01(lt.km_from_current / norm.max_km);
        }
    }

    // 7: km_from_home / max_km
    v[7] = clamp01(p.terminal.km_from_home / norm.max_km);

    // 8: tx_count_24h / max_tx_count_24h
    v[8] = clamp01(p.customer.tx_count_24h as f32 / norm.max_tx_count_24h);

    // 9, 10: binary flags
    v[9] = if p.terminal.is_online { 1.0 } else { 0.0 };
    v[10] = if p.terminal.card_present { 1.0 } else { 0.0 };

    // 11: 1 if merchant unknown to customer, else 0
    let is_known = p.customer.known_merchants.iter().any(|m| m == &p.merchant.id);
    v[11] = if is_known { 0.0 } else { 1.0 };

    // 12: mcc_risk lookup, default 0.5 when MCC not in table
    v[12] = mcc_risk.get(&p.merchant.mcc).copied().unwrap_or(0.5);

    // 13: merchant ticket vs ceiling
    v[13] = clamp01(p.merchant.avg_amount / norm.max_merchant_avg_amount);

    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Customer, LastTransaction, Merchant, Terminal, Transaction};

    fn norm() -> Normalization {
        Normalization {
            max_amount: 10_000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10_000.0,
        }
    }

    fn mcc_risk() -> MccRisk {
        let mut m = MccRisk::new();
        m.insert("5411".into(), 0.15);
        m.insert("5912".into(), 0.20);
        m
    }

    fn base_payload() -> FraudScoreRequest {
        FraudScoreRequest {
            id: "tx-1".into(),
            transaction: Transaction {
                amount: 100.0,
                installments: 1,
                requested_at: "2026-03-09T12:00:00Z".into(),
            },
            customer: Customer {
                avg_amount: 100.0,
                tx_count_24h: 1,
                known_merchants: vec!["MERC-001".into()],
            },
            merchant: Merchant {
                id: "MERC-001".into(),
                mcc: "5411".into(),
                avg_amount: 100.0,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 0.0,
            },
            last_transaction: None,
        }
    }

    #[test]
    fn returns_14_dimensions() {
        let v = vectorize(&base_payload(), &mcc_risk(), &norm()).unwrap();
        assert_eq!(v.len(), 14);
    }

    #[test]
    fn null_last_transaction_uses_minus_one_sentinel() {
        let v = vectorize(&base_payload(), &mcc_risk(), &norm()).unwrap();
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
    }

    #[test]
    fn last_transaction_present_computes_minutes() {
        let mut p = base_payload();
        p.transaction.requested_at = "2026-03-09T13:00:00Z".into();
        p.last_transaction = Some(LastTransaction {
            timestamp: "2026-03-09T12:00:00Z".into(),
            km_from_current: 500.0,
        });
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();
        assert!((v[5] - 60.0 / 1440.0).abs() < 1e-5);
        assert!((v[6] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn clamps_amount_above_ceiling() {
        let mut p = base_payload();
        p.transaction.amount = 25_000.0;
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();
        assert_eq!(v[0], 1.0);
    }

    #[test]
    fn unknown_mcc_defaults_to_half() {
        let mut p = base_payload();
        p.merchant.mcc = "9999".into();
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();
        assert_eq!(v[12], 0.5);
    }

    #[test]
    fn unknown_merchant_flips_bit_11() {
        let mut p = base_payload();
        p.customer.known_merchants = vec!["MERC-002".into()];
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();
        assert_eq!(v[11], 1.0);
    }

    #[test]
    fn day_of_week_is_monday_zero() {
        // 2026-03-09 is a Monday
        let mut p = base_payload();
        p.transaction.requested_at = "2026-03-09T12:00:00Z".into();
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();
        assert!((v[4] - 0.0).abs() < 1e-5);

        // 2026-03-15 is a Sunday
        p.transaction.requested_at = "2026-03-15T12:00:00Z".into();
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();
        assert!((v[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn matches_spec_legit_example() {
        // Payload taken verbatim from docs/br/REGRAS_DE_DETECCAO.md
        let p = FraudScoreRequest {
            id: "tx-1329056812".into(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".into(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };
        let v = vectorize(&p, &mcc_risk(), &norm()).unwrap();

        // expected: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292,
        //           0.15, 0, 1, 0, 0.15, 0.006]
        assert!((v[0] - 0.004112).abs() < 1e-4);
        assert!((v[1] - 0.16667).abs() < 1e-3);
        assert!((v[2] - 0.05).abs() < 1e-3);
        assert!((v[3] - 18.0 / 23.0).abs() < 1e-3);
        assert!((v[4] - 2.0 / 6.0).abs() < 1e-3); // 2026-03-11 is Wednesday → 2
        assert_eq!(v[5], -1.0);
        assert_eq!(v[6], -1.0);
        assert!((v[7] - 29.23 / 1000.0).abs() < 1e-4);
        assert!((v[8] - 3.0 / 20.0).abs() < 1e-4);
        assert_eq!(v[9], 0.0);
        assert_eq!(v[10], 1.0);
        assert_eq!(v[11], 0.0);
        assert!((v[12] - 0.15).abs() < 1e-5);
        assert!((v[13] - 60.25 / 10_000.0).abs() < 1e-5);
    }
}
