//! Scalar quantization: f32 in [0, 1] (plus the -1 sentinel) → u8.
//!
//! Storing 14-dim vectors as f32 is 56 bytes per record. Quantizing each
//! dimension to a single byte cuts that to 14 bytes — a 4x reduction
//! that lets the full 3M dataset fit comfortably in RAM.
//!
//! Mapping:
//!   sentinel -1.0      →  255  (reserved)
//!   v in [0.0, 1.0]    →  round(v * 254), so 0 maps to 0 and 1 maps to 254
//!
//! Round-trip precision is 1/254 ≈ 0.004, which is finer than the
//! per-dimension uncertainty introduced by the normalization itself,
//! so we expect zero accuracy loss in practice.

pub const SENTINEL_QUANT: u8 = 255;
pub const SCALE: f32 = 254.0;

#[inline]
pub fn quantize(v: f32) -> u8 {
    if v == -1.0 {
        return SENTINEL_QUANT;
    }
    if v <= 0.0 {
        return 0;
    }
    if v >= 1.0 {
        return SCALE as u8;
    }
    (v * SCALE).round() as u8
}

#[inline]
pub fn dequantize(q: u8) -> f32 {
    if q == SENTINEL_QUANT {
        -1.0
    } else {
        q as f32 / SCALE
    }
}

/// Quantize each element of `vec` into the corresponding slot of `out`.
/// Both slices must be the same length.
pub fn quantize_into(vec: &[f32], out: &mut [u8]) {
    debug_assert_eq!(vec.len(), out.len());
    for (i, &v) in vec.iter().enumerate() {
        out[i] = quantize(v);
    }
}

/// Squared Euclidean distance between a Float32 query and a quantized
/// (u8) reference slice. Dequantizes on the fly.
///
/// This is the hot-path distance used inside an IVF cluster: we never
/// keep dequantized copies of the full dataset in memory.
#[inline]
pub fn squared_dist_quant(query: &[f32], data: &[u8]) -> f32 {
    debug_assert_eq!(query.len(), data.len());
    let mut d = 0.0_f32;
    for i in 0..query.len() {
        let rv = if data[i] == SENTINEL_QUANT {
            -1.0
        } else {
            data[i] as f32 / SCALE
        };
        let diff = query[i] - rv;
        d += diff * diff;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_map_correctly() {
        assert_eq!(quantize(0.0), 0);
        assert_eq!(quantize(1.0), SCALE as u8);
    }

    #[test]
    fn sentinel_round_trip() {
        assert_eq!(quantize(-1.0), SENTINEL_QUANT);
        assert_eq!(dequantize(SENTINEL_QUANT), -1.0);
    }

    #[test]
    fn round_trip_within_precision() {
        let mut v = 0.0_f32;
        while v <= 1.0 {
            let back = dequantize(quantize(v));
            assert!((back - v).abs() <= 1.0 / SCALE + 1e-6, "v={v} back={back}");
            v += 0.01;
        }
    }

    #[test]
    fn out_of_range_clamps() {
        assert_eq!(quantize(-0.5), 0);
        assert_eq!(quantize(1.5), SCALE as u8);
    }

    #[test]
    fn squared_dist_quant_matching_zero() {
        let q = [0.5_f32, 0.0, 1.0];
        let data: [u8; 3] = [quantize(0.5), quantize(0.0), quantize(1.0)];
        assert!(squared_dist_quant(&q, &data) < 1e-3);
    }

    #[test]
    fn squared_dist_quant_penalises_mismatched_sentinel() {
        // query has a real value but stored side has -1 → diff = 1.5, sq = 2.25
        let q = [0.5_f32];
        let data = [SENTINEL_QUANT];
        let d = squared_dist_quant(&q, &data);
        assert!((d - 2.25).abs() < 1e-3, "d={d}");
    }
}
