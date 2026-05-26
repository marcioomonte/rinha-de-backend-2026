/// Squared Euclidean distance between two equal-length slices.
///
/// We square to avoid a `sqrt()` per call — relative ordering is the
/// same, and the hot path inside k-means and IVF only ever compares
/// distances against each other.
#[inline]
pub fn squared_dist(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut d = 0.0_f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        d += diff * diff;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_when_equal() {
        let a = [0.1_f32, 0.2, 0.3];
        assert!((squared_dist(&a, &a) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn pythagorean_triple() {
        let a = [0.0_f32, 0.0];
        let b = [3.0_f32, 4.0];
        assert!((squared_dist(&a, &b) - 25.0).abs() < 1e-5);
    }
}
