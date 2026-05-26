//! Lloyd's k-means clustering for IVF centroids.
//!
//! Alternates two steps until centroids settle:
//!   1. Assign — each point goes to its nearest centroid.
//!   2. Update — each centroid becomes the mean of points assigned to it.
//!
//! Uses a seeded Mulberry32 PRNG so the build is fully deterministic: the
//! same dataset produces the same centroids every time, which means the
//! preprocess step is cacheable and the runtime behaviour is identical
//! across rebuilds of the Docker image.

use std::collections::HashSet;

use crate::distance::squared_dist;

/// Small, fast, seedable PRNG. Ported from the Mulberry32 algorithm
/// (public domain). Not cryptographically secure; we only need
/// reproducibility.
pub struct Rng {
    state: u32,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed as u32 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    pub fn next_range(&mut self, n: usize) -> usize {
        // Cast to u64 to avoid overflow on the multiplication. The bias
        // here is fine for non-cryptographic use.
        (self.next_u32() as u64 * n as u64 / (1u64 << 32)) as usize
    }
}

pub struct KMeansOpts {
    pub k: usize,
    pub dim: usize,
    pub max_iters: usize,
    pub seed: u64,
    pub tol: f32,
}

/// Train k-means and return `k * dim` centroids in row-major layout.
///
/// `data` is row-major `n * dim` f32. Empty clusters during an iteration
/// are re-seeded from a random point so they don't become useless.
pub fn kmeans(data: &[f32], n: usize, opts: &KMeansOpts) -> Vec<f32> {
    let KMeansOpts { k, dim, max_iters, seed, tol } = *opts;
    debug_assert_eq!(data.len(), n * dim);

    let mut rng = Rng::new(seed);

    // Initialise: pick k distinct random points as starting centroids.
    let mut used: HashSet<usize> = HashSet::with_capacity(k);
    let mut centroids = vec![0.0_f32; k * dim];
    for ki in 0..k {
        let mut pi = rng.next_range(n);
        while !used.insert(pi) {
            pi = rng.next_range(n);
        }
        centroids[ki * dim..(ki + 1) * dim]
            .copy_from_slice(&data[pi * dim..(pi + 1) * dim]);
    }

    let mut assignments = vec![0_u32; n];
    let mut new_centroids = vec![0.0_f32; k * dim];
    let mut counts = vec![0_u32; k];

    for _iter in 0..max_iters {
        // 1. Assign every point to its nearest centroid.
        for i in 0..n {
            let point = &data[i * dim..(i + 1) * dim];
            let mut best_k = 0;
            let mut best_dist = f32::INFINITY;
            for ki in 0..k {
                let centroid = &centroids[ki * dim..(ki + 1) * dim];
                let d = squared_dist(point, centroid);
                if d < best_dist {
                    best_dist = d;
                    best_k = ki;
                }
            }
            assignments[i] = best_k as u32;
        }

        // 2. Recompute centroids as the mean of their assigned points.
        new_centroids.fill(0.0);
        counts.fill(0);
        for i in 0..n {
            let ki = assignments[i] as usize;
            counts[ki] += 1;
            let point = &data[i * dim..(i + 1) * dim];
            for j in 0..dim {
                new_centroids[ki * dim + j] += point[j];
            }
        }
        for ki in 0..k {
            if counts[ki] == 0 {
                // Empty cluster: re-seed from a random data point so it
                // doesn't stay useless forever.
                let pi = rng.next_range(n);
                new_centroids[ki * dim..(ki + 1) * dim]
                    .copy_from_slice(&data[pi * dim..(pi + 1) * dim]);
            } else {
                let inv = 1.0 / counts[ki] as f32;
                for j in 0..dim {
                    new_centroids[ki * dim + j] *= inv;
                }
            }
        }

        // 3. Convergence: stop if centroids barely moved.
        let shift: f32 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum();

        centroids.copy_from_slice(&new_centroids);
        if shift < tol {
            break;
        }
    }

    centroids
}

/// Assign every vector in `data` to its closest centroid.
/// Returns `(assignments, counts_per_cluster)`.
pub fn assign_all(
    data: &[f32],
    n: usize,
    centroids: &[f32],
    k: usize,
    dim: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut assignments = vec![0_u32; n];
    let mut counts = vec![0_u32; k];

    for i in 0..n {
        let point = &data[i * dim..(i + 1) * dim];
        let mut best_k = 0_u32;
        let mut best_dist = f32::INFINITY;
        for ki in 0..k {
            let centroid = &centroids[ki * dim..(ki + 1) * dim];
            let d = squared_dist(point, centroid);
            if d < best_dist {
                best_dist = d;
                best_k = ki as u32;
            }
        }
        assignments[i] = best_k;
        counts[best_k as usize] += 1;
    }

    (assignments, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn kmeans_separates_two_obvious_clusters() {
        // 10 points around (0, 0) and 10 points around (10, 10), K=2.
        let dim = 2;
        let mut data = Vec::with_capacity(20 * dim);
        for i in 0..10 {
            data.push(0.0 + (i as f32) * 0.01);
            data.push(0.0 + (i as f32) * 0.01);
        }
        for i in 0..10 {
            data.push(10.0 + (i as f32) * 0.01);
            data.push(10.0 + (i as f32) * 0.01);
        }
        let opts = KMeansOpts { k: 2, dim, max_iters: 20, seed: 1, tol: 1e-4 };
        let centroids = kmeans(&data, 20, &opts);

        // We don't know which centroid lands where (label-permutation), so
        // sort by the first dimension before checking.
        let c1 = centroids[0]; // dim 0 of centroid 0
        let c2 = centroids[2]; // dim 0 of centroid 1
        let (lo, hi) = if c1 < c2 { (c1, c2) } else { (c2, c1) };
        assert!(lo.abs() < 0.5, "lo={lo}");
        assert!((hi - 10.0).abs() < 0.5, "hi={hi}");
    }

    #[test]
    fn assign_all_groups_close_points_together() {
        let dim = 2;
        let centroids = vec![0.0_f32, 0.0, 10.0, 10.0];
        let data = vec![
            0.1, 0.1, // near centroid 0
            10.2, 10.2, // near centroid 1
            0.3, 0.0, // near centroid 0
        ];
        let (asg, counts) = assign_all(&data, 3, &centroids, 2, dim);
        assert_eq!(asg.to_vec(), vec![0, 1, 0]);
        assert_eq!(counts.to_vec(), vec![2, 1]);
    }
}
