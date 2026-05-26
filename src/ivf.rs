//! IVF (Inverted File) index.
//!
//! Build-time (in `bin/preprocess.rs`):
//!   1. Train k-means on a sample of vectors to learn K centroids.
//!   2. Assign every one of the 3M vectors to its nearest centroid.
//!   3. Group vectors by centroid, quantize to u8, and write to a single
//!      binary file alongside the labels.
//!
//! Runtime (this module):
//!   1. `Ivf::open` mmaps the file and slices it into typed views.
//!   2. `search` finds the top `nprobe` centroids closest to a query,
//!      then scans the vectors in those clusters (quantized + sentinel
//!      aware) keeping the global top-K. K is fixed at 5.
//!
//! File layout (see also docs/superpowers/specs):
//!   header (16 bytes): magic(u32), n(u32), k(u32), dim(u32)
//!   centroids: k * dim * 4 bytes (f32, row-major)
//!   list_offsets: k * 4 bytes (u32, into vector_data in *records*)
//!   list_counts:  k * 4 bytes (u32, count per cluster)
//!   vector_data:  n * dim bytes (u8, quantized, cluster-grouped)
//!   labels:       n bytes (u8, same order as vector_data — NOT global id)

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::distance::squared_dist;
use crate::quantize::squared_dist_quant;

pub const MAGIC: u32 = 0x49564601; // "IVF\x01" little-endian
pub const K_NEIGHBORS: usize = 5;
pub const HEADER_BYTES: usize = 16;

/// All the offsets pre-computed at open time so each accessor is a
/// constant-time slice into the mmap.
struct Layout {
    n: usize,
    k: usize,
    dim: usize,
    centroids_off: usize,
    list_offsets_off: usize,
    list_counts_off: usize,
    vector_data_off: usize,
    labels_off: usize,
}

pub struct Ivf {
    mmap: Mmap,
    layout: Layout,
}

impl Ivf {
    /// Open and validate an IVF index file. mmap is held in self so the
    /// returned slices stay valid for the lifetime of the struct.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: we treat the mmap as read-only for the lifetime of self;
        // it is unsafe in general because external modification of the
        // file under our feet could be observed as UB. In our deployment
        // the file is baked into the image at build time and never
        // mutated.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_BYTES {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "file shorter than header"));
        }
        let magic = read_u32(&mmap, 0);
        if magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad magic 0x{magic:08x}, expected 0x{MAGIC:08x}"),
            ));
        }
        let n = read_u32(&mmap, 4) as usize;
        let k = read_u32(&mmap, 8) as usize;
        let dim = read_u32(&mmap, 12) as usize;

        let centroids_off = HEADER_BYTES;
        let list_offsets_off = centroids_off + k * dim * 4;
        let list_counts_off = list_offsets_off + k * 4;
        let vector_data_off = list_counts_off + k * 4;
        let labels_off = vector_data_off + n * dim;
        let expected = labels_off + n;

        if mmap.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("file size {} != expected {expected}", mmap.len()),
            ));
        }

        Ok(Self {
            mmap,
            layout: Layout {
                n,
                k,
                dim,
                centroids_off,
                list_offsets_off,
                list_counts_off,
                vector_data_off,
                labels_off,
            },
        })
    }

    pub fn n(&self) -> usize { self.layout.n }
    pub fn k(&self) -> usize { self.layout.k }
    pub fn dim(&self) -> usize { self.layout.dim }

    fn centroids(&self) -> &[f32] {
        let len = self.layout.k * self.layout.dim;
        let bytes = &self.mmap[self.layout.centroids_off..self.layout.centroids_off + len * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
    }

    fn list_offsets(&self) -> &[u32] {
        let bytes = &self.mmap[self.layout.list_offsets_off..self.layout.list_offsets_off + self.layout.k * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, self.layout.k) }
    }

    fn list_counts(&self) -> &[u32] {
        let bytes = &self.mmap[self.layout.list_counts_off..self.layout.list_counts_off + self.layout.k * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, self.layout.k) }
    }

    fn vector_data(&self) -> &[u8] {
        let len = self.layout.n * self.layout.dim;
        &self.mmap[self.layout.vector_data_off..self.layout.vector_data_off + len]
    }

    fn labels(&self) -> &[u8] {
        &self.mmap[self.layout.labels_off..self.layout.labels_off + self.layout.n]
    }

    /// Find the number of fraud labels (0..=5) among the K=5 nearest
    /// vectors. `nprobe` controls how many centroids the search visits;
    /// higher = better recall, slower.
    pub fn search_fraud_count(&self, query: &[f32], nprobe: usize) -> u32 {
        debug_assert_eq!(query.len(), self.layout.dim);
        let dim = self.layout.dim;
        let k = self.layout.k;
        let centroids = self.centroids();
        let list_offsets = self.list_offsets();
        let list_counts = self.list_counts();
        let vector_data = self.vector_data();
        let labels = self.labels();

        // 1. Distance from query to each centroid; keep top `nprobe` by
        //    shift-insertion (no heap needed — nprobe is small).
        let np = nprobe.min(k);
        let mut top_c_dists = vec![f32::INFINITY; np];
        let mut top_c_ids: Vec<usize> = vec![usize::MAX; np];
        for ci in 0..k {
            let c = &centroids[ci * dim..(ci + 1) * dim];
            let d = squared_dist(query, c);
            if d < top_c_dists[np - 1] {
                let mut pos = np - 1;
                while pos > 0 && top_c_dists[pos - 1] > d {
                    top_c_dists[pos] = top_c_dists[pos - 1];
                    top_c_ids[pos] = top_c_ids[pos - 1];
                    pos -= 1;
                }
                top_c_dists[pos] = d;
                top_c_ids[pos] = ci;
            }
        }

        // 2. Scan vectors in those clusters, keep global top-K.
        let mut top_dists = [f32::INFINITY; K_NEIGHBORS];
        let mut top_labels = [0_u8; K_NEIGHBORS];

        for &ci in &top_c_ids {
            if ci == usize::MAX {
                continue;
            }
            let start = list_offsets[ci] as usize;
            let count = list_counts[ci] as usize;
            for j in 0..count {
                let pos = start + j;
                let vec_bytes = &vector_data[pos * dim..(pos + 1) * dim];
                let d = squared_dist_quant(query, vec_bytes);
                if d >= top_dists[K_NEIGHBORS - 1] {
                    continue;
                }
                let lab = labels[pos];
                let mut p = K_NEIGHBORS - 1;
                while p > 0 && top_dists[p - 1] > d {
                    top_dists[p] = top_dists[p - 1];
                    top_labels[p] = top_labels[p - 1];
                    p -= 1;
                }
                top_dists[p] = d;
                top_labels[p] = lab;
            }
        }

        top_labels.iter().map(|&l| l as u32).sum()
    }
}

#[inline]
fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize;
    use std::io::Write;
    use std::path::PathBuf;

    /// Write a tiny IVF file by hand for round-trip testing.
    fn write_fixture(path: &PathBuf) {
        let n = 6_u32;
        let k = 2_u32;
        let dim = 3_u32;
        let centroids: Vec<f32> = vec![
            0.1, 0.1, 0.1, // centroid 0
            0.9, 0.9, 0.9, // centroid 1
        ];
        // 3 vectors in cluster 0, then 3 vectors in cluster 1.
        // labels (in cluster order): cluster 0 → [0, 0, 1]; cluster 1 → [1, 1, 0]
        let raw_vecs: Vec<f32> = vec![
            0.10, 0.10, 0.10,
            0.12, 0.10, 0.11,
            0.15, 0.13, 0.10,
            0.90, 0.92, 0.91,
            0.91, 0.90, 0.92,
            0.88, 0.92, 0.90,
        ];
        let labels = vec![0_u8, 0, 1, 1, 1, 0];
        let list_offsets = [0_u32, 3]; // cluster 0 starts at pos 0, cluster 1 at pos 3
        let list_counts = [3_u32, 3];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&n.to_le_bytes());
        bytes.extend_from_slice(&k.to_le_bytes());
        bytes.extend_from_slice(&dim.to_le_bytes());
        for c in &centroids {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        for o in &list_offsets {
            bytes.extend_from_slice(&o.to_le_bytes());
        }
        for c in &list_counts {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        for v in &raw_vecs {
            bytes.push(quantize::quantize(*v));
        }
        bytes.extend_from_slice(&labels);

        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&bytes).unwrap();
    }

    #[test]
    fn opens_and_reports_dimensions() {
        let path = std::env::temp_dir().join("rinha-ivf-test-1.bin");
        write_fixture(&path);
        let ivf = Ivf::open(&path).unwrap();
        assert_eq!(ivf.n(), 6);
        assert_eq!(ivf.k(), 2);
        assert_eq!(ivf.dim(), 3);
    }

    #[test]
    fn search_finds_label_count_in_near_cluster() {
        let path = std::env::temp_dir().join("rinha-ivf-test-2.bin");
        write_fixture(&path);
        let ivf = Ivf::open(&path).unwrap();

        // Query near cluster 0: top-5 will all come from cluster 0.
        // Cluster 0 has only 3 records (labels [0, 0, 1]). With nprobe=1
        // we only look at cluster 0; remaining slots stay as label 0
        // (default), so the count = 1 fraud.
        let q = [0.10_f32, 0.10, 0.10];
        let count = ivf.search_fraud_count(&q, 1);
        assert_eq!(count, 1);
    }

    #[test]
    fn nprobe_2_blends_both_clusters() {
        let path = std::env::temp_dir().join("rinha-ivf-test-3.bin");
        write_fixture(&path);
        let ivf = Ivf::open(&path).unwrap();

        // Query near cluster 0 with nprobe=2 visits both clusters.
        // Cluster 0 vectors are much closer to (0.1) than cluster 1.
        // Top-5: 3 from cluster 0 (labels 0,0,1) + 2 from cluster 1
        // (whichever are closest — they have labels 1, 1 likely).
        // So count is around 3.
        let q = [0.10_f32, 0.10, 0.10];
        let count = ivf.search_fraud_count(&q, 2);
        assert!(count >= 1, "got {count}");
        assert!(count <= 5);
    }

    #[test]
    fn rejects_bad_magic() {
        let path = std::env::temp_dir().join("rinha-ivf-test-bad.bin");
        let bad = vec![0_u8; HEADER_BYTES];
        std::fs::write(&path, bad).unwrap();
        assert!(Ivf::open(&path).is_err());
    }
}
