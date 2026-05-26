//! Build-time job that reads `resources/references.json.gz`, trains the
//! IVF centroids, assigns every vector to a cluster, quantizes, and
//! writes the binary index that the server mmaps at runtime.
//!
//! Runs once during `docker build`. The output (`data/ivf.bin`) is
//! shipped as part of the image, so the runtime container does no JSON
//! parsing or k-means at all.

use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::time::Instant;

use flate2::read::GzDecoder;

// Pull in the library modules. Since this is a bin crate without an
// explicit lib, we declare the modules locally here. In a larger
// project we'd refactor into `lib.rs` + multiple binaries.
#[path = "../distance.rs"] mod distance;
#[path = "../kmeans.rs"] mod kmeans;
#[path = "../quantize.rs"] mod quantize;

use kmeans::{assign_all, kmeans, KMeansOpts};
use quantize::quantize_into;

const TOTAL_RECORDS: usize = 3_000_000;
const DIM: usize = 14;

// IVF build parameters
const K_CENTROIDS: usize = 1024;
const KMEANS_SAMPLE_SIZE: usize = 100_000;
const KMEANS_MAX_ITERS: usize = 20;
const KMEANS_TOL: f32 = 1e-4;
const SEED: u64 = 42;

const MAGIC: u32 = 0x49564601;

const INPUT: &str = "resources/references.json.gz";
const OUTPUT: &str = "data/ivf.bin";

fn main() {
    let t0 = Instant::now();
    println!("preprocess: reading {INPUT}");

    let (vectors, labels) = read_dataset(INPUT);
    assert_eq!(vectors.len(), TOTAL_RECORDS * DIM);
    assert_eq!(labels.len(), TOTAL_RECORDS);
    println!("  parsed {} records in {:.1}s", TOTAL_RECORDS, t0.elapsed().as_secs_f32());

    // Train k-means on a deterministic sub-sample
    let t1 = Instant::now();
    let sample = build_sample(&vectors, SEED);
    println!("  built {KMEANS_SAMPLE_SIZE}-point sample in {:.1}s", t1.elapsed().as_secs_f32());

    let t2 = Instant::now();
    let opts = KMeansOpts {
        k: K_CENTROIDS,
        dim: DIM,
        max_iters: KMEANS_MAX_ITERS,
        seed: SEED,
        tol: KMEANS_TOL,
    };
    let centroids = kmeans(&sample, KMEANS_SAMPLE_SIZE, &opts);
    println!("  trained k-means (K={K_CENTROIDS}) in {:.1}s", t2.elapsed().as_secs_f32());

    // Assign all 3M vectors to centroids
    let t3 = Instant::now();
    let (assignments, counts) = assign_all(&vectors, TOTAL_RECORDS, &centroids, K_CENTROIDS, DIM);
    println!("  assigned {TOTAL_RECORDS} vectors in {:.1}s", t3.elapsed().as_secs_f32());

    // Build inverted lists (cluster-grouped order) and write the file
    let t4 = Instant::now();
    write_index(OUTPUT, &vectors, &labels, &centroids, &assignments, &counts);
    println!("  wrote {OUTPUT} in {:.1}s", t4.elapsed().as_secs_f32());

    println!("preprocess: done in {:.1}s total", t0.elapsed().as_secs_f32());
}

/// Stream the .gz, parse 3M `{ "vector": [...], "label": "..." }` records
/// into a packed Float32 buffer + a per-record label byte.
fn read_dataset(path: &str) -> (Vec<f32>, Vec<u8>) {
    let file = std::fs::File::open(path).expect("open references.json.gz");
    let reader = BufReader::new(GzDecoder::new(file));

    // We use a tiny hand-rolled streaming JSON parser tailored to this
    // file's known shape (`[ {...}, {...}, ... ]`). Pulling in serde_json
    // for a 200-line repetitive grammar would be overkill and slower.
    parse_records(reader)
}

/// Parse the JSON shape:
///   [
///     { "vector": [f, f, ..., f], "label": "fraud" | "legit" },
///     ...
///   ]
fn parse_records<R: Read>(mut r: R) -> (Vec<f32>, Vec<u8>) {
    let mut vectors = Vec::with_capacity(TOTAL_RECORDS * DIM);
    let mut labels = Vec::with_capacity(TOTAL_RECORDS);

    // Slurp the decompressed bytes. ~284 MB at build time — fine, we have
    // plenty of memory in the builder stage.
    let mut buf = Vec::with_capacity(290 * 1024 * 1024);
    r.read_to_end(&mut buf).expect("read gunzipped JSON");
    let s = std::str::from_utf8(&buf).expect("UTF-8 JSON");

    // The records are tightly formatted. We just walk the string looking
    // for the next "vector" / "label" key.
    let bytes = s.as_bytes();
    let mut i = 0;
    let n = bytes.len();
    let mut count = 0_usize;
    while i < n && count < TOTAL_RECORDS {
        // Find next '{'
        while i < n && bytes[i] != b'{' { i += 1; }
        if i >= n { break; }
        // Parse one record: { "vector": [...], "label": "..." }
        i += 1; // skip '{'

        let (vec, after_vec) = read_vector_field(bytes, i);
        i = after_vec;
        for v in &vec {
            vectors.push(*v);
        }

        let (label_byte, after_lbl) = read_label_field(bytes, i);
        i = after_lbl;
        labels.push(label_byte);

        count += 1;
        if count % 250_000 == 0 {
            println!("    parsed {count}");
        }
    }
    assert_eq!(count, TOTAL_RECORDS, "expected {TOTAL_RECORDS} records, got {count}");

    (vectors, labels)
}

fn read_vector_field(b: &[u8], start: usize) -> ([f32; DIM], usize) {
    // Find "vector"
    let mut i = start;
    let key = b"\"vector\"";
    while i + key.len() <= b.len() && &b[i..i + key.len()] != key { i += 1; }
    i += key.len();
    // Find '['
    while i < b.len() && b[i] != b'[' { i += 1; }
    i += 1;

    let mut out = [0_f32; DIM];
    for d in 0..DIM {
        while i < b.len() && (b[i] == b' ' || b[i] == b',') { i += 1; }
        // Parse a number — supports decimals, '-', 'e'/'E', '.'.
        let num_start = i;
        while i < b.len() && b[i] != b',' && b[i] != b']' && b[i] != b' ' { i += 1; }
        let s = std::str::from_utf8(&b[num_start..i]).expect("number utf8");
        out[d] = s.parse::<f32>().expect("number parse");
    }
    // Skip past ']'
    while i < b.len() && b[i] != b']' { i += 1; }
    i += 1;
    (out, i)
}

fn read_label_field(b: &[u8], start: usize) -> (u8, usize) {
    let mut i = start;
    let key = b"\"label\"";
    while i + key.len() <= b.len() && &b[i..i + key.len()] != key { i += 1; }
    i += key.len();
    // Find next quote (start of the value string)
    while i < b.len() && b[i] != b'"' { i += 1; }
    i += 1;
    // Read until closing quote
    let v_start = i;
    while i < b.len() && b[i] != b'"' { i += 1; }
    let label_str = &b[v_start..i];
    i += 1;
    let byte = if label_str == b"fraud" { 1 } else { 0 };
    (byte, i)
}

/// Random subsample of vectors for k-means training. We pick distinct
/// indices using the same PRNG the runtime uses so the build is
/// deterministic.
fn build_sample(vectors: &[f32], seed: u64) -> Vec<f32> {
    let mut rng = kmeans::Rng::new(seed);
    let mut chosen = std::collections::HashSet::with_capacity(KMEANS_SAMPLE_SIZE);
    let mut sample = Vec::with_capacity(KMEANS_SAMPLE_SIZE * DIM);
    while chosen.len() < KMEANS_SAMPLE_SIZE {
        let idx = rng.next_range(TOTAL_RECORDS);
        if chosen.insert(idx) {
            sample.extend_from_slice(&vectors[idx * DIM..(idx + 1) * DIM]);
        }
    }
    sample
}

fn write_index(
    path: &str,
    vectors: &[f32],
    labels: &[u8],
    centroids: &[f32],
    assignments: &[u32],
    counts: &[u32],
) {
    // Compute starting offsets per cluster
    let mut list_offsets = vec![0_u32; K_CENTROIDS];
    let mut running = 0_u32;
    for ci in 0..K_CENTROIDS {
        list_offsets[ci] = running;
        running += counts[ci];
    }
    assert_eq!(running as usize, TOTAL_RECORDS);

    // Reorder into cluster-grouped layout: vector_data (quantized) and
    // labels in the same order.
    let mut vector_data = vec![0_u8; TOTAL_RECORDS * DIM];
    let mut reordered_labels = vec![0_u8; TOTAL_RECORDS];
    let mut write_pos = list_offsets.clone();
    for i in 0..TOTAL_RECORDS {
        let ci = assignments[i] as usize;
        let pos = write_pos[ci] as usize;
        write_pos[ci] += 1;
        quantize_into(
            &vectors[i * DIM..(i + 1) * DIM],
            &mut vector_data[pos * DIM..(pos + 1) * DIM],
        );
        reordered_labels[pos] = labels[i];
    }

    // Emit the file
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent).expect("mkdir data/");
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("create ivf.bin"));
    f.write_all(&MAGIC.to_le_bytes()).unwrap();
    f.write_all(&(TOTAL_RECORDS as u32).to_le_bytes()).unwrap();
    f.write_all(&(K_CENTROIDS as u32).to_le_bytes()).unwrap();
    f.write_all(&(DIM as u32).to_le_bytes()).unwrap();
    // Centroids
    for c in centroids {
        f.write_all(&c.to_le_bytes()).unwrap();
    }
    // list_offsets
    for o in &list_offsets {
        f.write_all(&o.to_le_bytes()).unwrap();
    }
    // list_counts
    for c in counts {
        f.write_all(&c.to_le_bytes()).unwrap();
    }
    // Vector data (quantized)
    f.write_all(&vector_data).unwrap();
    // Labels (cluster-grouped)
    f.write_all(&reordered_labels).unwrap();
    f.flush().unwrap();
}
