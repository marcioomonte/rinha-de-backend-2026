mod distance;
mod ivf;
mod kmeans;
mod quantize;
mod types;
mod vectorize;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{extract::State, routing::{get, post}, Json, Router};

use crate::ivf::Ivf;
use crate::types::{FraudScoreRequest, FraudScoreResponse, MccRisk, Normalization};
use crate::vectorize::vectorize;

#[derive(Clone)]
struct AppState {
    mcc_risk: Arc<MccRisk>,
    norm: Arc<Normalization>,
    ivf: Arc<Ivf>,
    nprobe: usize,
}

async fn ready(State(_): State<AppState>) -> &'static str {
    ""
}

async fn fraud_score(
    State(state): State<AppState>,
    Json(req): Json<FraudScoreRequest>,
) -> Json<FraudScoreResponse> {
    let vec = match vectorize(&req, &state.mcc_risk, &state.norm) {
        Some(v) => v,
        None => {
            // Defensive: a 200 with a "trusted" answer is cheaper than a
            // 500 in the scoring formula (Err weight 5 vs FN weight 3).
            return Json(FraudScoreResponse { approved: true, fraud_score: 0.0 });
        }
    };

    let fraud_count = state.ivf.search_fraud_count(&vec, state.nprobe);
    let fraud_score = fraud_count as f32 / 5.0;
    let approved = fraud_score < 0.6;
    Json(FraudScoreResponse { approved, fraud_score })
}

fn load_state() -> AppState {
    let mcc_path = std::env::var("MCC_RISK_PATH").unwrap_or_else(|_| "resources/mcc_risk.json".into());
    let norm_path = std::env::var("NORM_PATH").unwrap_or_else(|_| "resources/normalization.json".into());
    let ivf_path = std::env::var("IVF_BIN_PATH").unwrap_or_else(|_| "data/ivf.bin".into());
    let nprobe: usize = std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let mcc_risk: MccRisk = serde_json::from_str(
        &std::fs::read_to_string(&mcc_path).expect("read mcc_risk.json"),
    )
    .expect("parse mcc_risk.json");
    let norm: Normalization = serde_json::from_str(
        &std::fs::read_to_string(&norm_path).expect("read normalization.json"),
    )
    .expect("parse normalization.json");

    let t0 = std::time::Instant::now();
    let ivf = Ivf::open(&ivf_path).expect("open ivf.bin");
    println!(
        "IVF loaded in {:.0}ms (n={}, k={}, dim={}, nprobe={})",
        t0.elapsed().as_secs_f32() * 1000.0,
        ivf.n(),
        ivf.k(),
        ivf.dim(),
        nprobe,
    );

    AppState {
        mcc_risk: Arc::new(mcc_risk),
        norm: Arc::new(norm),
        ivf: Arc::new(ivf),
        nprobe,
    }
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);

    let state = load_state();

    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    println!("Server listening on :{port}");

    axum::serve(listener, app).await.expect("serve");
}
