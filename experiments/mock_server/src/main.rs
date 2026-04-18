use std::{collections::HashMap, io::Read};
use sha2::{Digest, Sha256};
use tiny_http::{Header, Response, Server};
use serde::{Deserialize, Serialize};

const N_VERT: usize = 20484;
const REGIONS: &[(&str, usize, usize)] = &[
    ("visual",     0,     3600),
    ("auditory",   3600,  6800),
    ("language",   6800,  10500),
    ("prefrontal", 10500, 14000),
    ("motor",      14000, 17200),
    ("parietal",   17200, 20484),
];

fn score_regions(text: &str) -> [f32; 6] {
    let t = text.to_lowercase();
    let count = |keywords: &[&str]| -> f32 {
        let n: usize = keywords.iter().map(|k| t.matches(k).count()).sum();
        (n as f32 / 3.0).min(1.0)
    };
    [
        // visual
        count(&["see","look","watch","color","red","blue","green","image","visual","bright","dark","shape","photograph","picture","light","shadow"]),
        // auditory
        count(&["hear","sound","loud","quiet","music","voice","noise","audio","ring","frequency","hz","silence","tone","pitch"]),
        // language / social / narrative / emotional
        count(&["said","told","felt","knew","thought","then","because","story","friend","love","fear","angry","sad","happy","cried","scared","laughed","shouted","married","died","born","called","replied","whispered","believed","wanted","hoped"]),
        // prefrontal / threat / novelty / decision
        count(&["emergency","urgent","warning","danger","threat","risk","must","evacuate","attack","collapse","fire","flood","shooting","alert","crisis","never","first","unknown","impossible","suddenly","unexpected","discovered","breakthrough","anomaly"]),
        // motor / action
        count(&["run","walk","move","jump","climb","push","pull","fell","drove","ran","stumbled","sprinted","crawled","swam","reached","grabbed"]),
        // parietal / spatial / quantitative
        count(&["north","south","east","west","left","right","above","below","meters","kilometers","map","position","location","distance","behind","front","beside","corner","level","floor","height","depth","width","celsius","degrees"]),
    ]
}

fn text_to_activations(text: &str, region_scores: &[f32; 6]) -> Vec<f32> {
    let mut acts = vec![0f32; N_VERT];
    let seed = text.as_bytes();
    let mut chunk = 0u32;
    let mut idx = 0;
    while idx < N_VERT {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.update(chunk.to_be_bytes());
        let hash = hasher.finalize();
        for j in (0..28).step_by(4) {
            if idx >= N_VERT { break; }
            let raw = u32::from_be_bytes([hash[j], hash[j+1], hash[j+2], hash[j+3]]);
            // map to [-1, 1]
            acts[idx] = (raw as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32;
            idx += 1;
        }
        chunk += 1;
    }
    // Apply region-specific scaling based on keyword presence
    for (ri, &(_, lo, hi)) in REGIONS.iter().enumerate() {
        let scale = 1.0 + region_scores[ri] * 2.5;
        let end = hi.min(N_VERT);
        for v in &mut acts[lo..end] {
            *v *= scale;
        }
    }
    acts
}

fn compute_region_stats(acts: &[f32]) -> (HashMap<String, serde_json::Value>, serde_json::Value) {
    let g_mean: f32 = acts.iter().sum::<f32>() / acts.len() as f32;
    let g_var: f32 = acts.iter().map(|x| (x - g_mean).powi(2)).sum::<f32>() / acts.len() as f32;
    let g_std = g_var.sqrt().max(1e-8);
    let g_min = acts.iter().cloned().fold(f32::INFINITY, f32::min);
    let g_max = acts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut region_stats = HashMap::new();
    for &(name, lo, hi) in REGIONS {
        let hi = hi.min(acts.len());
        let v = &acts[lo..hi];
        let rm: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let rs = (v.iter().map(|x| (x - rm).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
        let mut sorted: Vec<f32> = v.iter().cloned().map(f32::abs).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let peak = sorted[((sorted.len() as f32 * 0.95) as usize).min(sorted.len().saturating_sub(1))];
        let rel = (rm - g_mean) / g_std;
        region_stats.insert(name.to_string(), serde_json::json!({
            "mean": rm, "std": rs, "rel_activation": rel, "peak": peak,
            "n_vertices": hi - lo
        }));
    }

    let global = serde_json::json!({
        "global_mean": g_mean, "global_std": g_std,
        "global_min": g_min,  "global_max": g_max
    });
    (region_stats, global)
}

#[derive(Deserialize)]
struct PredictReq {
    text: Option<String>,
    #[allow(dead_code)]
    seq_len: Option<usize>,
}

fn main() {
    let addr = "127.0.0.1:8081";
    let server = Server::http(addr).expect("failed to bind 8081");
    eprintln!("[mock_server] listening at http://{addr}");
    eprintln!("[mock_server] demo mode — keyword-aware SHA256 activations");

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();

        if url == "/health" || url.starts_with("/health?") {
            let _ = request.respond(Response::from_string("ok"));
            continue;
        }

        if url.starts_with("/api/predict") && *request.method() == tiny_http::Method::Post {
            let mut body = String::new();
            request.as_reader().read_to_string(&mut body).unwrap_or(0);
            let req: PredictReq = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(_) => {
                    let _ = request.respond(Response::from_string(r#"{"error":"bad request"}"#).with_status_code(400));
                    continue;
                }
            };
            let text = req.text.unwrap_or_default();
            let scores = score_regions(&text);
            let acts = text_to_activations(&text, &scores);
            let vertex_acts: Vec<f32> = acts.clone();
            let (region_stats, global_stats) = compute_region_stats(&acts);

            let resp = serde_json::json!({
                "region_stats":  region_stats,
                "global_stats":  global_stats,
                "vertex_acts":   vertex_acts,
                "temporal_acts": [[global_stats["global_mean"], global_stats["global_mean"], global_stats["global_mean"], global_stats["global_mean"], global_stats["global_mean"], global_stats["global_mean"]]],
                "seq_len":       16,
                "modality":      "text",
                "elapsed_ms":    1.0,
                "demo_mode":     true
            });

            let json = resp.to_string();
            let header = Header::from_bytes(b"Content-Type".as_ref(), b"application/json".as_ref())
                .unwrap();
            let _ = request.respond(Response::from_string(json).with_header(header));
            continue;
        }

        let _ = request.respond(Response::from_string("not found").with_status_code(404));
    }
}
