use std::{collections::HashMap, fs};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
struct Stimulus {
    id: String,
    content_type: String,
    source_type: String,
    language_structure: String,
    text: String,
}

fn load_corpus(path: &str) -> Vec<Stimulus> {
    let data = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read corpus at {path}: {e}"));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse corpus JSON: {e}"))
}

#[derive(Debug, Deserialize, Clone)]
struct RegionStat {
    mean: f32,
    rel_activation: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct GlobalStat {
    global_mean: f32,
    global_max: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct PredictResp {
    region_stats: HashMap<String, RegionStat>,
    global_stats: GlobalStat,
    vertex_acts: Vec<f32>,
    demo_mode: bool,
}

fn parse_predict_resp(json: &str) -> Result<PredictResp, serde_json::Error> {
    serde_json::from_str(json)
}

fn check_health(base_url: &str) -> bool {
    ureq::get(&format!("{base_url}/health"))
        .call()
        .map(|r| r.status() == 200)
        .unwrap_or(false)
}

fn predict_one(base_url: &str, text: &str) -> Result<PredictResp, String> {
    let body = serde_json::json!({ "text": text, "seq_len": 16 });
    let raw = ureq::post(&format!("{base_url}/api/predict"))
        .send_json(&body)
        .map_err(|e| e.to_string())?
        .into_string()
        .map_err(|e| e.to_string())?;
    parse_predict_resp(&raw).map_err(|e| e.to_string())
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_corpus_returns_correct_count() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, r#"[
            {{"id":"x1","content_type":"Factual","source_type":"enc","language_structure":"ls","text":"hello"}},
            {{"id":"x2","content_type":"Narrative","source_type":"enc","language_structure":"ls","text":"world"}}
        ]"#).unwrap();
        let stimuli = load_corpus(f.path().to_str().unwrap());
        assert_eq!(stimuli.len(), 2);
        assert_eq!(stimuli[0].id, "x1");
        assert_eq!(stimuli[1].content_type, "Narrative");
    }

    #[test]
    fn parse_predict_resp_parses_correctly() {
        let json = r#"{
            "region_stats": {
                "language": {"mean": 0.5, "std": 0.1, "rel_activation": 1.2, "peak": 0.9, "n_vertices": 3700},
                "prefrontal": {"mean": 0.3, "std": 0.08, "rel_activation": 0.4, "peak": 0.6, "n_vertices": 3500}
            },
            "global_stats": {"global_mean": 0.2, "global_std": 0.25, "global_min": -1.0, "global_max": 0.95},
            "vertex_acts": [0.1, 0.2, 0.3],
            "temporal_acts": [],
            "seq_len": 16,
            "modality": "text",
            "elapsed_ms": 42.0,
            "demo_mode": false
        }"#;
        let resp = parse_predict_resp(json).unwrap();
        assert!((resp.global_stats.global_mean - 0.2).abs() < 1e-5);
        assert!((resp.region_stats["language"].rel_activation - 1.2).abs() < 1e-5);
        assert_eq!(resp.demo_mode, false);
        assert_eq!(resp.vertex_acts.len(), 3);
    }

    #[test]
    fn check_health_returns_false_when_server_not_running() {
        // port 19991 is virtually guaranteed to be unused
        assert!(!check_health("http://localhost:19991"));
    }

    #[test]
    fn predict_one_returns_err_when_server_not_running() {
        let result = predict_one("http://localhost:19991", "hello world");
        assert!(result.is_err());
    }
}
