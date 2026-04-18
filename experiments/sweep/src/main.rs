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

#[derive(Debug, Clone, Serialize)]
struct SweepRecord {
    id: String,
    content_type: String,
    source_type: String,
    language_structure: String,
    demo_mode: bool,
    global_mean: f32,
    global_max: f32,
    visual_rel: f32,
    auditory_rel: f32,
    language_rel: f32,
    prefrontal_rel: f32,
    motor_rel: f32,
    parietal_rel: f32,
    vertex_acts: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct HeatmapData {
    content_types: Vec<String>,
    regions: Vec<String>,
    matrix: Vec<Vec<f32>>,
}

fn make_record(stimulus: &Stimulus, resp: &PredictResp) -> SweepRecord {
    let region = |name: &str| -> f32 {
        resp.region_stats.get(name).map(|r| r.rel_activation).unwrap_or(0.0)
    };
    SweepRecord {
        id: stimulus.id.clone(),
        content_type: stimulus.content_type.clone(),
        source_type: stimulus.source_type.clone(),
        language_structure: stimulus.language_structure.clone(),
        demo_mode: resp.demo_mode,
        global_mean: resp.global_stats.global_mean,
        global_max: resp.global_stats.global_max,
        visual_rel:     region("visual"),
        auditory_rel:   region("auditory"),
        language_rel:   region("language"),
        prefrontal_rel: region("prefrontal"),
        motor_rel:      region("motor"),
        parietal_rel:   region("parietal"),
        vertex_acts: resp.vertex_acts.clone(),
    }
}

fn rank_results(records: &[SweepRecord]) -> Vec<SweepRecord> {
    let mut sorted = records.to_vec();
    sorted.sort_by(|a, b| b.global_mean.partial_cmp(&a.global_mean).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

fn build_heatmap(records: &[SweepRecord]) -> HeatmapData {
    let regions = ["visual", "auditory", "language", "prefrontal", "motor", "parietal"];
    let mut types: Vec<String> = records.iter().map(|r| r.content_type.clone()).collect();
    types.sort();
    types.dedup();

    let matrix = types.iter().map(|ct| {
        let group: Vec<&SweepRecord> = records.iter().filter(|r| &r.content_type == ct).collect();
        let n = group.len() as f32;
        regions.iter().map(|region| {
            let sum: f32 = group.iter().map(|r| match *region {
                "visual"     => r.visual_rel,
                "auditory"   => r.auditory_rel,
                "language"   => r.language_rel,
                "prefrontal" => r.prefrontal_rel,
                "motor"      => r.motor_rel,
                "parietal"   => r.parietal_rel,
                _            => 0.0,
            }).sum();
            if n > 0.0 { sum / n } else { 0.0 }
        }).collect()
    }).collect();

    HeatmapData {
        content_types: types,
        regions: regions.iter().map(|s| s.to_string()).collect(),
        matrix,
    }
}

fn write_ranked_csv(ranked: &[SweepRecord], path: &str) {
    let header = "rank,id,content_type,source_type,language_structure,demo_mode,global_mean,global_max,visual_rel,auditory_rel,language_rel,prefrontal_rel,motor_rel,parietal_rel\n";
    let rows: String = ranked.iter().enumerate().map(|(i, r)| {
        format!("{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            i + 1, r.id, r.content_type, r.source_type, r.language_structure,
            r.demo_mode, r.global_mean, r.global_max,
            r.visual_rel, r.auditory_rel, r.language_rel,
            r.prefrontal_rel, r.motor_rel, r.parietal_rel)
    }).collect();
    fs::write(path, format!("{header}{rows}"))
        .unwrap_or_else(|e| panic!("Failed to write CSV to {path}: {e}"));
}

fn write_raw_json(records: &[SweepRecord], path: &str) {
    let json = serde_json::to_string_pretty(records)
        .unwrap_or_else(|e| panic!("Serialization failed: {e}"));
    fs::write(path, json).unwrap_or_else(|e| panic!("Failed to write JSON to {path}: {e}"));
}

fn write_heatmap_json(heatmap: &HeatmapData, path: &str) {
    let json = serde_json::to_string_pretty(heatmap)
        .unwrap_or_else(|e| panic!("Serialization failed: {e}"));
    fs::write(path, json).unwrap_or_else(|e| panic!("Failed to write heatmap to {path}: {e}"));
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

    fn make_test_record(id: &str, ct: &str, global_mean: f32, lang_rel: f32) -> SweepRecord {
        SweepRecord {
            id: id.to_string(),
            content_type: ct.to_string(),
            source_type: "test".to_string(),
            language_structure: "test".to_string(),
            demo_mode: true,
            global_mean,
            global_max: global_mean + 0.1,
            visual_rel: 0.0,
            auditory_rel: 0.0,
            language_rel: lang_rel,
            prefrontal_rel: 0.0,
            motor_rel: 0.0,
            parietal_rel: 0.0,
            vertex_acts: vec![],
        }
    }

    #[test]
    fn rank_results_sorts_by_global_mean_descending() {
        let records = vec![
            make_test_record("a", "Factual", 0.1, 0.0),
            make_test_record("b", "ThreatSafety", 0.9, 0.0),
            make_test_record("c", "Narrative", 0.5, 0.0),
        ];
        let ranked = rank_results(&records);
        assert_eq!(ranked[0].id, "b");
        assert_eq!(ranked[1].id, "c");
        assert_eq!(ranked[2].id, "a");
    }

    #[test]
    fn build_heatmap_produces_correct_dimensions() {
        let records = vec![
            make_test_record("a", "Factual", 0.1, 0.3),
            make_test_record("b", "Narrative", 0.5, 0.8),
            make_test_record("c", "Factual", 0.2, 0.4),
        ];
        let heatmap = build_heatmap(&records);
        // 2 unique content types
        assert_eq!(heatmap.content_types.len(), 2);
        assert_eq!(heatmap.regions.len(), 6);
        assert_eq!(heatmap.matrix.len(), 2);
        assert_eq!(heatmap.matrix[0].len(), 6);
        // Factual language_rel mean = (0.3 + 0.4) / 2 = 0.35
        let factual_idx = heatmap.content_types.iter().position(|x| x == "Factual").unwrap();
        let lang_idx = heatmap.regions.iter().position(|x| x == "language").unwrap();
        assert!((heatmap.matrix[factual_idx][lang_idx] - 0.35).abs() < 1e-4);
    }

    #[test]
    fn write_ranked_csv_produces_correct_header() {
        let records = vec![make_test_record("x1", "Factual", 0.5, 0.2)];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        write_ranked_csv(&records, path);
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.starts_with("rank,id,content_type,source_type,language_structure,demo_mode,global_mean,global_max,visual_rel,auditory_rel,language_rel,prefrontal_rel,motor_rel,parietal_rel"));
        assert!(content.contains("1,x1,Factual"));
    }
}
