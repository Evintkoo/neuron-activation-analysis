# Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based 3D brain heatmap viewer and analysis dashboard that reads activation data from the core pipeline and renders it as interactive visualizations.

**Architecture:** A Rust `viz/builder` crate re-runs the activation pipeline, exports a single `viz_data.json` file, then serves a static web app at `localhost:8080`. The web app uses Three.js for 3D brain mesh rendering with per-vertex coolwarm coloring, and Chart.js for the analysis dashboard. The server is a minimal `tiny_http` binary — no Python, no npm. All DOM manipulation uses `createElement`/`replaceChildren` — no `innerHTML`.

**Tech Stack:** Rust (`viz/builder` crate: `experiments` + `analysis` deps, `tiny_http 0.12`, `flate2 1`, `ndarray 0.15`), Three.js r158 (CDN), Chart.js 4 (CDN), OBJ mesh from `tribe-playground/brain.obj.gz`

---

## File Map

| File | Role |
|------|------|
| `viz/builder/Cargo.toml` | New binary crate |
| `viz/builder/src/main.rs` | CLI entry: build data, serve HTTP |
| `viz/builder/src/export.rs` | `VizData` struct + `build_viz_data()` |
| `viz/brain_heatmap/index.html` | Brain viewer shell |
| `viz/brain_heatmap/dashboard.html` | Analysis dashboard shell |
| `viz/brain_heatmap/js/colors.js` | Coolwarm colormap |
| `viz/brain_heatmap/js/brain.js` | Three.js mesh loader + vertex coloring, exports `init()` |
| `viz/brain_heatmap/js/controls.js` | Content type toggle + layer slider |
| `viz/brain_heatmap/js/overlay.js` | Theory overlay sidebar |
| `viz/brain_heatmap/js/app.js` | Single entry point that composes all modules |
| `viz/brain_heatmap/js/dashboard.js` | Chart.js theory bars + contrastive delta viewer |
| `Cargo.toml` (workspace root) | Add `"viz/builder"` to members |

---

### Task 1: viz/builder crate scaffold

**Files:**
- Create: `viz/builder/Cargo.toml`
- Create: `viz/builder/src/main.rs` (stub)
- Create: `viz/builder/src/export.rs` (stub)
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Write the failing test**

```rust
// viz/builder/src/export.rs
#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_compiles() {
        assert!(true);
    }
}
```

- [ ] **Step 2: Create `viz/builder/Cargo.toml`**

```toml
[package]
name = "viz_builder"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "viz_builder"
path = "src/main.rs"

[dependencies]
experiments = { path = "../../experiments" }
analysis = { path = "../../analysis" }
ndarray = "0.15"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tiny_http = "0.12"
flate2 = "1"
```

- [ ] **Step 3: Create `viz/builder/src/main.rs` stub**

```rust
mod export;

fn main() {
    println!("viz_builder");
}
```

- [ ] **Step 4: Add `"viz/builder"` to workspace root `Cargo.toml`**

Change:
```toml
members = ["experiments", "analysis", "runner"]
```
To:
```toml
members = ["experiments", "analysis", "runner", "viz/builder"]
```

- [ ] **Step 5: Run test to verify it compiles**

```bash
cargo test -p viz_builder
```
Expected: `test placeholder_compiles ... ok`

- [ ] **Step 6: Commit**

```bash
git add viz/builder/Cargo.toml viz/builder/src/main.rs viz/builder/src/export.rs Cargo.toml
git commit -m "feat: add viz/builder crate scaffold"
```

---

### Task 2: VizData struct and build_viz_data()

**Files:**
- Modify: `viz/builder/src/export.rs`

`build_viz_data()` re-runs the activation pipeline (same parameters as the runner: 50 stimuli per type, `MockTribeModel` 128 neurons, 64 input dim) and returns a `VizData` struct with per-type, per-layer mean activation vectors normalized to `[0, 1]`.

- [ ] **Step 1: Write the failing tests**

```rust
// viz/builder/src/export.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_viz_data_returns_all_13_types() {
        let data = build_viz_data();
        assert_eq!(data.activation_maps.len(), 13);
    }

    #[test]
    fn activation_values_are_normalized() {
        let data = build_viz_data();
        for (_, entry) in &data.activation_maps {
            for v in entry.early.iter().chain(entry.mid.iter()).chain(entry.late.iter()) {
                assert!(*v >= 0.0 && *v <= 1.0 + 1e-9, "value out of [0,1]: {}", v);
            }
        }
    }

    #[test]
    fn serializes_to_valid_json() {
        let data = build_viz_data();
        let json = serde_json::to_string(&data).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["activation_maps"].is_object());
        assert!(parsed["theory_fit"].is_object());
        assert!(parsed["n_neurons"].is_number());
        assert!(parsed["silhouette_score"].is_number());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p viz_builder
```
Expected: FAIL — `build_viz_data` not defined

- [ ] **Step 3: Implement `VizData` and `build_viz_data()`**

```rust
// viz/builder/src/export.rs
use std::collections::HashMap;
use ndarray::Array2;
use experiments::{
    ActivationExtractor, ActivationFingerprint, ContentType,
    DctScorer, FepScorer, GwtScorer, IitScorer, KMeansClusterer,
    MockTribeModel, PcaReducer, SyntheticGenerator, TheoryFitReport,
};
use analysis::{one_way_anova, ContrastiveDelta};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct LayerVec {
    pub early: Vec<f64>,
    pub mid: Vec<f64>,
    pub late: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct ContrastiveEntry {
    pub pair: String,
    pub l2_norm: f64,
    pub early_delta: Vec<f64>,
    pub mid_delta: Vec<f64>,
    pub late_delta: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct VizData {
    pub activation_maps: HashMap<String, LayerVec>,
    pub theory_fit: TheoryFitReport,
    pub contrastive_deltas: Vec<ContrastiveEntry>,
    pub anova_p_value: f64,
    pub silhouette_score: f64,
    pub n_neurons: usize,
}

pub fn build_viz_data() -> VizData {
    let generator = SyntheticGenerator::new(42, 64);
    let stimuli = generator.generate_all(50);
    let model = MockTribeModel::new(128, 64);
    let extractor = ActivationExtractor::new(model);
    let records = extractor.extract_batch(&stimuli);
    let n_neurons = records[0].early.len();

    // per-type mean activation, normalized to [0,1] per layer
    let mut activation_maps = HashMap::new();
    for &ct in ContentType::all().iter() {
        let ct_records: Vec<_> = records.iter().filter(|r| r.content_type == ct).collect();
        let n = ct_records.len() as f64;
        let mut early = vec![0.0f64; n_neurons];
        let mut mid = vec![0.0f64; n_neurons];
        let mut late = vec![0.0f64; n_neurons];
        for rec in &ct_records {
            for i in 0..n_neurons {
                early[i] += rec.early[i].abs();
                mid[i] += rec.mid[i].abs();
                late[i] += rec.late[i].abs();
            }
        }
        for i in 0..n_neurons {
            early[i] /= n;
            mid[i] /= n;
            late[i] /= n;
        }
        normalize(&mut early);
        normalize(&mut mid);
        normalize(&mut late);
        activation_maps.insert(ct.label().to_string(), LayerVec { early, mid, late });
    }

    // theory fit via mid-layer PCA + k-means fingerprints
    let n_stimuli = records.len();
    let mut mid_mat = Array2::<f64>::zeros((n_stimuli, n_neurons));
    for (i, rec) in records.iter().enumerate() {
        for j in 0..n_neurons {
            mid_mat[[i, j]] = rec.mid[j];
        }
    }
    let reducer = PcaReducer::fit(&mid_mat, 8);
    let reduced = reducer.transform(&mid_mat);
    let cluster = KMeansClusterer::run(&reduced, 13, 42);
    let fingerprints = ActivationFingerprint::from_clusters(
        &reduced, &cluster.labels, &ContentType::all(),
    );
    let theory_fit = TheoryFitReport {
        dct_score: DctScorer::score(&fingerprints),
        gwt_score: GwtScorer::score(&records, 0.5, 3),
        fep_score: FepScorer::score(&records),
        iit_score: IitScorer::score(&records),
    };

    // B3 ThreatSafety vs B4 Novelty contrastive delta
    let threat: Vec<_> = records.iter()
        .filter(|r| r.content_type == ContentType::ThreatSafety).collect();
    let novelty: Vec<_> = records.iter()
        .filter(|r| r.content_type == ContentType::Novelty).collect();
    let delta = ContrastiveDelta::compute(threat[0], novelty[0]);
    let contrastive_deltas = vec![ContrastiveEntry {
        pair: "ThreatSafety vs Novelty".to_string(),
        l2_norm: delta.l2_norm(),
        early_delta: delta.early_delta,
        mid_delta: delta.mid_delta,
        late_delta: delta.late_delta,
    }];

    // ANOVA across content type groups
    let groups: Vec<Vec<f64>> = ContentType::all().iter().map(|&ct| {
        records.iter()
            .filter(|r| r.content_type == ct)
            .map(|r| r.early.iter().map(|v| v.abs()).sum::<f64>() / n_neurons as f64)
            .collect()
    }).collect();
    let anova_p_value = one_way_anova(&groups).p_value;

    VizData { activation_maps, theory_fit, contrastive_deltas, anova_p_value,
              silhouette_score: cluster.silhouette_score, n_neurons }
}

fn normalize(v: &mut Vec<f64>) {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = (max - min).max(1e-10);
    for x in v.iter_mut() {
        *x = (*x - min) / range;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p viz_builder
```
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add viz/builder/src/export.rs viz/builder/Cargo.toml
git commit -m "feat: add VizData struct and build_viz_data pipeline"
```

---

### Task 3: main.rs — write JSON, decompress brain.obj, serve HTTP

**Files:**
- Modify: `viz/builder/src/main.rs`

The binary writes `viz/brain_heatmap/data/viz_data.json`, attempts to decompress `brain.obj.gz` from sibling `tribe-playground/` directory, then starts a `tiny_http` server at `localhost:8080`.

- [ ] **Step 1: Write the failing tests**

```rust
// viz/builder/src/main.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mime_type_for_html() {
        assert_eq!(mime_for_ext("html"), "text/html");
    }

    #[test]
    fn mime_type_for_js() {
        assert_eq!(mime_for_ext("js"), "application/javascript");
    }

    #[test]
    fn mime_type_for_obj() {
        assert_eq!(mime_for_ext("obj"), "text/plain");
    }

    #[test]
    fn mime_type_unknown_defaults_to_octet() {
        assert_eq!(mime_for_ext("xyz"), "application/octet-stream");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p viz_builder
```
Expected: FAIL — `mime_for_ext` not defined

- [ ] **Step 3: Implement `main.rs`**

```rust
// viz/builder/src/main.rs
mod export;

use std::{fs, io::Read, path::{Path, PathBuf}};
use tiny_http::{Header, Response, Server};

fn main() {
    let viz_dir = Path::new("viz/brain_heatmap");
    let data_dir = viz_dir.join("data");
    let assets_dir = viz_dir.join("assets");
    fs::create_dir_all(&data_dir).expect("failed to create viz/brain_heatmap/data");
    fs::create_dir_all(&assets_dir).expect("failed to create viz/brain_heatmap/assets");

    println!("Building visualization data...");
    let viz_data = export::build_viz_data();
    let json = serde_json::to_string_pretty(&viz_data).expect("serialization failed");
    let json_path = data_dir.join("viz_data.json");
    fs::write(&json_path, &json).expect("failed to write viz_data.json");
    println!("  Wrote {}", json_path.display());

    let brain_dest = assets_dir.join("brain.obj");
    if brain_dest.exists() {
        println!("  brain.obj already present");
    } else {
        let gz_candidates = [
            PathBuf::from("../tribe-playground/brain.obj.gz"),
            assets_dir.join("brain.obj.gz"),
        ];
        let found = gz_candidates.iter().find(|p| p.exists());
        match found {
            Some(gz_path) => match decompress_gz(gz_path, &brain_dest) {
                Ok(_) => println!("  Decompressed brain.obj from {}", gz_path.display()),
                Err(e) => println!("  Decompression failed ({}), using sphere fallback", e),
            },
            None => println!("  brain.obj.gz not found — JS will use sphere fallback"),
        }
    }

    let addr = "127.0.0.1:8080";
    let server = Server::http(addr).expect("failed to start HTTP server");
    println!("\nServing at http://{}", addr);
    println!("Open http://{}/index.html in your browser\nPress Ctrl+C to stop.\n", addr);

    for request in server.incoming_requests() {
        let url = request.url().to_string();
        let rel = url.trim_start_matches('/');
        let rel = if rel.is_empty() { "index.html" } else { rel };
        let file_path: PathBuf = viz_dir.join(rel);

        let response = if file_path.is_file() {
            match fs::read(&file_path) {
                Ok(bytes) => {
                    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    let mime = mime_for_ext(ext);
                    let header = Header::from_bytes(&b"Content-Type"[..], mime.as_bytes())
                        .expect("invalid header");
                    Response::from_data(bytes).with_header(header)
                }
                Err(_) => Response::from_string("500").with_status_code(500),
            }
        } else {
            Response::from_string("404 Not Found").with_status_code(404)
        };

        let _ = request.respond(response);
    }
}

fn decompress_gz(src: &Path, dest: &Path) -> std::io::Result<()> {
    let file = fs::File::open(src)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder.read_to_end(&mut bytes)?;
    fs::write(dest, bytes)
}

fn mime_for_ext(ext: &str) -> &'static str {
    match ext {
        "html" => "text/html",
        "js" => "application/javascript",
        "css" => "text/css",
        "json" => "application/json",
        "obj" => "text/plain",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mime_type_for_html() {
        assert_eq!(mime_for_ext("html"), "text/html");
    }

    #[test]
    fn mime_type_for_js() {
        assert_eq!(mime_for_ext("js"), "application/javascript");
    }

    #[test]
    fn mime_type_for_obj() {
        assert_eq!(mime_for_ext("obj"), "text/plain");
    }

    #[test]
    fn mime_type_unknown_defaults_to_octet() {
        assert_eq!(mime_for_ext("xyz"), "application/octet-stream");
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p viz_builder
```
Expected: 7 tests PASS (3 from export + 4 from main)

- [ ] **Step 5: Commit**

```bash
git add viz/builder/src/main.rs
git commit -m "feat: add viz_builder HTTP server, JSON export, and brain.obj decompression"
```

---

### Task 4: HTML shells

**Files:**
- Create: `viz/brain_heatmap/index.html`
- Create: `viz/brain_heatmap/dashboard.html`

Both pages import Three.js/Chart.js from CDN and use a single `<script type="module">` entry point.

- [ ] **Step 1: Create `viz/brain_heatmap/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Activation Cartography — Brain Viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d0d; color: #e0e0e0;
               font-family: system-ui, sans-serif;
               display: flex; height: 100vh; overflow: hidden; }
        #sidebar { width: 280px; min-width: 280px; padding: 20px;
                   background: #1a1a2e; overflow-y: auto;
                   display: flex; flex-direction: column; gap: 16px; }
        #canvas-container { flex: 1; position: relative; }
        canvas { width: 100% !important; height: 100% !important; display: block; }
        h1 { font-size: 14px; color: #7c9cbf; letter-spacing: 1px;
             text-transform: uppercase; }
        h2 { font-size: 12px; color: #5a7a9a; text-transform: uppercase;
             letter-spacing: 0.5px; margin-bottom: 6px; }
        label { font-size: 13px; color: #aaa; display: block; margin-bottom: 4px; }
        select, input[type=range] { width: 100%; background: #252545;
            border: 1px solid #333; color: #e0e0e0;
            padding: 6px; border-radius: 4px; }
        .section { border-top: 1px solid #2a2a4a; padding-top: 14px; }
        #theory-panel { font-size: 12px; }
        .theory-row { display: flex; justify-content: space-between;
                      padding: 3px 0; border-bottom: 1px solid #1e1e3a; }
        .theory-name { color: #7c9cbf; }
        .theory-score { font-family: monospace; }
        .theory-winner { color: #f0c040; font-weight: bold; }
        #legend { display: flex; align-items: center; gap: 8px; font-size: 11px; }
        #legend-bar { height: 12px; flex: 1; border-radius: 3px;
            background: linear-gradient(to right,
                #054f8f, #74add1, #dcdcdc, #d9340d, #680016); }
        #status { font-size: 11px; color: #5a5a8a; margin-top: auto; }
        a { color: #7c9cbf; font-size: 12px; text-decoration: none; }
        a:hover { color: #9abcdf; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Activation Cartography</h1>
        <div class="section">
            <h2>Content Type</h2>
            <label for="type-select">Select type</label>
            <select id="type-select"></select>
        </div>
        <div class="section">
            <h2>Processing Layer</h2>
            <label for="layer-slider" id="layer-label">Mid (layer 1)</label>
            <input type="range" id="layer-slider" min="0" max="2" value="1" step="1">
        </div>
        <div class="section">
            <h2>Theory Overlay</h2>
            <label>
                <input type="checkbox" id="overlay-toggle"> Show theory scores
            </label>
            <div id="theory-panel"></div>
        </div>
        <div class="section">
            <h2>Color Scale</h2>
            <div id="legend">
                <span>Low</span>
                <div id="legend-bar"></div>
                <span>High</span>
            </div>
        </div>
        <div id="status">Loading data...</div>
        <a href="dashboard.html">Analysis Dashboard</a>
    </div>
    <div id="canvas-container">
        <canvas id="three-canvas"></canvas>
    </div>
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/"
        }
    }
    </script>
    <script type="module" src="js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `viz/brain_heatmap/dashboard.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Activation Cartography — Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0d0d0d; color: #e0e0e0;
               font-family: system-ui, sans-serif;
               padding: 32px; max-width: 960px; margin: 0 auto; }
        h1 { font-size: 18px; color: #7c9cbf; margin-bottom: 24px;
             text-transform: uppercase; letter-spacing: 1px; }
        h2 { font-size: 13px; color: #5a7a9a; text-transform: uppercase;
             letter-spacing: 0.5px; margin-bottom: 12px; margin-top: 32px; }
        .card { background: #1a1a2e; border-radius: 8px;
                padding: 20px; margin-bottom: 20px; }
        .chart-wrap { position: relative; height: 280px; }
        .stat-row { display: flex; justify-content: space-between;
                    padding: 6px 0; border-bottom: 1px solid #1e1e3a;
                    font-size: 13px; }
        .stat-label { color: #aaa; }
        .stat-value { font-family: monospace; color: #e0e0e0; }
        .pass { color: #4caf50; }
        .fail { color: #ef5350; }
        #delta-label { font-size: 12px; color: #7c9cbf;
                       text-align: center; margin-top: 8px; }
        #delta-slider { width: 100%; margin: 8px 0; }
        #delta-display { display: flex; flex-wrap: wrap; gap: 2px;
                         font-size: 11px; font-family: monospace;
                         margin-top: 8px; }
        #delta-display span { padding: 2px 4px; border-radius: 2px; }
        a { color: #7c9cbf; font-size: 12px; text-decoration: none; }
        a:hover { color: #9abcdf; }
    </style>
</head>
<body>
    <h1>Activation Cartography — Dashboard</h1>
    <a href="index.html">Brain Viewer</a>

    <h2>Theory Fit Scores</h2>
    <div class="card">
        <div class="chart-wrap">
            <canvas id="theory-chart"></canvas>
        </div>
    </div>

    <h2>Statistical Summary</h2>
    <div class="card" id="stats-panel"></div>

    <h2>Contrastive Delta: ThreatSafety vs Novelty</h2>
    <div class="card">
        <label for="delta-slider">Layer</label>
        <input type="range" id="delta-slider" min="0" max="2" value="0" step="1">
        <div id="delta-label">Early layer delta</div>
        <div id="delta-display"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script type="module" src="js/dashboard.js"></script>
</body>
</html>
```

- [ ] **Step 3: Build and confirm no errors**

```bash
cargo build -p viz_builder && echo "ok"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add viz/brain_heatmap/index.html viz/brain_heatmap/dashboard.html
git commit -m "feat: add HTML shells for brain viewer and dashboard"
```

---

### Task 5: Coolwarm colormap and app.js entry point

**Files:**
- Create: `viz/brain_heatmap/js/colors.js`
- Create: `viz/brain_heatmap/js/app.js` (stub — just logs "app loaded")

`colors.js` exports `coolwarmRgb(t)` → `[r, g, b]` each in `[0, 1]`. `app.js` will be the single module entry point that orchestrates brain, controls, and overlay.

- [ ] **Step 1: Create `viz/brain_heatmap/js/colors.js`**

```js
// viz/brain_heatmap/js/colors.js
const STOPS = [
    [0.00, 0.019, 0.188, 0.380],
    [0.25, 0.129, 0.400, 0.674],
    [0.50, 0.865, 0.865, 0.865],
    [0.75, 0.705, 0.016, 0.149],
    [1.00, 0.404, 0.000, 0.122],
];

export function coolwarmRgb(t) {
    t = Math.max(0, Math.min(1, t));
    let i = STOPS.findIndex(s => s[0] > t) - 1;
    if (i < 0) i = 0;
    if (i >= STOPS.length - 1) i = STOPS.length - 2;
    const [t0, r0, g0, b0] = STOPS[i];
    const [t1, r1, g1, b1] = STOPS[i + 1];
    const a = (t - t0) / (t1 - t0);
    return [r0 + a * (r1 - r0), g0 + a * (g1 - g0), b0 + a * (b1 - b0)];
}
```

- [ ] **Step 2: Create `viz/brain_heatmap/js/app.js` stub**

```js
// viz/brain_heatmap/js/app.js
console.log('app loaded');
```

- [ ] **Step 3: Commit**

```bash
git add viz/brain_heatmap/js/colors.js viz/brain_heatmap/js/app.js
git commit -m "feat: add coolwarm colormap and app.js stub"
```

---

### Task 6: Three.js brain mesh + vertex coloring

**Files:**
- Create: `viz/brain_heatmap/js/brain.js`

`brain.js` exports `init(vizData)` which sets up the Three.js renderer, loads `assets/brain.obj` (falls back to sphere), partitions vertices into 128 neuron buckets, and applies coolwarm vertex colors. Also exports `setType(type)` and `setLayer(layer)` for UI controls.

- [ ] **Step 1: Create `viz/brain_heatmap/js/brain.js`**

```js
// viz/brain_heatmap/js/brain.js
import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { coolwarmRgb } from './colors.js';

const canvas = document.getElementById('three-canvas');
const container = document.getElementById('canvas-container');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(container.clientWidth, container.clientHeight);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0d0d);

const camera = new THREE.PerspectiveCamera(
    45, container.clientWidth / container.clientHeight, 0.1, 1000
);
camera.position.set(0, 0, 3);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 2, 3);
scene.add(dir);

let brainMesh = null;
let vizData = null;
let currentType = null;
let currentLayer = 'mid';

export function init(data) {
    vizData = data;
    const types = Object.keys(data.activation_maps).sort();
    currentType = types[0];

    const sel = document.getElementById('type-select');
    types.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t;
        opt.textContent = t;
        sel.appendChild(opt);
    });
    sel.value = currentType;

    loadMesh();
    animate();
    window.addEventListener('resize', onResize);
    document.getElementById('status').textContent = 'Ready';
}

export function setType(t) {
    currentType = t;
    applyColors();
}

export function setLayer(l) {
    currentLayer = l;
    applyColors();
}

function loadMesh() {
    new OBJLoader().load(
        'assets/brain.obj',
        obj => {
            obj.traverse(child => {
                if (child.isMesh) buildMesh(child.geometry.toNonIndexed());
            });
        },
        undefined,
        () => buildMesh(new THREE.IcosahedronGeometry(1, 6).toNonIndexed())
    );
}

function buildMesh(geometry) {
    const mat = new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 30 });
    brainMesh = new THREE.Mesh(geometry, mat);
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    brainMesh.position.sub(center);
    const size = geometry.boundingBox.getSize(new THREE.Vector3()).length();
    brainMesh.scale.setScalar(2.5 / size);
    scene.add(brainMesh);
    applyColors();
}

function applyColors() {
    if (!brainMesh || !vizData || !currentType) return;
    const entry = vizData.activation_maps[currentType];
    if (!entry) return;
    const layerData = entry[currentLayer];
    const nNeurons = layerData.length;
    const positions = brainMesh.geometry.attributes.position;
    const nVerts = positions.count;
    const groupSize = Math.ceil(nVerts / nNeurons);
    const colors = new Float32Array(nVerts * 3);
    for (let v = 0; v < nVerts; v++) {
        const idx = Math.min(Math.floor(v / groupSize), nNeurons - 1);
        const [r, g, b] = coolwarmRgb(layerData[idx]);
        colors[v * 3]     = r;
        colors[v * 3 + 1] = g;
        colors[v * 3 + 2] = b;
    }
    brainMesh.geometry.setAttribute(
        'color', new THREE.BufferAttribute(colors, 3)
    );
    brainMesh.geometry.attributes.color.needsUpdate = true;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onResize() {
    const w = container.clientWidth, h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}
```

- [ ] **Step 2: Commit**

```bash
git add viz/brain_heatmap/js/brain.js
git commit -m "feat: add Three.js brain mesh loader with coolwarm vertex coloring"
```

---

### Task 7: Controls + overlay modules + app.js wiring

**Files:**
- Create: `viz/brain_heatmap/js/controls.js`
- Create: `viz/brain_heatmap/js/overlay.js`
- Modify: `viz/brain_heatmap/js/app.js`

`controls.js` wires the layer slider and type dropdown to `setType`/`setLayer`. `overlay.js` renders theory scores into the sidebar panel using safe DOM methods. `app.js` fetches data and initializes all three modules.

- [ ] **Step 1: Create `viz/brain_heatmap/js/controls.js`**

```js
// viz/brain_heatmap/js/controls.js
const LAYER_NAMES = ['early', 'mid', 'late'];
const LAYER_LABELS = ['Early (low-level)', 'Mid (semantic)', 'Late (output)'];

export function initControls(setType, setLayer) {
    document.getElementById('type-select').addEventListener('change', e => {
        setType(e.target.value);
    });

    const slider = document.getElementById('layer-slider');
    const layerLabel = document.getElementById('layer-label');
    slider.addEventListener('input', e => {
        const idx = parseInt(e.target.value, 10);
        layerLabel.textContent = LAYER_LABELS[idx];
        setLayer(LAYER_NAMES[idx]);
    });
}
```

- [ ] **Step 2: Create `viz/brain_heatmap/js/overlay.js`**

The theory panel is built with `createElement` — no `innerHTML`.

```js
// viz/brain_heatmap/js/overlay.js
export function initOverlay(vizData) {
    const checkbox = document.getElementById('overlay-toggle');
    checkbox.addEventListener('change', () => renderPanel(vizData));
    renderPanel(vizData);
}

function renderPanel(vizData) {
    const { dct_score, gwt_score, fep_score, iit_score } = vizData.theory_fit;
    const scores = { DCT: dct_score, GWT: gwt_score, FEP: fep_score, IIT: iit_score };
    const winner = Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];

    const panel = document.getElementById('theory-panel');
    const rows = Object.entries(scores).map(([name, score]) => {
        const row = document.createElement('div');
        row.className = 'theory-row';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'theory-name' + (name === winner ? ' theory-winner' : '');
        nameSpan.textContent = name === winner ? name + ' \u2605' : name;

        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'theory-score' + (name === winner ? ' theory-winner' : '');
        scoreSpan.textContent = score.toFixed(3);

        row.appendChild(nameSpan);
        row.appendChild(scoreSpan);
        return row;
    });

    panel.replaceChildren(...rows);
}
```

- [ ] **Step 3: Update `viz/brain_heatmap/js/app.js`**

```js
// viz/brain_heatmap/js/app.js
import { init, setType, setLayer } from './brain.js';
import { initControls } from './controls.js';
import { initOverlay } from './overlay.js';

fetch('data/viz_data.json')
    .then(r => r.json())
    .then(data => {
        init(data);
        initControls(setType, setLayer);
        initOverlay(data);
    })
    .catch(err => {
        document.getElementById('status').textContent = 'Error: ' + err.message;
    });
```

- [ ] **Step 4: Manual verification**

```bash
cargo run --bin viz_builder
```
Open `http://localhost:8080/index.html`. Verify:
- 3D mesh or sphere renders with coolwarm colors
- Content type dropdown updates vertex colors
- Layer slider updates label text and re-colors mesh
- Theory panel lists DCT/GWT/FEP/IIT with winner in gold

- [ ] **Step 5: Commit**

```bash
git add viz/brain_heatmap/js/controls.js viz/brain_heatmap/js/overlay.js viz/brain_heatmap/js/app.js
git commit -m "feat: wire content type, layer slider, and theory overlay controls"
```

---

### Task 8: Analysis dashboard — theory chart + stats + contrastive delta

**Files:**
- Create: `viz/brain_heatmap/js/dashboard.js`

All DOM manipulation uses `createElement`/`replaceChildren`. The contrastive delta viewer colors neuron spans using `element.style.background` (safe — values come from our own local JSON).

- [ ] **Step 1: Create `viz/brain_heatmap/js/dashboard.js`**

```js
// viz/brain_heatmap/js/dashboard.js
import { coolwarmRgb } from './colors.js';

fetch('data/viz_data.json')
    .then(r => r.json())
    .then(data => {
        renderTheoryChart(data.theory_fit);
        renderStats(data);
        renderContrastiveDelta(data.contrastive_deltas);
    });

function renderTheoryChart(fit) {
    const { dct_score, gwt_score, fep_score, iit_score } = fit;
    const labels = ['DCT', 'GWT', 'FEP', 'IIT'];
    const scores = [dct_score, gwt_score, fep_score, iit_score];
    const maxIdx = scores.indexOf(Math.max(...scores));
    const bgColors = scores.map((_, i) =>
        i === maxIdx ? 'rgba(240,192,64,0.85)' : 'rgba(124,156,191,0.6)'
    );
    new Chart(document.getElementById('theory-chart'), {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Theory Fit', data: scores,
            backgroundColor: bgColors, borderWidth: 1 }] },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y: { min: 0, max: 1,
                     ticks: { color: '#aaa' }, grid: { color: '#1e1e3a' } },
                x: { ticks: { color: '#e0e0e0' }, grid: { color: 'transparent' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function renderStats(data) {
    const { dct_score, gwt_score, fep_score, iit_score } = data.theory_fit;
    const allScores = [dct_score, gwt_score, fep_score, iit_score];
    const sorted = [...allScores].sort((a, b) => b - a);
    const margin = sorted[0] - sorted[1];
    const winnerName = ['DCT', 'GWT', 'FEP', 'IIT'][allScores.indexOf(sorted[0])];
    const marginOk = margin >= 0.1;
    const silh = typeof data.silhouette_score === 'number' ? data.silhouette_score : null;
    const silhOk = silh !== null && silh > 0.6;

    const rows = [
        ['Winner', winnerName + ' (margin: ' + margin.toFixed(3) + ')', marginOk],
        ['Silhouette', silh !== null ? silh.toFixed(4) : 'N/A', silhOk],
        ['ANOVA p-value', typeof data.anova_p_value === 'number'
            ? data.anova_p_value.toFixed(6) : 'N/A', null],
        ['Margin \u2265 0.1', marginOk ? 'PASS' : 'FAIL', marginOk],
    ];

    const panel = document.getElementById('stats-panel');
    panel.replaceChildren(...rows.map(([label, val, ok]) => {
        const row = document.createElement('div');
        row.className = 'stat-row';
        const lEl = document.createElement('span');
        lEl.className = 'stat-label';
        lEl.textContent = label;
        const vEl = document.createElement('span');
        vEl.className = 'stat-value' +
            (ok === true ? ' pass' : ok === false ? ' fail' : '');
        vEl.textContent = val;
        row.appendChild(lEl);
        row.appendChild(vEl);
        return row;
    }));
}

function renderContrastiveDelta(deltas) {
    const display = document.getElementById('delta-display');
    const label = document.getElementById('delta-label');
    if (!deltas || deltas.length === 0) {
        label.textContent = 'No contrastive data available';
        return;
    }
    const entry = deltas[0];
    const layers = [entry.early_delta, entry.mid_delta, entry.late_delta];
    const layerNames = ['Early layer', 'Mid layer', 'Late layer'];

    function render(idx) {
        const delta = layers[idx];
        label.textContent = layerNames[idx] + ' delta — L2: ' + entry.l2_norm.toFixed(4);
        const shown = delta.slice(0, 32);
        const maxAbs = Math.max(...shown.map(Math.abs), 1e-10);
        const spans = shown.map((v, i) => {
            const t = (v / maxAbs + 1) / 2;
            const [r, g, b] = coolwarmRgb(t);
            const span = document.createElement('span');
            span.style.background =
                'rgb(' + [r, g, b].map(x => Math.round(x * 255)).join(',') + ')';
            span.style.color = (t > 0.35 && t < 0.65) ? '#333' : '#fff';
            span.textContent = 'N' + i + ': ' + v.toFixed(3);
            return span;
        });
        display.replaceChildren(...spans);
    }

    render(parseInt(document.getElementById('delta-slider').value, 10));
    document.getElementById('delta-slider').addEventListener('input', e => {
        render(parseInt(e.target.value, 10));
    });
}
```

- [ ] **Step 2: Manual verification**

```bash
cargo run --bin viz_builder
```
Open `http://localhost:8080/dashboard.html`. Verify:
- Bar chart renders 4 bars, winner in gold
- Stats panel shows winner, silhouette, p-value, pass/fail in green/red
- Contrastive delta shows 32 colored neuron blocks
- Slider switches early/mid/late with label update

- [ ] **Step 3: Commit**

```bash
git add viz/brain_heatmap/js/dashboard.js
git commit -m "feat: add analysis dashboard with theory chart, stats, and contrastive delta viewer"
```

---

### Task 9: Integration — full end-to-end verification

**Files:**
- Create: `viz/brain_heatmap/assets/README.md`

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace
```
Expected: 53+ tests pass (49 existing + 7 viz_builder tests), 0 failures

- [ ] **Step 2: Run the full viz pipeline**

```bash
cargo run --bin viz_builder
```
Expected output (exact text varies by whether brain.obj.gz is found):
```
Building visualization data...
  Wrote viz/brain_heatmap/data/viz_data.json
  brain.obj.gz not found — JS will use sphere fallback

Serving at http://127.0.0.1:8080
Open http://127.0.0.1:8080/index.html in your browser
Press Ctrl+C to stop.
```

- [ ] **Step 3: Verify brain viewer checklist**

Open `http://localhost:8080/index.html`:
- [ ] 3D mesh renders (brain.obj or icosphere)
- [ ] Coolwarm colors applied to mesh surface
- [ ] Changing content type dropdown updates colors in < 100ms
- [ ] Moving layer slider updates label + re-colors mesh
- [ ] Theory panel shows 4 rows, winner highlighted in gold

- [ ] **Step 4: Verify dashboard checklist**

Open `http://localhost:8080/dashboard.html`:
- [ ] Bar chart renders with 4 bars, winner in gold
- [ ] Stats panel shows winner name, margin, silhouette, p-value
- [ ] Pass/fail indicators correct (green/red)
- [ ] Contrastive delta section shows 32 colored neuron spans
- [ ] Slider switches between early/mid/late layers

- [ ] **Step 5: Create `viz/brain_heatmap/assets/README.md`**

```
# Brain OBJ Asset

The viz_builder binary auto-decompresses brain.obj.gz from ../tribe-playground/
if that sibling directory exists. If not found, the viewer uses an icosphere fallback.

## Quick Start

  cargo run --bin viz_builder
  # Open http://localhost:8080/index.html

## With real brain mesh

Place brain.obj.gz from tribe-playground into this directory:
  cp /path/to/tribe-playground/brain.obj.gz viz/brain_heatmap/assets/
  cargo run --bin viz_builder

## What it renders

- Brain viewer: 3D mesh with coolwarm vertex coloring per content type x layer
  Vertices partitioned into 128 neuron buckets (uniform index partition).
  Documented limitation: heuristic mapping, not anatomically grounded.

- Dashboard: Theory fit bar chart, statistical summary, contrastive delta viewer
```

- [ ] **Step 6: Final commit**

```bash
git add viz/brain_heatmap/assets/README.md
git commit -m "docs: add viz asset README with quick start and limitations"
```

---

### Task 10: Stimulus explorer — live fingerprint from custom input

**Files:**
- Modify: `viz/builder/src/main.rs` (add POST /api/explore handler)
- Modify: `viz/brain_heatmap/dashboard.html` (add explorer section)
- Modify: `viz/brain_heatmap/js/dashboard.js` (fetch + render explorer result)

The stimulus explorer accepts arbitrary text, hashes it into a 64-dim float vector, runs `MockTribeModel`, and returns the activation fingerprint. This demonstrates the spec §8.3 "input custom text → see live activation fingerprint" feature using the mock model (real model integration is a follow-on task).

- [ ] **Step 1: Write the failing test for the text-to-vector hash**

```rust
// viz/builder/src/main.rs — add to #[cfg(test)] mod tests
#[test]
fn text_to_vec_produces_64_floats_in_range() {
    let v = text_to_input_vec("hello world");
    assert_eq!(v.len(), 64);
    assert!(v.iter().all(|&x| x >= -1.0 && x <= 1.0));
}

#[test]
fn same_text_produces_same_vec() {
    let a = text_to_input_vec("test input");
    let b = text_to_input_vec("test input");
    assert_eq!(a, b);
}

#[test]
fn different_text_produces_different_vec() {
    let a = text_to_input_vec("hello");
    let b = text_to_input_vec("world");
    assert_ne!(a, b);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p viz_builder
```
Expected: FAIL — `text_to_input_vec` not defined

- [ ] **Step 3: Add `text_to_input_vec` and POST handler to `main.rs`**

Add this function before `main()` in `viz/builder/src/main.rs`:

```rust
fn text_to_input_vec(text: &str) -> Vec<f64> {
    // Deterministic hash of text bytes into 64 floats in [-1, 1].
    // Uses FNV-1a variant: each output dimension i mixes text chars with seed i.
    let bytes = text.as_bytes();
    (0..64).map(|i: usize| {
        let mut h: u64 = 0xcbf29ce484222325u64.wrapping_add(i as u64 * 0x100000001b3);
        for (j, &b) in bytes.iter().enumerate() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= (j as u64).wrapping_mul(0x9e3779b97f4a7c15);
        }
        // map u64 to [-1, 1]
        (h as i64 as f64) / (i64::MAX as f64)
    }).collect()
}
```

Add a POST `/api/explore` branch in the request handling loop in `main()` (insert before the existing file-serve logic):

```rust
        if request.method() == &tiny_http::Method::Post
            && request.url().starts_with("/api/explore") {
            let mut body = String::new();
            request.as_reader().read_to_string(&mut body).unwrap_or(0);
            let text = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["text"].as_str().map(|s| s.to_string()))
                .unwrap_or_default();
            let input = text_to_input_vec(&text);
            let model = experiments::MockTribeModel::new(128, 64);
            let probes = experiments::TribeModel::forward(&model, &input);
            let result = serde_json::json!({
                "early": probes.early,
                "mid": probes.mid,
                "late": probes.late,
                "input_text": text,
            });
            let json = result.to_string();
            let header = Header::from_bytes(&b"Content-Type"[..], b"application/json")
                .expect("invalid header");
            let _ = request.respond(Response::from_string(json).with_header(header));
            continue;
        }
```

Add `use std::io::Read;` at the top of `main.rs` (after the existing `use` statements) and `use experiments::{MockTribeModel, TribeModel};`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p viz_builder
```
Expected: 10 tests PASS (7 existing + 3 new)

- [ ] **Step 5: Add stimulus explorer section to `dashboard.html`**

Add before the closing `</body>` tag and after the contrastive delta card:

```html
    <h2>Stimulus Explorer</h2>
    <div class="card">
        <label for="explorer-input">Enter any text (hashed into 64-dim input vector via mock model)</label>
        <div style="display:flex;gap:8px;margin:8px 0;">
            <input type="text" id="explorer-input"
                   placeholder="Type something..."
                   style="flex:1;background:#252545;border:1px solid #333;
                          color:#e0e0e0;padding:8px;border-radius:4px;">
            <button id="explorer-btn"
                    style="background:#7c9cbf;color:#0d0d0d;border:none;
                           padding:8px 16px;border-radius:4px;cursor:pointer;">
                Run
            </button>
        </div>
        <div id="explorer-label" style="font-size:12px;color:#7c9cbf;margin-bottom:6px;"></div>
        <div id="explorer-display" style="display:flex;flex-wrap:wrap;gap:2px;
             font-size:11px;font-family:monospace;margin-top:4px;"></div>
    </div>
```

- [ ] **Step 6: Add `renderExplorer()` to `dashboard.js`**

Append to the bottom of `viz/brain_heatmap/js/dashboard.js`:

```js
function initExplorer() {
    const btn = document.getElementById('explorer-btn');
    const input = document.getElementById('explorer-input');
    if (!btn || !input) return;

    btn.addEventListener('click', () => runExplorer(input.value.trim()));
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') runExplorer(input.value.trim());
    });
}

function runExplorer(text) {
    if (!text) return;
    const label = document.getElementById('explorer-label');
    const display = document.getElementById('explorer-display');
    label.textContent = 'Running...';
    display.replaceChildren();

    fetch('/api/explore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    })
    .then(r => r.json())
    .then(result => {
        const layerData = result.mid; // show mid-layer activations
        const maxVal = Math.max(...layerData.map(Math.abs), 1e-10);
        label.textContent = 'Mid-layer activation for: "' + result.input_text + '"';
        const spans = layerData.slice(0, 32).map((v, i) => {
            const t = (v / maxVal + 1) / 2;
            const [r, g, b] = coolwarmRgb(t);
            const span = document.createElement('span');
            span.style.background =
                'rgb(' + [r, g, b].map(x => Math.round(x * 255)).join(',') + ')';
            span.style.color = (t > 0.35 && t < 0.65) ? '#333' : '#fff';
            span.textContent = 'N' + i + ': ' + v.toFixed(3);
            return span;
        });
        display.replaceChildren(...spans);
    })
    .catch(err => { label.textContent = 'Error: ' + err.message; });
}
```

Also add a call to `initExplorer()` at the end of the `.then(data => { ... })` block in `dashboard.js`:

```js
fetch('data/viz_data.json')
    .then(r => r.json())
    .then(data => {
        renderTheoryChart(data.theory_fit);
        renderStats(data);
        renderContrastiveDelta(data.contrastive_deltas);
        initExplorer();
    });
```

- [ ] **Step 7: Manual verification**

```bash
cargo run --bin viz_builder
```
Open `http://localhost:8080/dashboard.html`. Verify:
- Stimulus explorer text input + Run button appear at bottom
- Typing "hello world" and clicking Run shows 32 colored activation cells
- Different inputs produce different patterns
- Enter key triggers the same as clicking Run

- [ ] **Step 8: Commit**

```bash
git add viz/builder/src/main.rs viz/brain_heatmap/dashboard.html viz/brain_heatmap/js/dashboard.js
git commit -m "feat: add stimulus explorer with POST /api/explore and live fingerprint display"
```
