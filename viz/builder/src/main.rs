// viz/builder/src/main.rs
mod export;

use std::{fs, io::Read, path::{Path, PathBuf}};
use tiny_http::{Header, Response, Server};
use experiments::{MockTribeModel, TribeModel};

fn text_to_input_vec(text: &str) -> Vec<f64> {
    let bytes = text.as_bytes();
    (0..64).map(|i: usize| {
        let mut h: u64 = 0xcbf29ce484222325u64.wrapping_add(i as u64 * 0x100000001b3);
        for (j, &b) in bytes.iter().enumerate() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= (j as u64).wrapping_mul(0x9e3779b97f4a7c15);
        }
        (h as i64 as f64) / (i64::MAX as f64)
    }).collect()
}

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

    for mut request in server.incoming_requests() {
        // POST /api/explore — live stimulus fingerprint
        if request.method() == &tiny_http::Method::Post
            && request.url().starts_with("/api/explore")
        {
            let mut body = String::new();
            request.as_reader().read_to_string(&mut body).unwrap_or(0);
            let text = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["text"].as_str().map(|s| s.to_string()))
                .unwrap_or_default();
            let input = text_to_input_vec(&text);
            let model = MockTribeModel::new(128, 64);
            let probes = model.forward(&input);
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

        let url = request.url().to_string();
        let rel = url.trim_start_matches('/');
        let rel = if rel.is_empty() { "index.html" } else { rel };
        // Reject paths containing .. to prevent directory traversal
        let rel_path = PathBuf::from(rel);
        if rel_path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
            let _ = request.respond(Response::from_string("403 Forbidden").with_status_code(403));
            continue;
        }
        let file_path: PathBuf = viz_dir.join(rel_path);

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

    #[test]
    fn path_with_parent_dir_components_is_rejected() {
        use std::path::PathBuf;
        let p = PathBuf::from("../../secret.txt");
        let has_parent = p.components().any(|c| matches!(c, std::path::Component::ParentDir));
        assert!(has_parent, ".. components should be detected");
    }

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
}
