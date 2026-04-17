use crate::{ActivationFingerprint, ContentType};

pub struct DctScorer;

const VERBAL_TYPES: &[ContentType] = &[
    ContentType::TextVerbal,
    ContentType::Narrative,
    ContentType::Abstract,
    ContentType::Factual,
];

const NONVERBAL_TYPES: &[ContentType] = &[
    ContentType::ImageVisual,
    ContentType::Audio,
    ContentType::Spatial,
];

impl DctScorer {
    /// Returns a score in [0, 1] where 1 = perfect DCT separation (verbal and non-verbal maximally different)
    pub fn score(fingerprints: &[ActivationFingerprint]) -> f64 {
        let verbal_centroids: Vec<&Vec<f64>> = fingerprints
            .iter()
            .filter(|f| VERBAL_TYPES.contains(&f.content_type))
            .map(|f| &f.centroid)
            .collect();

        let nonverbal_centroids: Vec<&Vec<f64>> = fingerprints
            .iter()
            .filter(|f| NONVERBAL_TYPES.contains(&f.content_type))
            .map(|f| &f.centroid)
            .collect();

        if verbal_centroids.is_empty() || nonverbal_centroids.is_empty() {
            return 0.0;
        }

        let verbal_mean = mean_centroid(&verbal_centroids);
        let nonverbal_mean = mean_centroid(&nonverbal_centroids);

        let cos_sim = cosine_similarity(&verbal_mean, &nonverbal_mean);
        (1.0 - cos_sim.abs()).clamp(0.0, 1.0)
    }
}

fn mean_centroid(centroids: &[&Vec<f64>]) -> Vec<f64> {
    let dim = centroids[0].len();
    (0..dim)
        .map(|d| centroids.iter().map(|c| c[d]).sum::<f64>() / centroids.len() as f64)
        .collect()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActivationFingerprint, ContentType};

    fn fp(ct: ContentType, centroid: Vec<f64>) -> ActivationFingerprint {
        ActivationFingerprint {
            content_type: ct,
            centroid,
            variance: vec![0.1; 4],
            sample_count: 10,
        }
    }

    #[test]
    fn separated_verbal_visual_clusters_score_high() {
        let fps = vec![
            fp(ContentType::TextVerbal, vec![1.0, 0.0, 0.0, 0.0]),
            fp(ContentType::ImageVisual, vec![0.0, 0.0, 0.0, 1.0]),
        ];
        let score = DctScorer::score(&fps);
        assert!(score > 0.7, "expected high DCT score for separated clusters, got {}", score);
    }

    #[test]
    fn identical_clusters_score_low() {
        let fps = vec![
            fp(ContentType::TextVerbal, vec![0.5, 0.5, 0.5, 0.5]),
            fp(ContentType::ImageVisual, vec![0.5, 0.5, 0.5, 0.5]),
        ];
        let score = DctScorer::score(&fps);
        assert!(score < 0.2, "expected low DCT score for identical clusters, got {}", score);
    }
}
