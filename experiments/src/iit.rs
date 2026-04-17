use crate::ActivationRecord;

pub struct IitScorer;

impl IitScorer {
    /// Returns normalized mutual information proxy in [0, 1]
    pub fn score(records: &[ActivationRecord]) -> f64 {
        if records.is_empty() { return 0.0; }

        let scores: Vec<f64> = records.iter().map(|rec| {
            let layers = [&rec.early, &rec.mid, &rec.late];
            let layer_scores: Vec<f64> = layers.iter().map(|layer| {
                let n = layer.len();
                if n < 4 { return 0.0; }
                let mid = n / 2;
                let a = &layer[..mid];
                let b = &layer[mid..];
                normalized_mutual_information(a, b)
            }).collect();
            layer_scores.iter().sum::<f64>() / 3.0
        }).collect();

        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

fn normalized_mutual_information(a: &[f64], b: &[f64]) -> f64 {
    let bins = 10usize;
    let n = a.len().min(b.len()) as f64;
    if n == 0.0 { return 0.0; }

    let bin_idx = |v: f64| -> usize {
        let clamped = v.clamp(0.0, 1.0);
        let idx = (clamped * bins as f64).floor() as usize;
        idx.min(bins - 1)
    };

    let mut joint = vec![vec![0.0_f64; bins]; bins];
    for (&va, &vb) in a.iter().zip(b.iter()) {
        joint[bin_idx(va)][bin_idx(vb)] += 1.0 / n;
    }

    let marginal_a: Vec<f64> = (0..bins).map(|i| joint[i].iter().sum()).collect();
    let marginal_b: Vec<f64> = (0..bins).map(|j| joint.iter().map(|row| row[j]).sum()).collect();

    let entropy = |p: &[f64]| -> f64 {
        -p.iter().filter(|&&v| v > 1e-10).map(|&v| v * v.ln()).sum::<f64>()
    };

    let h_a = entropy(&marginal_a);
    let h_b = entropy(&marginal_b);
    let h_joint: f64 = -joint.iter().flatten().filter(|&&v| v > 1e-10)
        .map(|&v| v * v.ln()).sum::<f64>();

    let mi = (h_a + h_b - h_joint).max(0.0);
    let norm = h_a.max(h_b);
    if norm < 1e-10 { return 0.0; }
    (mi / norm).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActivationRecord, ContentType};

    fn record(ct: ContentType, pattern: Vec<f64>) -> ActivationRecord {
        ActivationRecord {
            stimulus_id: 0,
            content_type: ct,
            early: pattern.clone(),
            mid: pattern.clone(),
            late: pattern,
        }
    }

    #[test]
    fn high_integration_scores_higher_than_uniform() {
        let structured: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 0.9 } else { 0.1 }).collect();
        let uniform = vec![0.5_f64; 20];

        let structured_score = IitScorer::score(&[record(ContentType::Narrative, structured)]);
        let uniform_score = IitScorer::score(&[record(ContentType::Factual, uniform)]);

        assert!(structured_score > uniform_score,
            "structured ({}) should score higher than uniform ({})", structured_score, uniform_score);
    }

    #[test]
    fn score_is_in_valid_range() {
        let v: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
        let records = vec![record(ContentType::Audio, v)];
        let score = IitScorer::score(&records);
        assert!(score >= 0.0 && score <= 1.0, "score {} not in [0,1]", score);
    }
}
