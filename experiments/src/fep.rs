use crate::{ActivationRecord, ContentType};

fn novelty_rank(ct: ContentType) -> f64 {
    match ct {
        ContentType::Novelty => 1.0,
        ContentType::ThreatSafety => 0.9,
        ContentType::Emotional => 0.8,
        ContentType::SocialInteraction => 0.7,
        ContentType::Narrative => 0.65,
        ContentType::Abstract => 0.6,
        ContentType::Multimodal => 0.55,
        ContentType::RewardSignal => 0.5,
        ContentType::ImageVisual => 0.4,
        ContentType::Audio => 0.35,
        ContentType::Spatial => 0.3,
        ContentType::TextVerbal => 0.2,
        ContentType::Factual => 0.1,
    }
}

pub struct FepScorer;

impl FepScorer {
    /// Returns Spearman correlation mapped to [0, 1] between novelty rank and activation magnitude.
    /// Score near 1.0 = FEP predicts well (high novelty → high activation).
    pub fn score(records: &[ActivationRecord]) -> f64 {
        if records.len() < 2 { return 0.0; }

        let pairs: Vec<(f64, f64)> = records.iter().map(|r| {
            let novelty = novelty_rank(r.content_type);
            let mag = mean_activation_magnitude(r);
            (novelty, mag)
        }).collect();

        let rho = spearman_correlation(&pairs);
        ((rho + 1.0) / 2.0).clamp(0.0, 1.0)
    }
}

fn mean_activation_magnitude(r: &ActivationRecord) -> f64 {
    let all: Vec<f64> = r.early.iter().chain(&r.mid).chain(&r.late)
        .map(|&v| v.abs())
        .collect();
    all.iter().sum::<f64>() / all.len() as f64
}

fn spearman_correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    let rank_vec = |vals: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0; n];
        for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
            ranks[*orig_idx] = (rank + 1) as f64;
        }
        ranks
    };
    let xs: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let rx = rank_vec(&xs);
    let ry = rank_vec(&ys);
    pearson_correlation(&rx, &ry)
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - mean_a) * (y - mean_b)).sum::<f64>() / n;
    let std_a = (a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n).sqrt();
    let std_b = (b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / n).sqrt();
    if std_a == 0.0 || std_b == 0.0 { return 0.0; }
    cov / (std_a * std_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_with_magnitude(ct: ContentType, mag: f64) -> ActivationRecord {
        ActivationRecord {
            stimulus_id: 0,
            content_type: ct,
            early: vec![mag; 16],
            mid: vec![mag; 16],
            late: vec![mag; 16],
        }
    }

    #[test]
    fn positive_correlation_scores_high() {
        // High novelty types have high activation = FEP predicts
        let records = vec![
            record_with_magnitude(ContentType::Novelty, 0.9),
            record_with_magnitude(ContentType::ThreatSafety, 0.8),
            record_with_magnitude(ContentType::Factual, 0.1),
            record_with_magnitude(ContentType::TextVerbal, 0.2),
        ];
        let score = FepScorer::score(&records);
        assert!(score > 0.6, "expected positive FEP score, got {}", score);
    }

    #[test]
    fn negative_correlation_scores_low() {
        let records = vec![
            record_with_magnitude(ContentType::Novelty, 0.1),
            record_with_magnitude(ContentType::Factual, 0.9),
        ];
        let score = FepScorer::score(&records);
        assert!(score < 0.5, "expected low FEP score, got {}", score);
    }
}
