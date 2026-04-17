use crate::ActivationRecord;

pub struct GwtScorer;

impl GwtScorer {
    /// tau: activation threshold (e.g. 0.5)
    /// tau_steps: number of probe layers to include (1-3)
    /// Returns mean ignition score in [0, 1] across all records
    pub fn score(records: &[ActivationRecord], tau: f64, tau_steps: usize) -> f64 {
        if records.is_empty() {
            return 0.0;
        }

        let scores: Vec<f64> = records.iter().map(|rec| {
            let layers = [&rec.early, &rec.mid, &rec.late];
            let spread_counts = layers.iter().map(|layer| {
                let active = layer.iter().filter(|&&v| v > tau).count();
                active as f64 / layer.len() as f64
            }).collect::<Vec<_>>();

            let steps = tau_steps.min(3);
            spread_counts[..steps].iter().sum::<f64>() / steps as f64
        }).collect();

        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActivationRecord, ContentType};

    fn record(ct: ContentType, activation_fraction: f64) -> ActivationRecord {
        let n = 100;
        let active = (n as f64 * activation_fraction) as usize;
        let v: Vec<f64> = (0..n).map(|i| if i < active { 0.9 } else { 0.1 }).collect();
        ActivationRecord {
            stimulus_id: 0,
            content_type: ct,
            early: v.clone(),
            mid: v.clone(),
            late: v,
        }
    }

    #[test]
    fn high_spread_stimulus_scores_high() {
        let records = vec![record(ContentType::ThreatSafety, 0.80)];
        let score = GwtScorer::score(&records, 0.5, 3);
        assert!(score > 0.7, "expected high GWT score, got {}", score);
    }

    #[test]
    fn low_spread_stimulus_scores_low() {
        let records = vec![record(ContentType::Factual, 0.10)];
        let score = GwtScorer::score(&records, 0.5, 3);
        assert!(score < 0.3, "expected low GWT score, got {}", score);
    }
}
