use serde::{Deserialize, Serialize};
use experiments::ActivationRecord;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContrastiveDelta {
    pub early_delta: Vec<f64>,
    pub mid_delta: Vec<f64>,
    pub late_delta: Vec<f64>,
}

impl ContrastiveDelta {
    pub fn compute(a: &ActivationRecord, b: &ActivationRecord) -> Self {
        let diff = |x: &[f64], y: &[f64]| -> Vec<f64> {
            x.iter().zip(y.iter()).map(|(xi, yi)| yi - xi).collect()
        };
        Self {
            early_delta: diff(&a.early, &b.early),
            mid_delta: diff(&a.mid, &b.mid),
            late_delta: diff(&a.late, &b.late),
        }
    }

    pub fn l2_norm(&self) -> f64 {
        let sum: f64 = self.early_delta.iter()
            .chain(&self.mid_delta)
            .chain(&self.late_delta)
            .map(|v| v * v)
            .sum();
        sum.sqrt()
    }

    pub fn mean_absolute_delta(&self) -> f64 {
        let all: Vec<f64> = self.early_delta.iter()
            .chain(&self.mid_delta)
            .chain(&self.late_delta)
            .map(|v| v.abs())
            .collect();
        all.iter().sum::<f64>() / all.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use experiments::ContentType;

    fn record(id: usize, ct: ContentType, vals: Vec<f64>) -> ActivationRecord {
        ActivationRecord { stimulus_id: id, content_type: ct, early: vals.clone(), mid: vals.clone(), late: vals }
    }

    #[test]
    fn delta_is_element_wise_difference() {
        let a = record(0, ContentType::ImageVisual, vec![1.0, 2.0, 3.0]);
        let b = record(1, ContentType::Emotional, vec![4.0, 6.0, 9.0]);
        let delta = ContrastiveDelta::compute(&a, &b);
        approx::assert_abs_diff_eq!(delta.early_delta[0], 3.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(delta.early_delta[1], 4.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(delta.early_delta[2], 6.0, epsilon = 1e-9);
    }

    #[test]
    fn l2_norm_of_zero_delta_is_zero() {
        let a = record(0, ContentType::Audio, vec![0.5, 0.5]);
        let b = record(1, ContentType::Audio, vec![0.5, 0.5]);
        let delta = ContrastiveDelta::compute(&a, &b);
        approx::assert_abs_diff_eq!(delta.l2_norm(), 0.0, epsilon = 1e-9);
    }
}
