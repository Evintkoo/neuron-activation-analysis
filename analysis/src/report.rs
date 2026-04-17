use serde::{Deserialize, Serialize};
use experiments::TheoryFitReport;

#[derive(Debug, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub theory_fit: TheoryFitReport,
    pub silhouette_scores: Vec<f64>,
    pub anova_p_value: f64,
    pub contrastive_deltas: Vec<f64>,
    pub bootstrap_cis: Vec<(f64, f64)>,
    pub bonferroni_significant: Vec<bool>,
}

pub struct SuccessCriteria {
    pub silhouette_met: bool,
    pub winner_margin_met: bool,
    pub contrastive_significant: bool,
}

impl ExperimentReport {
    pub fn to_json(&self) -> serde_json::Result<String> {
        #[derive(Serialize)]
        struct WithWinner<'a> {
            #[serde(flatten)]
            report: &'a ExperimentReport,
            winner: &'static str,
            winner_margin: f64,
        }
        serde_json::to_string_pretty(&WithWinner {
            report: self,
            winner: self.theory_fit.winner(),
            winner_margin: self.theory_fit.winner_margin(),
        })
    }

    pub fn success_criteria(&self) -> SuccessCriteria {
        let above_threshold = self.silhouette_scores.iter().filter(|&&s| s > 0.6).count();
        let total = self.silhouette_scores.len();
        let threshold_count = ((total as f64) * (8.0 / 13.0)).ceil() as usize;

        SuccessCriteria {
            silhouette_met: above_threshold >= threshold_count,
            winner_margin_met: self.theory_fit.winner_margin() >= 0.1,
            contrastive_significant: self.bonferroni_significant.iter().all(|&s| s),
        }
    }

    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_report() -> ExperimentReport {
        ExperimentReport {
            theory_fit: TheoryFitReport { dct_score: 0.7, gwt_score: 0.5, fep_score: 0.8, iit_score: 0.6 },
            silhouette_scores: vec![0.75, 0.68, 0.72],
            anova_p_value: 0.001,
            contrastive_deltas: vec![0.3, 0.5, 0.2, 0.4],
            bootstrap_cis: vec![(0.6, 0.9), (0.4, 0.7)],
            bonferroni_significant: vec![true, false, true, true],
        }
    }

    #[test]
    fn serializes_to_valid_json() {
        let report = dummy_report();
        let json = report.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["winner"], "FEP");
    }

    #[test]
    fn meets_success_criteria_check() {
        let report = dummy_report();
        let criteria = report.success_criteria();
        assert!(criteria.silhouette_met);   // 3/3 above 0.6
        assert!(criteria.winner_margin_met); // 0.8 - 0.7 = 0.1
    }
}
