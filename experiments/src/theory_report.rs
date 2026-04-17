use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryFitReport {
    pub dct_score: f64,
    pub gwt_score: f64,
    pub fep_score: f64,
    pub iit_score: f64,
}

impl TheoryFitReport {
    pub fn winner(&self) -> &'static str {
        let scores = [
            ("DCT", self.dct_score),
            ("GWT", self.gwt_score),
            ("FEP", self.fep_score),
            ("IIT", self.iit_score),
        ];
        scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
    }

    pub fn winner_margin(&self) -> f64 {
        let mut scores = [self.dct_score, self.gwt_score, self.fep_score, self.iit_score];
        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        scores[0] - scores[1]
    }

    pub fn meets_success_criteria(&self) -> bool {
        self.winner_margin() >= 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winner_is_highest_mean_score() {
        let report = TheoryFitReport {
            dct_score: 0.8,
            gwt_score: 0.6,
            fep_score: 0.9,
            iit_score: 0.7,
        };
        assert_eq!(report.winner(), "FEP");
    }

    #[test]
    fn winner_margin_is_correct() {
        let report = TheoryFitReport {
            dct_score: 0.5,
            gwt_score: 0.5,
            fep_score: 0.8,
            iit_score: 0.6,
        };
        approx::assert_abs_diff_eq!(report.winner_margin(), 0.2, epsilon = 1e-9);
    }

    #[test]
    fn serializes_to_json() {
        let report = TheoryFitReport {
            dct_score: 0.5,
            gwt_score: 0.5,
            fep_score: 0.5,
            iit_score: 0.5,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("dct_score"));
    }
}
