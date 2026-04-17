use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaResult {
    pub f_stat: f64,
    pub p_value: f64,
    pub df_between: usize,
    pub df_within: usize,
}

pub fn one_way_anova(groups: &[Vec<f64>]) -> AnovaResult {
    let k = groups.len();
    let n: usize = groups.iter().map(|g| g.len()).sum();

    let grand_mean: f64 = groups.iter().flat_map(|g| g.iter()).sum::<f64>() / n as f64;

    let ss_between: f64 = groups.iter().map(|g| {
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        g.len() as f64 * (mean - grand_mean).powi(2)
    }).sum();

    let ss_within: f64 = groups.iter().map(|g| {
        let mean = g.iter().sum::<f64>() / g.len() as f64;
        g.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
    }).sum();

    let df_between = k - 1;
    let df_within = n - k;

    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;
    let f_stat = if ms_within == 0.0 { f64::INFINITY } else { ms_between / ms_within };

    let dist = FisherSnedecor::new(df_between as f64, df_within as f64)
        .expect("invalid F distribution parameters");
    let p_value = 1.0 - dist.cdf(f_stat);

    AnovaResult { f_stat, p_value, df_between, df_within }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearly_different_groups_have_low_p() {
        let groups = vec![
            vec![0.1, 0.2, 0.15, 0.05, 0.12],
            vec![99.8, 100.1, 100.2, 99.9, 100.0],
        ];
        let result = one_way_anova(&groups);
        assert!(result.p_value < 0.001, "expected p < 0.001, got {}", result.p_value);
    }

    #[test]
    fn same_group_has_high_p() {
        let groups = vec![
            vec![5.0, 5.1, 4.9, 5.0, 5.05],
            vec![5.0, 4.95, 5.1, 5.0, 4.98],
        ];
        let result = one_way_anova(&groups);
        assert!(result.p_value > 0.5, "expected high p-value for similar groups, got {}", result.p_value);
    }

    #[test]
    fn f_stat_is_positive() {
        let groups = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = one_way_anova(&groups);
        assert!(result.f_stat > 0.0);
    }
}
