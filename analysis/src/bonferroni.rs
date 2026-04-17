pub fn bonferroni_correct(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let corrected_alpha = alpha / p_values.len() as f64;
    p_values.iter().map(|&p| p < corrected_alpha).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significant_after_correction() {
        let p_values = vec![0.001_f64; 10];
        let significant = bonferroni_correct(&p_values, 0.05);
        assert!(significant.iter().all(|&s| s));
    }

    #[test]
    fn not_significant_after_correction() {
        let p_values = vec![0.04_f64; 10];
        let significant = bonferroni_correct(&p_values, 0.05);
        assert!(significant.iter().all(|&s| !s));
    }
}
