use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;

pub fn bootstrap_ci(data: &[f64], n_bootstrap: usize, confidence: f64, seed: u64) -> (f64, f64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let n = data.len();
    let mut boot_means: Vec<f64> = (0..n_bootstrap).map(|_| {
        let sample: Vec<f64> = (0..n).map(|_| data[rng.gen_range(0..n)]).collect();
        sample.iter().sum::<f64>() / n as f64
    }).collect();

    boot_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    (boot_means[lo_idx], boot_means[hi_idx.min(n_bootstrap - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_contains_true_mean() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect(); // mean = 50.5
        let (lo, hi) = bootstrap_ci(&data, 1000, 0.95, 42);
        assert!(lo < 50.5 && 50.5 < hi, "CI [{}, {}] should contain 50.5", lo, hi);
    }

    #[test]
    fn ci_lo_less_than_hi() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lo, hi) = bootstrap_ci(&data, 500, 0.95, 0);
        assert!(lo < hi);
    }
}
