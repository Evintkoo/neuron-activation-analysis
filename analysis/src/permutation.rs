use rand::{seq::SliceRandom, SeedableRng};
use rand::rngs::SmallRng;

pub fn permutation_test(a: &[f64], b: &[f64], n_permutations: usize, seed: u64) -> f64 {
    let observed_diff = mean(a) - mean(b);
    let mut combined: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let na = a.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut count_as_extreme = 0usize;

    for _ in 0..n_permutations {
        combined.shuffle(&mut rng);
        let perm_diff = mean(&combined[..na]) - mean(&combined[na..]);
        if perm_diff.abs() >= observed_diff.abs() {
            count_as_extreme += 1;
        }
    }

    count_as_extreme as f64 / n_permutations as f64
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearly_different_groups_have_low_p() {
        let a: Vec<f64> = (0..20).map(|_| 0.1).collect();
        let b: Vec<f64> = (0..20).map(|_| 5.0).collect();
        let p = permutation_test(&a, &b, 1000, 42);
        assert!(p < 0.05, "expected p < 0.05, got {}", p);
    }

    #[test]
    fn same_distribution_has_high_p() {
        let a: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let b = a.clone();
        let p = permutation_test(&a, &b, 1000, 42);
        assert!(p > 0.5, "expected p > 0.5, got {}", p);
    }
}
