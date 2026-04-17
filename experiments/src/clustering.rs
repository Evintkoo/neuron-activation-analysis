use std::collections::HashMap;
use ndarray::Array2;
use rand_xoshiro::Xoshiro256Plus;
use rand::SeedableRng;
use linfa::traits::{Fit, Predict};
use linfa_clustering::KMeans;
use linfa::DatasetBase;

pub struct ClusterResult {
    pub labels: Vec<usize>,
    pub centroids: Array2<f64>,
    pub silhouette_score: f64,
}

impl ClusterResult {
    pub fn label_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &label in &self.labels {
            *counts.entry(label).or_insert(0) += 1;
        }
        counts
    }
}

pub struct KMeansClusterer;

impl KMeansClusterer {
    pub fn run(data: &Array2<f64>, k: usize, seed: u64) -> ClusterResult {
        let rng = Xoshiro256Plus::seed_from_u64(seed);
        let dataset = DatasetBase::from(data.clone());
        let model = KMeans::params_with_rng(k, rng)
            .max_n_iterations(300)
            .tolerance(1e-4)
            .fit(&dataset)
            .expect("k-means fit failed");

        let predicted = model.predict(dataset);
        let labels: Vec<usize> = predicted
            .targets()
            .iter()
            .copied()
            .collect();

        let centroids = model.centroids().clone();
        let silhouette_score = Self::silhouette(data, &labels, k);

        ClusterResult { labels, centroids, silhouette_score }
    }

    fn silhouette(data: &Array2<f64>, labels: &[usize], k: usize) -> f64 {
        let n = data.nrows();
        if n < 2 { return 0.0; }

        let scores: Vec<f64> = (0..n).map(|i| {
            let row_i = data.row(i);
            let same_cluster: Vec<usize> = (0..n)
                .filter(|&j| j != i && labels[j] == labels[i])
                .collect();

            if same_cluster.is_empty() { return 0.0; }

            let a = same_cluster.iter()
                .map(|&j| euclidean(row_i.view(), data.row(j)))
                .sum::<f64>() / same_cluster.len() as f64;

            let b = (0..k)
                .filter(|&c| c != labels[i])
                .map(|c| {
                    let other: Vec<usize> = (0..n)
                        .filter(|&j| labels[j] == c)
                        .collect();
                    if other.is_empty() { return f64::INFINITY; }
                    other.iter()
                        .map(|&j| euclidean(row_i.view(), data.row(j)))
                        .sum::<f64>() / other.len() as f64
                })
                .fold(f64::INFINITY, f64::min);

            (b - a) / a.max(b)
        }).collect();

        scores.iter().sum::<f64>() / n as f64
    }
}

fn euclidean(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn two_clear_clusters() -> Array2<f64> {
        let mut rows: Vec<f64> = Vec::new();
        for i in 0..10 {
            rows.push(i as f64 * 0.01);
            rows.push(i as f64 * 0.01);
        }
        for i in 0..10 {
            rows.push(10.0 + i as f64 * 0.01);
            rows.push(10.0 + i as f64 * 0.01);
        }
        Array2::from_shape_vec((20, 2), rows).unwrap()
    }

    #[test]
    fn clusters_obvious_data_into_two_groups() {
        let data = two_clear_clusters();
        let result = KMeansClusterer::run(&data, 2, 42);
        let counts = result.label_counts();
        assert_eq!(counts.len(), 2);
        assert!(counts.values().all(|&c| c == 10));
    }

    #[test]
    fn silhouette_score_high_for_clear_clusters() {
        let data = two_clear_clusters();
        let result = KMeansClusterer::run(&data, 2, 42);
        assert!(result.silhouette_score > 0.9,
            "expected silhouette > 0.9, got {}", result.silhouette_score);
    }

    #[test]
    fn centroids_shape_is_correct() {
        let data = two_clear_clusters();
        let result = KMeansClusterer::run(&data, 2, 42);
        assert_eq!(result.centroids.nrows(), 2);
        assert_eq!(result.centroids.ncols(), 2);
    }
}
