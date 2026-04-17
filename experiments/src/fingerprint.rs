use serde::{Deserialize, Serialize};
use ndarray::Array2;
use crate::ContentType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationFingerprint {
    pub content_type: ContentType,
    pub centroid: Vec<f64>,
    pub variance: Vec<f64>,
    pub sample_count: usize,
}

impl ActivationFingerprint {
    /// data: [n_stimuli, n_features], labels: cluster label per stimulus
    /// content_types: ordered list mapping label index → ContentType
    pub fn from_clusters(
        data: &Array2<f64>,
        labels: &[usize],
        content_types: &[ContentType],
    ) -> Vec<ActivationFingerprint> {
        content_types.iter().enumerate().map(|(label, &ct)| {
            let indices: Vec<usize> = labels.iter().enumerate()
                .filter(|(_, &l)| l == label)
                .map(|(i, _)| i)
                .collect();

            let n = indices.len();
            let dim = data.ncols();

            if n == 0 {
                return ActivationFingerprint {
                    content_type: ct,
                    centroid: vec![0.0; dim],
                    variance: vec![0.0; dim],
                    sample_count: 0,
                };
            }

            let centroid: Vec<f64> = (0..dim)
                .map(|d| indices.iter().map(|&i| data[[i, d]]).sum::<f64>() / n as f64)
                .collect();

            let variance: Vec<f64> = (0..dim)
                .map(|d| {
                    let mean = centroid[d];
                    indices.iter().map(|&i| (data[[i, d]] - mean).powi(2)).sum::<f64>() / n as f64
                })
                .collect();

            ActivationFingerprint { content_type: ct, centroid, variance, sample_count: n }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use crate::ContentType;

    #[test]
    fn fingerprint_centroid_is_column_mean() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let labels = vec![0usize, 0, 0];
        let fps = ActivationFingerprint::from_clusters(&data, &labels, &[ContentType::Audio]);
        let audio_fp = fps.iter().find(|f| f.content_type == ContentType::Audio).unwrap();
        approx::assert_abs_diff_eq!(audio_fp.centroid[0], 3.0, epsilon = 1e-9);
        approx::assert_abs_diff_eq!(audio_fp.centroid[1], 4.0, epsilon = 1e-9);
    }

    #[test]
    fn variance_is_non_negative() {
        let data = Array2::from_shape_fn((10, 4), |(i, j)| (i * j) as f64);
        let labels = vec![0usize; 10];
        let fps = ActivationFingerprint::from_clusters(&data, &labels, &[ContentType::Narrative]);
        let fp = &fps[0];
        assert!(fp.variance.iter().all(|&v| v >= 0.0));
    }
}
