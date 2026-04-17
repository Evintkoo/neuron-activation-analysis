use ndarray::{Array1, Array2, Axis};
use linfa::traits::Fit;
use linfa_reduction::Pca;
use linfa::DatasetBase;

pub struct PcaReducer {
    components: Array2<f64>,
    mean: Array1<f64>,
    n_components: usize,
}

impl PcaReducer {
    pub fn fit(data: &Array2<f64>, n_components: usize) -> Self {
        let dataset = DatasetBase::from(data.clone());
        let pca = Pca::params(n_components)
            .fit(&dataset)
            .expect("PCA fit failed");

        let mean = data.mean_axis(Axis(0)).unwrap();
        let components = pca.components().clone();

        Self { components, mean, n_components }
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        let centered = data - &self.mean;
        let projected = centered.dot(&self.components.t());
        // If the data is rank-deficient, linfa may return fewer components than
        // requested. Pad with zeros to always return exactly n_components columns.
        let actual = projected.ncols();
        if actual == self.n_components {
            projected
        } else {
            let mut out = Array2::zeros((projected.nrows(), self.n_components));
            out.slice_mut(ndarray::s![.., ..actual]).assign(&projected);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn reduces_to_target_components() {
        let data = Array2::from_shape_fn((20, 16), |(i, j)| (i * j) as f64 * 0.01);
        let reducer = PcaReducer::fit(&data, 4);
        let reduced = reducer.transform(&data);
        assert_eq!(reduced.nrows(), 20);
        assert_eq!(reduced.ncols(), 4);
    }

    #[test]
    fn transform_is_deterministic() {
        let data = Array2::from_shape_fn((10, 8), |(i, j)| (i + j) as f64);
        let reducer = PcaReducer::fit(&data, 3);
        let r1 = reducer.transform(&data);
        let r2 = reducer.transform(&data);
        for (a, b) in r1.iter().zip(r2.iter()) {
            approx::assert_abs_diff_eq!(a, b, epsilon = 1e-10);
        }
    }
}
