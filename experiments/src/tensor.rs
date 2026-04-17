use ndarray::{Array2, Array3, s};
use serde::{Deserialize, Serialize};
use crate::ActivationRecord;

// Layout: [stimuli, neurons, layers]  layers: 0=early, 1=mid, 2=late
#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationTensor {
    data: Vec<f64>,
    num_stimuli: usize,
    num_neurons: usize,
    num_layers: usize,
}

impl ActivationTensor {
    pub fn from_records(records: &[ActivationRecord]) -> Self {
        let n_stimuli = records.len();
        assert!(n_stimuli > 0, "cannot build tensor from empty records");
        let n_neurons = records[0].early.len();
        let n_layers = 3;
        let mut data = vec![0.0_f64; n_stimuli * n_neurons * n_layers];

        for (si, rec) in records.iter().enumerate() {
            for ni in 0..n_neurons {
                data[si * n_neurons * n_layers + ni * n_layers + 0] = rec.early[ni];
                data[si * n_neurons * n_layers + ni * n_layers + 1] = rec.mid[ni];
                data[si * n_neurons * n_layers + ni * n_layers + 2] = rec.late[ni];
            }
        }

        Self { data, num_stimuli: n_stimuli, num_neurons: n_neurons, num_layers: n_layers }
    }

    pub fn num_stimuli(&self) -> usize { self.num_stimuli }
    pub fn num_neurons(&self) -> usize { self.num_neurons }
    pub fn num_layers(&self) -> usize { self.num_layers }

    pub fn as_array3(&self) -> Array3<f64> {
        Array3::from_shape_vec(
            (self.num_stimuli, self.num_neurons, self.num_layers),
            self.data.clone(),
        ).unwrap()
    }

    pub fn layer_matrix(&self, layer: usize) -> Array2<f64> {
        let arr3 = self.as_array3();
        arr3.slice(s![.., .., layer]).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActivationRecord, ContentType};

    fn make_record(id: usize, ct: ContentType, val: f64) -> ActivationRecord {
        ActivationRecord {
            stimulus_id: id,
            content_type: ct,
            early: vec![val; 8],
            mid: vec![val * 1.1; 8],
            late: vec![val * 1.2; 8],
        }
    }

    #[test]
    fn tensor_shape_is_correct() {
        let records = vec![
            make_record(0, ContentType::Audio, 0.1),
            make_record(1, ContentType::Audio, 0.2),
        ];
        let tensor = ActivationTensor::from_records(&records);
        assert_eq!(tensor.num_stimuli(), 2);
        assert_eq!(tensor.num_neurons(), 8);
        assert_eq!(tensor.num_layers(), 3);
    }

    #[test]
    fn tensor_round_trips_bincode() {
        let records = vec![make_record(0, ContentType::Factual, 0.5)];
        let tensor = ActivationTensor::from_records(&records);
        let bytes = bincode::serialize(&tensor).unwrap();
        let restored: ActivationTensor = bincode::deserialize(&bytes).unwrap();
        assert_eq!(tensor.num_stimuli(), restored.num_stimuli());
    }

    #[test]
    fn get_layer_returns_correct_slice() {
        let records = vec![make_record(0, ContentType::Narrative, 1.0)];
        let tensor = ActivationTensor::from_records(&records);
        let early = tensor.layer_matrix(0);
        assert_eq!(early.nrows(), 1);
        assert_eq!(early.ncols(), 8);
        approx::assert_abs_diff_eq!(early[[0, 0]], 1.0, epsilon = 1e-9);
    }
}
