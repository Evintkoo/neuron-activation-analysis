use crate::model::{ActivationProbes, TribeModel};

pub struct MockTribeModel {
    neuron_count: usize,
    input_dim: usize,
}

impl MockTribeModel {
    pub fn new(neuron_count: usize, input_dim: usize) -> Self {
        Self { neuron_count, input_dim }
    }
}

impl TribeModel for MockTribeModel {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn neuron_count(&self) -> usize {
        self.neuron_count
    }

    fn forward(&self, input: &[f64]) -> ActivationProbes {
        let n = self.neuron_count;
        let sum: f64 = input.iter().sum();

        let early: Vec<f64> = (0..n)
            .map(|i| (sum * (i as f64 + 1.0)).tanh())
            .collect();
        let mid: Vec<f64> = (0..n)
            .map(|i| (sum * (i as f64 + 0.5) * 1.1).tanh())
            .collect();
        let late: Vec<f64> = (0..n)
            .map(|i| (sum * (i as f64 + 0.1) * 1.3).tanh())
            .collect();

        ActivationProbes { early, mid, late }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TribeModel;

    #[test]
    fn mock_returns_correct_probe_dimensions() {
        let model = MockTribeModel::new(64, 3);
        let input = vec![0.0_f64; model.input_dim()];
        let probes = model.forward(&input);
        assert_eq!(probes.early.len(), 64);
        assert_eq!(probes.mid.len(), 64);
        assert_eq!(probes.late.len(), 64);
    }

    #[test]
    fn mock_same_input_same_output() {
        let model = MockTribeModel::new(32, 3);
        let input = vec![0.5_f64; model.input_dim()];
        let p1 = model.forward(&input);
        let p2 = model.forward(&input);
        assert_eq!(p1.early, p2.early);
    }

    #[test]
    fn mock_different_inputs_different_outputs() {
        let model = MockTribeModel::new(32, 3);
        let input_a = vec![0.1_f64; model.input_dim()];
        let input_b = vec![0.9_f64; model.input_dim()];
        let pa = model.forward(&input_a);
        let pb = model.forward(&input_b);
        assert_ne!(pa.early, pb.early);
    }
}
