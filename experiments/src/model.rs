pub struct ActivationProbes {
    pub early: Vec<f64>,
    pub mid: Vec<f64>,
    pub late: Vec<f64>,
}

pub trait TribeModel: Send + Sync {
    fn input_dim(&self) -> usize;
    fn neuron_count(&self) -> usize;
    fn forward(&self, input: &[f64]) -> ActivationProbes;
}
