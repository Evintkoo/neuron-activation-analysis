use rayon::prelude::*;
use crate::{ContentType, Stimulus, TribeModel};

#[derive(Debug, Clone)]
pub struct ActivationRecord {
    pub stimulus_id: usize,
    pub content_type: ContentType,
    pub early: Vec<f64>,
    pub mid: Vec<f64>,
    pub late: Vec<f64>,
}

pub struct ActivationExtractor<M: TribeModel> {
    model: M,
}

impl<M: TribeModel> ActivationExtractor<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }

    pub fn extract(&self, stimulus: &Stimulus) -> ActivationRecord {
        let probes = self.model.forward(&stimulus.vector);
        ActivationRecord {
            stimulus_id: stimulus.id,
            content_type: stimulus.content_type,
            early: probes.early,
            mid: probes.mid,
            late: probes.late,
        }
    }

    pub fn extract_batch(&self, stimuli: &[Stimulus]) -> Vec<ActivationRecord> {
        stimuli.par_iter().map(|s| self.extract(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentType, MockTribeModel, SyntheticGenerator};

    fn make_extractor() -> ActivationExtractor<MockTribeModel> {
        let model = MockTribeModel::new(32, 16);
        ActivationExtractor::new(model)
    }

    #[test]
    fn extract_single_stimulus_returns_three_probes() {
        let ext = make_extractor();
        let gen = SyntheticGenerator::new(0, 16);
        let stimulus = &gen.generate(ContentType::Audio, 1)[0];
        let record = ext.extract(stimulus);
        assert_eq!(record.early.len(), 32);
        assert_eq!(record.mid.len(), 32);
        assert_eq!(record.late.len(), 32);
        assert_eq!(record.stimulus_id, 0);
        assert_eq!(record.content_type, ContentType::Audio);
    }

    #[test]
    fn extract_batch_returns_correct_count() {
        let ext = make_extractor();
        let gen = SyntheticGenerator::new(1, 16);
        let stimuli = gen.generate(ContentType::Narrative, 10);
        let records = ext.extract_batch(&stimuli);
        assert_eq!(records.len(), 10);
    }
}
