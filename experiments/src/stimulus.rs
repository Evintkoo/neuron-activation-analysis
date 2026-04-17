use serde::{Deserialize, Serialize};
use crate::ContentType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stimulus {
    pub id: usize,
    pub content_type: ContentType,
    pub vector: Vec<f64>,
}

impl Stimulus {
    pub fn new(content_type: ContentType, vector: Vec<f64>, id: usize) -> Self {
        Self { id, content_type, vector }
    }
}

#[derive(Debug, Default, Clone)]
pub struct StimulusSet {
    stimuli: Vec<Stimulus>,
}

impl StimulusSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, s: Stimulus) {
        self.stimuli.push(s);
    }

    pub fn by_type(&self, ct: ContentType) -> Vec<&Stimulus> {
        self.stimuli.iter().filter(|s| s.content_type == ct).collect()
    }

    pub fn all(&self) -> &[Stimulus] {
        &self.stimuli
    }

    pub fn len(&self) -> usize {
        self.stimuli.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stimuli.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContentType;

    #[test]
    fn stimulus_stores_vector_and_type() {
        let v = vec![0.1_f64, 0.2, 0.3];
        let s = Stimulus::new(ContentType::Narrative, v.clone(), 0);
        assert_eq!(s.content_type, ContentType::Narrative);
        assert_eq!(s.vector, v);
        assert_eq!(s.id, 0);
    }

    #[test]
    fn stimulus_set_groups_by_content_type() {
        let mut set = StimulusSet::new();
        set.push(Stimulus::new(ContentType::Audio, vec![1.0], 0));
        set.push(Stimulus::new(ContentType::Audio, vec![2.0], 1));
        set.push(Stimulus::new(ContentType::Factual, vec![3.0], 2));

        let audio = set.by_type(ContentType::Audio);
        assert_eq!(audio.len(), 2);

        let factual = set.by_type(ContentType::Factual);
        assert_eq!(factual.len(), 1);
    }

    #[test]
    fn stimulus_round_trips_bincode() {
        let s = Stimulus::new(ContentType::Spatial, vec![0.5, 0.6], 42);
        let bytes = bincode::serialize(&s).unwrap();
        let s2: Stimulus = bincode::deserialize(&bytes).unwrap();
        assert_eq!(s.id, s2.id);
        assert_eq!(s.content_type, s2.content_type);
    }
}
