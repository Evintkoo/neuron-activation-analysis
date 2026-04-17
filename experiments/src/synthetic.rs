use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use crate::{ContentType, Stimulus};

pub struct SyntheticGenerator {
    base_seed: u64,
    dim: usize,
}

impl SyntheticGenerator {
    pub fn new(base_seed: u64, dim: usize) -> Self {
        Self { base_seed, dim }
    }

    pub fn generate(&self, ct: ContentType, count: usize) -> Vec<Stimulus> {
        let type_offset = ContentType::all()
            .iter()
            .position(|&c| c == ct)
            .unwrap() as u64;
        let seed = self.base_seed.wrapping_add(type_offset * 1_000_003);
        let mut rng = SmallRng::seed_from_u64(seed);

        (0..count)
            .map(|i| {
                let vector: Vec<f64> = (0..self.dim)
                    .map(|_| rng.gen_range(-1.0_f64..1.0))
                    .collect();
                Stimulus::new(ct, vector, i)
            })
            .collect()
    }

    pub fn generate_all(&self, count_per_type: usize) -> Vec<Stimulus> {
        ContentType::all()
            .iter()
            .flat_map(|&ct| self.generate(ct, count_per_type))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContentType;

    #[test]
    fn generates_correct_count_per_type() {
        let gen = SyntheticGenerator::new(42, 16);
        let set = gen.generate(ContentType::Narrative, 50);
        assert_eq!(set.len(), 50);
    }

    #[test]
    fn same_seed_same_output() {
        let gen1 = SyntheticGenerator::new(0, 16);
        let gen2 = SyntheticGenerator::new(0, 16);
        let s1 = gen1.generate(ContentType::Audio, 5);
        let s2 = gen2.generate(ContentType::Audio, 5);
        assert_eq!(s1[0].vector, s2[0].vector);
    }

    #[test]
    fn different_types_different_vectors() {
        let gen = SyntheticGenerator::new(1, 16);
        let text = gen.generate(ContentType::TextVerbal, 1);
        let image = gen.generate(ContentType::ImageVisual, 1);
        assert_ne!(text[0].vector, image[0].vector);
    }

    #[test]
    fn vector_dimension_matches() {
        let dim = 32;
        let gen = SyntheticGenerator::new(7, dim);
        let set = gen.generate(ContentType::Factual, 3);
        for s in &set {
            assert_eq!(s.vector.len(), dim);
        }
    }
}
