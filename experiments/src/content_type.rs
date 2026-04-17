use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    // Modality
    TextVerbal,
    ImageVisual,
    Audio,
    Multimodal,
    // Semantic/Cognitive
    Narrative,
    Abstract,
    Emotional,
    Factual,
    Spatial,
    // Social/Behavioral
    SocialInteraction,
    RewardSignal,
    ThreatSafety,
    Novelty,
}

impl ContentType {
    pub fn all() -> [ContentType; 13] {
        [
            ContentType::TextVerbal,
            ContentType::ImageVisual,
            ContentType::Audio,
            ContentType::Multimodal,
            ContentType::Narrative,
            ContentType::Abstract,
            ContentType::Emotional,
            ContentType::Factual,
            ContentType::Spatial,
            ContentType::SocialInteraction,
            ContentType::RewardSignal,
            ContentType::ThreatSafety,
            ContentType::Novelty,
        ]
    }

    pub fn label(&self) -> &'static str {
        match self {
            ContentType::TextVerbal => "Text/Verbal",
            ContentType::ImageVisual => "Image/Visual",
            ContentType::Audio => "Audio",
            ContentType::Multimodal => "Multimodal",
            ContentType::Narrative => "Narrative",
            ContentType::Abstract => "Abstract",
            ContentType::Emotional => "Emotional",
            ContentType::Factual => "Factual",
            ContentType::Spatial => "Spatial",
            ContentType::SocialInteraction => "Social Interaction",
            ContentType::RewardSignal => "Reward Signal",
            ContentType::ThreatSafety => "Threat/Safety",
            ContentType::Novelty => "Novelty",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_variants_count() {
        assert_eq!(ContentType::all().len(), 13);
    }

    #[test]
    fn round_trip_serialization() {
        let ct = ContentType::Narrative;
        let encoded = bincode::serialize(&ct).unwrap();
        let decoded: ContentType = bincode::deserialize(&encoded).unwrap();
        assert_eq!(ct, decoded);
    }

    #[test]
    fn display_is_non_empty() {
        for ct in ContentType::all() {
            assert!(!ct.label().is_empty());
        }
    }
}
