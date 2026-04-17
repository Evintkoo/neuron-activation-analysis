pub mod contrastive;
pub use contrastive::ContrastiveDelta;

pub mod anova;
pub use anova::{one_way_anova, AnovaResult};

pub mod permutation;
pub use permutation::permutation_test;

pub mod bootstrap;
pub use bootstrap::bootstrap_ci;

pub mod bonferroni;
pub use bonferroni::bonferroni_correct;
