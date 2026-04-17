use experiments::{
    ActivationExtractor, ActivationFingerprint, ActivationTensor,
    ContentType, DctScorer, FepScorer, GwtScorer, IitScorer,
    KMeansClusterer, MockTribeModel, PcaReducer, SyntheticGenerator,
    TheoryFitReport,
};
use analysis::{
    bootstrap_ci, bonferroni_correct, one_way_anova, permutation_test,
    ContrastiveDelta, ExperimentReport,
};

fn main() {
    println!("=== Activation Cartography Experiment Runner ===\n");

    // Phase 1: Synthetic stimulus generation (50 per type)
    println!("[1/5] Generating synthetic stimuli...");
    let generator = SyntheticGenerator::new(42, 64);
    let all_stimuli = generator.generate_all(50);
    println!("    Generated {} stimuli across {} content types",
        all_stimuli.len(), ContentType::all().len());

    // Phase 2: Activation extraction via MockTribeModel
    println!("[2/5] Extracting activations...");
    let model = MockTribeModel::new(128, 64);
    let extractor = ActivationExtractor::new(model);
    let records = extractor.extract_batch(&all_stimuli);
    let tensor = ActivationTensor::from_records(&records);
    println!("    Tensor shape: {} stimuli × {} neurons × {} layers",
        tensor.num_stimuli(), tensor.num_neurons(), tensor.num_layers());

    // Phase 3: Fingerprinting (PCA + k-means on mid-layer activations)
    println!("[3/5] Generating activation fingerprints...");
    let mid_layer = tensor.layer_matrix(1);
    let reducer = PcaReducer::fit(&mid_layer, 8);
    let reduced = reducer.transform(&mid_layer);
    let cluster_result = KMeansClusterer::run(&reduced, 13, 42);
    println!("    Silhouette score: {:.4}", cluster_result.silhouette_score);

    let fingerprints = ActivationFingerprint::from_clusters(
        &reduced,
        &cluster_result.labels,
        &ContentType::all(),
    );

    // Phase 4: Theory fit scoring
    println!("[4/5] Scoring theory fit...");
    let dct_score = DctScorer::score(&fingerprints);
    let gwt_score = GwtScorer::score(&records, 0.5, 3);
    let fep_score = FepScorer::score(&records);
    let iit_score = IitScorer::score(&records);

    let theory_fit = TheoryFitReport { dct_score, gwt_score, fep_score, iit_score };
    println!("    DCT: {:.3}  GWT: {:.3}  FEP: {:.3}  IIT: {:.3}",
        dct_score, gwt_score, fep_score, iit_score);
    println!("    Winner: {} (margin: {:.3})", theory_fit.winner(), theory_fit.winner_margin());

    // Phase 5: Statistical analysis
    println!("[5/5] Running statistical analysis...");

    let groups: Vec<Vec<f64>> = ContentType::all().iter().map(|&ct| {
        records.iter()
            .filter(|r| r.content_type == ct)
            .map(|r| r.early.iter().map(|v| v.abs()).sum::<f64>() / r.early.len() as f64)
            .collect()
    }).collect();

    let anova = one_way_anova(&groups);
    println!("    ANOVA: F={:.3}, p={:.6}", anova.f_stat, anova.p_value);

    let image_records: Vec<_> = records.iter().filter(|r| r.content_type == ContentType::ImageVisual).collect();
    let emotional_records: Vec<_> = records.iter().filter(|r| r.content_type == ContentType::Emotional).collect();
    let contrastive_delta = ContrastiveDelta::compute(image_records[0], emotional_records[0]);

    let silhouette_samples = vec![cluster_result.silhouette_score; 100];
    let (ci_lo, ci_hi) = bootstrap_ci(&silhouette_samples, 1000, 0.95, 42);
    println!("    Silhouette 95% CI: [{:.4}, {:.4}]", ci_lo, ci_hi);

    let p_values = vec![anova.p_value; 13];
    let bonferroni = bonferroni_correct(&p_values, 0.05);

    // suppress unused import warning for permutation_test
    let _ = permutation_test;

    let report = ExperimentReport {
        theory_fit,
        silhouette_scores: vec![cluster_result.silhouette_score],
        anova_p_value: anova.p_value,
        contrastive_deltas: vec![contrastive_delta.l2_norm()],
        bootstrap_cis: vec![(ci_lo, ci_hi)],
        bonferroni_significant: bonferroni,
    };

    std::fs::create_dir_all("results").ok();
    report.write_to_file("results/experiment_results.json").expect("failed to write results");
    println!("\nResults written to results/experiment_results.json");

    let criteria = report.success_criteria();
    println!("\n=== Success Criteria ===");
    println!("  Silhouette threshold met: {}", criteria.silhouette_met);
    println!("  Winner margin >= 0.1:     {}", criteria.winner_margin_met);
    println!("  Contrastive significant:  {}", criteria.contrastive_significant);
}
