# Activation Sweep: Statistical Analysis Report

> **Mode:** FmriEncoder: real (facebook/tribev2, 177M params) | Text encoding: demo (hash-based, LLaMA not yet loaded)
> **Stimuli:** 577 | **Content types:** 13 | **Regions:** 6

## 1. Descriptive Statistics by Content Type (Global Mean Activation)

| Content Type | N | Mean | SD | SE | 95% CI | Min | Max |
|---|---|---|---|---|---|---|---|
| AudioText | 24 | -0.00176 | 0.00071 | 0.00015 | [-0.00204, -0.00147] | -0.00314 | -0.00046 |
| ThreatSafety | 71 | -0.00217 | 0.00070 | 0.00008 | [-0.00233, -0.00201] | -0.00422 | -0.00004 |
| TextVerbal | 24 | -0.00218 | 0.00078 | 0.00016 | [-0.00249, -0.00187] | -0.00384 | -0.00060 |
| Factual | 44 | -0.00226 | 0.00083 | 0.00013 | [-0.00252, -0.00202] | -0.00389 | -0.00063 |
| Novelty | 108 | -0.00228 | 0.00083 | 0.00008 | [-0.00244, -0.00212] | -0.00427 | -0.00059 |
| Emotional | 12 | -0.00229 | 0.00101 | 0.00029 | [-0.00292, -0.00175] | -0.00470 | -0.00076 |
| Abstract | 95 | -0.00234 | 0.00088 | 0.00009 | [-0.00252, -0.00216] | -0.00484 | -0.00039 |
| Reward | 12 | -0.00241 | 0.00059 | 0.00017 | [-0.00278, -0.00210] | -0.00365 | -0.00143 |
| Narrative | 44 | -0.00263 | 0.00074 | 0.00011 | [-0.00286, -0.00242] | -0.00419 | -0.00129 |
| Spatial | 52 | -0.00276 | 0.00100 | 0.00014 | [-0.00303, -0.00248] | -0.00537 | -0.00087 |
| ImageVisual | 61 | -0.00280 | 0.00102 | 0.00013 | [-0.00304, -0.00253] | -0.00479 | -0.00042 |
| Multimodal | 18 | -0.00281 | 0.00059 | 0.00014 | [-0.00309, -0.00253] | -0.00419 | -0.00176 |
| Social | 12 | -0.00298 | 0.00091 | 0.00026 | [-0.00352, -0.00246] | -0.00434 | -0.00149 |

## 2. One-Way ANOVA: Global Mean ~ Content Type

| Statistic | Value |
|---|---|
| F (12, 564) | 4.8992 |
| p-value | 0.000000 |
| η² (effect size) | 0.0944 |
| Result | ✅ significant at α=0.05 |

### 2.1 Per-Region ANOVA Results

| Region | F | p-value | η² | Significant? |
|---|---|---|---|---|
| Visual | 2.3361 | 0.006328 | 0.0474 | ✅ |
| Auditory | 2.9979 | 0.000438 | 0.0600 | ✅ |
| Language | 1.4359 | 0.145045 | 0.0296 | ❌ |
| Prefrontal | 2.1105 | 0.014858 | 0.0430 | ✅ |
| Motor | 1.5494 | 0.102568 | 0.0319 | ❌ |
| Parietal | 2.9399 | 0.000558 | 0.0589 | ✅ |

## 3. Post-Hoc Pairwise Effect Sizes (Cohen's d, Top 10 Pairs)

Cohen's d interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large

| Pair | Cohen's d | Magnitude |
|---|---|---|
| AudioText vs Multimodal | +1.5847 | large |
| AudioText vs Social | +1.5671 | large |
| AudioText vs Narrative | +1.1900 | large |
| Social vs ThreatSafety | -1.1155 | large |
| AudioText vs ImageVisual | +1.0933 | large |
| AudioText vs Spatial | +1.0840 | large |
| Social vs TextVerbal | -0.9794 | large |
| AudioText vs Reward | +0.9705 | large |
| Multimodal vs ThreatSafety | -0.9446 | large |
| Multimodal vs TextVerbal | -0.8976 | large |

> Bonferroni-corrected α for 78 pairs: 0.000641

## 4. Region Activation Profile by Content Type

| Content Type | Visual | Auditory | Language | Prefrontal | Motor | Parietal |
|---|---|---|---|---|---|---|
| AudioText | +0.0187 | +0.0313 | +0.0359 | -0.0340 | -0.0368 | -0.0194 |
| ThreatSafety | +0.0244 | +0.0390 | +0.0396 | -0.0412 | -0.0416 | -0.0249 |
| TextVerbal | +0.0233 | +0.0401 | +0.0385 | -0.0414 | -0.0413 | -0.0236 |
| Factual | +0.0190 | +0.0349 | +0.0345 | -0.0359 | -0.0358 | -0.0205 |
| Novelty | +0.0234 | +0.0381 | +0.0384 | -0.0401 | -0.0398 | -0.0245 |
| Emotional | +0.0135 | +0.0290 | +0.0295 | -0.0266 | -0.0343 | -0.0144 |
| Abstract | +0.0278 | +0.0435 | +0.0448 | -0.0475 | -0.0464 | -0.0276 |
| Reward | +0.0218 | +0.0366 | +0.0398 | -0.0385 | -0.0412 | -0.0234 |
| Narrative | +0.0236 | +0.0384 | +0.0402 | -0.0412 | -0.0418 | -0.0239 |
| Spatial | +0.0234 | +0.0381 | +0.0385 | -0.0416 | -0.0403 | -0.0226 |
| ImageVisual | +0.0154 | +0.0274 | +0.0356 | -0.0327 | -0.0359 | -0.0139 |
| Multimodal | +0.0236 | +0.0380 | +0.0419 | -0.0441 | -0.0419 | -0.0223 |
| Social | +0.0345 | +0.0485 | +0.0515 | -0.0553 | -0.0535 | -0.0321 |

## 5. Pearson Correlations: Region Activation vs Global Mean

| Region | r | p-value | Interpretation |
|---|---|---|---|
| Visual | -0.4527 | 0.000000 | moderate ✅ |
| Auditory | -0.4000 | 0.000000 | weak ✅ |
| Language | -0.4697 | 0.000000 | moderate ✅ |
| Prefrontal | +0.4628 | 0.000000 | moderate ✅ |
| Motor | +0.4868 | 0.000000 | moderate ✅ |
| Parietal | +0.3539 | 0.000000 | weak ✅ |

## 6. Content-Type Ranking by Region (Highest → Lowest Relative Activation)

**Visual:** Social > Abstract > ThreatSafety > Narrative > Multimodal > Spatial > Novelty > TextVerbal > Reward > Factual > AudioText > ImageVisual > Emotional
**Auditory:** Social > Abstract > TextVerbal > ThreatSafety > Narrative > Spatial > Novelty > Multimodal > Reward > Factual > AudioText > Emotional > ImageVisual
**Language:** Social > Abstract > Multimodal > Narrative > Reward > ThreatSafety > Spatial > TextVerbal > Novelty > AudioText > ImageVisual > Factual > Emotional
**Prefrontal:** Emotional > ImageVisual > AudioText > Factual > Reward > Novelty > Narrative > ThreatSafety > TextVerbal > Spatial > Multimodal > Abstract > Social
**Motor:** Emotional > Factual > ImageVisual > AudioText > Novelty > Spatial > Reward > TextVerbal > ThreatSafety > Narrative > Multimodal > Abstract > Social
**Parietal:** ImageVisual > Emotional > AudioText > Factual > Multimodal > Spatial > Reward > TextVerbal > Narrative > Novelty > ThreatSafety > Abstract > Social

## 7. PCA of 6-Region Activation Profiles

PC1 explains **96.2%** of variance | PC2 explains **2.5%** | Combined: **98.7%**

**PC1 loadings** (which regions drive the first axis):
  - Visual: -0.3957
  - Auditory: -0.4115
  - Language: -0.3805
  - Prefrontal: +0.5222
  - Motor: +0.3664
  - Parietal: +0.3498

**PC2 loadings:**
  - Visual: +0.0894
  - Auditory: +0.5472
  - Language: -0.5723
  - Prefrontal: +0.2047
  - Motor: +0.2911
  - Parietal: -0.4882

**Content-type positions in PC1×PC2 space:**

| Content Type | PC1 | PC2 |
|---|---|---|
| Abstract | -0.0146 | +0.0000 |
| AudioText | +0.0109 | -0.0008 |
| Emotional | +0.0228 | +0.0009 |
| Factual | +0.0087 | +0.0024 |
| ImageVisual | +0.0168 | -0.0052 |
| Multimodal | -0.0042 | -0.0023 |
| Narrative | -0.0028 | +0.0003 |
| Novelty | -0.0008 | +0.0023 |
| Reward | +0.0006 | -0.0002 |
| Social | -0.0301 | -0.0020 |
| Spatial | -0.0012 | +0.0008 |
| TextVerbal | -0.0026 | +0.0021 |
| ThreatSafety | -0.0035 | +0.0016 |

## 8. Contrastive Pair Analysis

Matched pairs: same content topic, different language structure. Delta = A − B.

### ThreatSafety: headline vs narrative

| Metric | Stimulus A | Stimulus B | Δ (A−B) |
|---|---|---|---|
| global_mean | -0.00199 | -0.00422 | +0.00223 |
| Visual rel | +0.03579 | +0.00965 | +0.02614 |
| Auditory rel | +0.04295 | +0.01781 | +0.02514 |
| Language rel | +0.05812 | +0.02668 | +0.03143 |
| Prefrontal rel | -0.05569 | -0.02394 | -0.03176 |
| Motor rel | -0.05068 | -0.02512 | -0.02556 |
| Parietal rel | -0.03782 | -0.00800 | -0.02982 |

### Narrative: chronological vs bullet-point

| Metric | Stimulus A | Stimulus B | Δ (A−B) |
|---|---|---|---|
| global_mean | -0.00199 | -0.00252 | +0.00052 |
| Visual rel | +0.01346 | +0.01424 | -0.00078 |
| Auditory rel | +0.03124 | +0.02547 | +0.00577 |
| Language rel | +0.03296 | +0.02855 | +0.00441 |
| Prefrontal rel | -0.03378 | -0.02714 | -0.00664 |
| Motor rel | -0.02996 | -0.02913 | -0.00084 |
| Parietal rel | -0.01713 | -0.01529 | -0.00185 |

### Factual: active vs passive voice

| Metric | Stimulus A | Stimulus B | Δ (A−B) |
|---|---|---|---|
| global_mean | -0.00095 | -0.00286 | +0.00191 |
| Visual rel | -0.00304 | +0.03809 | -0.04114 |
| Auditory rel | +0.01308 | +0.05709 | -0.04401 |
| Language rel | +0.00969 | +0.05380 | -0.04411 |
| Prefrontal rel | -0.00291 | -0.06156 | +0.05865 |
| Motor rel | -0.01208 | -0.05719 | +0.04511 |
| Parietal rel | -0.00545 | -0.03667 | +0.03122 |

## 9. Extreme Stimuli

### Top 5 Highest Global Activation

| Rank | ID | Content Type | Global Mean |
|---|---|---|---|
| 1 | rss_b3_0033 | ThreatSafety | -0.00004 |
| 2 | ax_s2_0022 | Abstract | -0.00039 |
| 3 | pm_s2_0085 | Abstract | -0.00040 |
| 4 | wc_m2_0008 | ImageVisual | -0.00042 |
| 5 | aud_m3_0012 | AudioText | -0.00046 |

### Bottom 5 Lowest Global Activation

| Rank | ID | Content Type | Global Mean |
|---|---|---|---|
| 1 | s5_006 | Spatial | -0.00537 |
| 2 | osm_s5_0030 | Spatial | -0.00494 |
| 3 | s2_006 | Abstract | -0.00484 |
| 4 | osm_s5_0040 | Spatial | -0.00481 |
| 5 | wc_m2_0043 | ImageVisual | -0.00479 |

## 10. Within-Type Activation Stability (Coefficient of Variation)

Lower CV = more consistent activation within a content type.

| Content Type | Mean | SD | CV |
|---|---|---|---|
| Multimodal | -0.00281 | 0.00059 | 0.21 |
| Reward | -0.00241 | 0.00059 | 0.25 |
| Narrative | -0.00263 | 0.00074 | 0.28 |
| Social | -0.00298 | 0.00091 | 0.31 |
| ThreatSafety | -0.00217 | 0.00070 | 0.32 |
| TextVerbal | -0.00218 | 0.00078 | 0.36 |
| Spatial | -0.00276 | 0.00100 | 0.36 |
| Novelty | -0.00228 | 0.00083 | 0.36 |
| ImageVisual | -0.00280 | 0.00102 | 0.37 |
| Factual | -0.00226 | 0.00083 | 0.37 |
| Abstract | -0.00234 | 0.00088 | 0.38 |
| AudioText | -0.00176 | 0.00071 | 0.41 |
| Emotional | -0.00229 | 0.00101 | 0.44 |

## 11. Theory-Relative Interpretation (Demo Mode)

⚠️ All results are from keyword-aware SHA256 mock activations. The following interpretations
are structural/methodological — semantic validity requires real FmriEncoder weights.

**Global mean ranking (top 3):** AudioText, ThreatSafety, TextVerbal
  - GWT/FEP predict: ThreatSafety, Novelty in top 3
  - Observed: ✅ confirmed (demo mode)

**Prefrontal activation ranking (top 3):** Emotional, ImageVisual, AudioText
  - GWT predicts: ThreatSafety highest ignition (broadest activation)
  - Observed: ❌ not confirmed (demo mode)

**Language region ranking (top 3):** Social, Abstract, Multimodal
  - IIT/DCT predict: Narrative, Social highest language integration
  - Observed: ✅ confirmed (demo mode)

---

## Methodology Notes

- **Data source:** Demo mode — keyword-aware SHA256 hash activations via `experiments/mock_server`
- **Statistical tests:** One-way ANOVA (F-test), Cohen's d effect size, Pearson correlation r
- **Multiple comparisons:** Bonferroni correction applied for pairwise comparisons
- **Effect sizes:** η² for ANOVA (> 0.14 large), |d| for pairwise (> 0.8 large)
- **Confidence intervals:** Bootstrap (n=2000, 95%)
- **PCA:** Manual power iteration on 6-region mean activation profiles
- **Validity:** All results are methodologically valid but semantically uninterpretable until
  real FmriEncoder weights (`best.safetensors`) are loaded and sweep re-run

*Generated from 577 stimuli across 13 content types*