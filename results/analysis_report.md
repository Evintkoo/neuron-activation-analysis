# Activation Sweep: Statistical Analysis Report

> **Mode:** FmriEncoder: real (facebook/tribev2, 177M params) | Text encoding: demo (hash-based, LLaMA not yet loaded)
> **Stimuli:** 162 | **Content types:** 13 | **Regions:** 6

## 1. Descriptive Statistics by Content Type (Global Mean Activation)

| Content Type | N | Mean | SD | SE | 95% CI | Min | Max |
|---|---|---|---|---|---|---|---|
| TextVerbal | 12 | -0.00204 | 0.00066 | 0.00019 | [-0.00243, -0.00167] | -0.00316 | -0.00112 |
| Factual | 14 | -0.00204 | 0.00085 | 0.00023 | [-0.00248, -0.00162] | -0.00389 | -0.00080 |
| AudioText | 12 | -0.00206 | 0.00069 | 0.00020 | [-0.00247, -0.00168] | -0.00314 | -0.00101 |
| ThreatSafety | 14 | -0.00215 | 0.00079 | 0.00021 | [-0.00256, -0.00178] | -0.00422 | -0.00105 |
| Novelty | 12 | -0.00218 | 0.00079 | 0.00023 | [-0.00264, -0.00175] | -0.00346 | -0.00089 |
| Emotional | 12 | -0.00229 | 0.00101 | 0.00029 | [-0.00292, -0.00175] | -0.00470 | -0.00076 |
| Reward | 12 | -0.00241 | 0.00059 | 0.00017 | [-0.00278, -0.00210] | -0.00365 | -0.00143 |
| Abstract | 12 | -0.00248 | 0.00107 | 0.00031 | [-0.00314, -0.00192] | -0.00484 | -0.00109 |
| ImageVisual | 12 | -0.00251 | 0.00094 | 0.00027 | [-0.00308, -0.00199] | -0.00436 | -0.00114 |
| Narrative | 14 | -0.00254 | 0.00074 | 0.00020 | [-0.00291, -0.00217] | -0.00402 | -0.00129 |
| Multimodal | 12 | -0.00275 | 0.00049 | 0.00014 | [-0.00304, -0.00247] | -0.00360 | -0.00176 |
| Spatial | 12 | -0.00290 | 0.00116 | 0.00033 | [-0.00362, -0.00229] | -0.00537 | -0.00152 |
| Social | 12 | -0.00298 | 0.00091 | 0.00026 | [-0.00352, -0.00246] | -0.00434 | -0.00149 |

## 2. One-Way ANOVA: Global Mean ~ Content Type

| Statistic | Value |
|---|---|
| F (12, 149) | 1.6716 |
| p-value | 0.078476 |
| η² (effect size) | 0.1186 |
| Result | ❌ not significant at α=0.05 |

### 2.1 Per-Region ANOVA Results

| Region | F | p-value | η² | Significant? |
|---|---|---|---|---|
| Visual | 1.8730 | 0.041968 | 0.1311 | ✅ |
| Auditory | 1.9932 | 0.028481 | 0.1383 | ✅ |
| Language | 1.8340 | 0.047491 | 0.1287 | ✅ |
| Prefrontal | 2.0173 | 0.026331 | 0.1398 | ✅ |
| Motor | 1.8339 | 0.047504 | 0.1287 | ✅ |
| Parietal | 1.8104 | 0.051161 | 0.1272 | ❌ |

## 3. Post-Hoc Pairwise Effect Sizes (Cohen's d, Top 10 Pairs)

Cohen's d interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large

| Pair | Cohen's d | Magnitude |
|---|---|---|
| Multimodal vs TextVerbal | -1.2264 | large |
| Social vs TextVerbal | -1.1910 | large |
| AudioText vs Multimodal | +1.1500 | large |
| AudioText vs Social | +1.1432 | large |
| Factual vs Social | +1.0719 | large |
| Factual vs Multimodal | +0.9973 | large |
| Social vs ThreatSafety | -0.9874 | large |
| Novelty vs Social | +0.9420 | large |
| Spatial vs TextVerbal | -0.9183 | large |
| Multimodal vs ThreatSafety | -0.8985 | large |

> Bonferroni-corrected α for 78 pairs: 0.000641

## 4. Region Activation Profile by Content Type

| Content Type | Visual | Auditory | Language | Prefrontal | Motor | Parietal |
|---|---|---|---|---|---|---|
| TextVerbal | +0.0222 | +0.0393 | +0.0357 | -0.0380 | -0.0396 | -0.0237 |
| Factual | +0.0191 | +0.0350 | +0.0322 | -0.0328 | -0.0345 | -0.0227 |
| AudioText | +0.0204 | +0.0337 | +0.0388 | -0.0358 | -0.0400 | -0.0216 |
| ThreatSafety | +0.0263 | +0.0392 | +0.0406 | -0.0409 | -0.0419 | -0.0283 |
| Novelty | +0.0199 | +0.0343 | +0.0361 | -0.0356 | -0.0356 | -0.0233 |
| Emotional | +0.0135 | +0.0290 | +0.0295 | -0.0266 | -0.0343 | -0.0144 |
| Reward | +0.0218 | +0.0366 | +0.0398 | -0.0385 | -0.0412 | -0.0234 |
| Abstract | +0.0403 | +0.0580 | +0.0592 | -0.0623 | -0.0626 | -0.0399 |
| ImageVisual | +0.0250 | +0.0383 | +0.0444 | -0.0438 | -0.0448 | -0.0244 |
| Narrative | +0.0290 | +0.0446 | +0.0504 | -0.0513 | -0.0496 | -0.0290 |
| Multimodal | +0.0224 | +0.0354 | +0.0382 | -0.0389 | -0.0396 | -0.0220 |
| Spatial | +0.0265 | +0.0415 | +0.0437 | -0.0483 | -0.0449 | -0.0236 |
| Social | +0.0345 | +0.0485 | +0.0515 | -0.0553 | -0.0535 | -0.0321 |

## 5. Pearson Correlations: Region Activation vs Global Mean

| Region | r | p-value | Interpretation |
|---|---|---|---|
| Visual | -0.5143 | 0.000000 | moderate ✅ |
| Auditory | -0.4746 | 0.000000 | moderate ✅ |
| Language | -0.5250 | 0.000000 | moderate ✅ |
| Prefrontal | +0.5369 | 0.000000 | moderate ✅ |
| Motor | +0.5350 | 0.000000 | moderate ✅ |
| Parietal | +0.4107 | 0.000000 | moderate ✅ |

## 6. Content-Type Ranking by Region (Highest → Lowest Relative Activation)

**Visual:** Abstract > Social > Narrative > Spatial > ThreatSafety > ImageVisual > Multimodal > TextVerbal > Reward > AudioText > Novelty > Factual > Emotional
**Auditory:** Abstract > Social > Narrative > Spatial > TextVerbal > ThreatSafety > ImageVisual > Reward > Multimodal > Factual > Novelty > AudioText > Emotional
**Language:** Abstract > Social > Narrative > ImageVisual > Spatial > ThreatSafety > Reward > AudioText > Multimodal > Novelty > TextVerbal > Factual > Emotional
**Prefrontal:** Emotional > Factual > Novelty > AudioText > TextVerbal > Reward > Multimodal > ThreatSafety > ImageVisual > Spatial > Narrative > Social > Abstract
**Motor:** Emotional > Factual > Novelty > Multimodal > TextVerbal > AudioText > Reward > ThreatSafety > ImageVisual > Spatial > Narrative > Social > Abstract
**Parietal:** Emotional > AudioText > Multimodal > Factual > Novelty > Reward > Spatial > TextVerbal > ImageVisual > ThreatSafety > Narrative > Social > Abstract

## 7. PCA of 6-Region Activation Profiles

PC1 explains **96.8%** of variance | PC2 explains **1.8%** | Combined: **98.6%**

**PC1 loadings** (which regions drive the first axis):
  - Visual: -0.3648
  - Auditory: -0.3869
  - Language: -0.4295
  - Prefrontal: +0.5132
  - Motor: +0.4203
  - Parietal: +0.3044

**PC2 loadings:**
  - Visual: +0.2115
  - Auditory: +0.4068
  - Language: -0.4429
  - Prefrontal: +0.2757
  - Motor: +0.2559
  - Parietal: -0.6724

**Content-type positions in PC1×PC2 space:**

| Content Type | PC1 | PC2 |
|---|---|---|
| Abstract | -0.0433 | +0.0023 |
| AudioText | +0.0107 | -0.0019 |
| Emotional | +0.0284 | -0.0020 |
| Factual | +0.0170 | +0.0042 |
| ImageVisual | -0.0021 | -0.0031 |
| Multimodal | +0.0081 | -0.0010 |
| Narrative | -0.0159 | -0.0025 |
| Novelty | +0.0132 | +0.0017 |
| Reward | +0.0062 | -0.0008 |
| Social | -0.0245 | -0.0003 |
| Spatial | -0.0057 | -0.0030 |
| TextVerbal | +0.0076 | +0.0030 |
| ThreatSafety | +0.0002 | +0.0034 |

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
| 1 | s3_010 | Emotional | -0.00076 |
| 2 | s4_009 | Factual | -0.00080 |
| 3 | s3_002 | Emotional | -0.00087 |
| 4 | b4_008 | Novelty | -0.00089 |
| 5 | s4_cp_active | Factual | -0.00095 |

### Bottom 5 Lowest Global Activation

| Rank | ID | Content Type | Global Mean |
|---|---|---|---|
| 1 | s5_006 | Spatial | -0.00537 |
| 2 | s2_006 | Abstract | -0.00484 |
| 3 | s3_008 | Emotional | -0.00470 |
| 4 | m2_001 | ImageVisual | -0.00436 |
| 5 | b1_008 | Social | -0.00434 |

## 10. Within-Type Activation Stability (Coefficient of Variation)

Lower CV = more consistent activation within a content type.

| Content Type | Mean | SD | CV |
|---|---|---|---|
| Multimodal | -0.00275 | 0.00049 | 0.18 |
| Reward | -0.00241 | 0.00059 | 0.25 |
| Narrative | -0.00254 | 0.00074 | 0.29 |
| Social | -0.00298 | 0.00091 | 0.31 |
| TextVerbal | -0.00204 | 0.00066 | 0.32 |
| AudioText | -0.00206 | 0.00069 | 0.34 |
| Novelty | -0.00218 | 0.00079 | 0.36 |
| ThreatSafety | -0.00215 | 0.00079 | 0.37 |
| ImageVisual | -0.00251 | 0.00094 | 0.38 |
| Spatial | -0.00290 | 0.00116 | 0.40 |
| Factual | -0.00204 | 0.00085 | 0.42 |
| Abstract | -0.00248 | 0.00107 | 0.43 |
| Emotional | -0.00229 | 0.00101 | 0.44 |

## 11. Theory-Relative Interpretation (Demo Mode)

⚠️ All results are from keyword-aware SHA256 mock activations. The following interpretations
are structural/methodological — semantic validity requires real FmriEncoder weights.

**Global mean ranking (top 3):** TextVerbal, Factual, AudioText
  - GWT/FEP predict: ThreatSafety, Novelty in top 3
  - Observed: ❌ not confirmed (demo mode)

**Prefrontal activation ranking (top 3):** Emotional, Factual, Novelty
  - GWT predicts: ThreatSafety highest ignition (broadest activation)
  - Observed: ❌ not confirmed (demo mode)

**Language region ranking (top 3):** Abstract, Social, Narrative
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

*Generated from 162 stimuli across 13 content types*