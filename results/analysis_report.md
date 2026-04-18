# Activation Sweep: Statistical Analysis Report

> **Mode:** Demo (keyword-aware SHA256 mock — re-run with real weights for semantic validity)
> **Stimuli:** 162 | **Content types:** 13 | **Regions:** 6

## 1. Descriptive Statistics by Content Type (Global Mean Activation)

| Content Type | N | Mean | SD | SE | 95% CI | Min | Max |
|---|---|---|---|---|---|---|---|
| ImageVisual | 12 | +0.00269 | 0.00565 | 0.00163 | [-0.00044, +0.00591] | -0.00426 | +0.01512 |
| Abstract | 12 | +0.00171 | 0.00429 | 0.00124 | [-0.00074, +0.00422] | -0.00452 | +0.00883 |
| Novelty | 12 | +0.00132 | 0.00350 | 0.00101 | [-0.00078, +0.00329] | -0.00513 | +0.00868 |
| ThreatSafety | 14 | +0.00102 | 0.00565 | 0.00151 | [-0.00205, +0.00377] | -0.01062 | +0.00882 |
| Narrative | 14 | +0.00031 | 0.00490 | 0.00131 | [-0.00221, +0.00274] | -0.00806 | +0.00806 |
| Reward | 12 | -0.00009 | 0.00476 | 0.00138 | [-0.00305, +0.00248] | -0.01041 | +0.00623 |
| Factual | 14 | -0.00016 | 0.00458 | 0.00122 | [-0.00245, +0.00213] | -0.00729 | +0.00988 |
| TextVerbal | 12 | -0.00026 | 0.00670 | 0.00193 | [-0.00408, +0.00358] | -0.01045 | +0.01548 |
| AudioText | 12 | -0.00046 | 0.00599 | 0.00173 | [-0.00402, +0.00273] | -0.01067 | +0.00936 |
| Multimodal | 12 | -0.00062 | 0.00505 | 0.00146 | [-0.00353, +0.00217] | -0.00849 | +0.00796 |
| Social | 12 | -0.00102 | 0.00426 | 0.00123 | [-0.00344, +0.00138] | -0.00646 | +0.00709 |
| Spatial | 12 | -0.00147 | 0.00580 | 0.00167 | [-0.00506, +0.00173] | -0.01337 | +0.00954 |
| Emotional | 12 | -0.00176 | 0.00468 | 0.00135 | [-0.00471, +0.00075] | -0.01288 | +0.00469 |

## 2. One-Way ANOVA: Global Mean ~ Content Type

| Statistic | Value |
|---|---|
| F (12, 149) | 0.7032 |
| p-value | 0.746590 |
| η² (effect size) | 0.0536 |
| Result | ❌ not significant at α=0.05 |

### 2.1 Per-Region ANOVA Results

| Region | F | p-value | η² | Significant? |
|---|---|---|---|---|
| Visual | 0.6692 | 0.778788 | 0.0511 | ❌ |
| Auditory | 0.8046 | 0.645287 | 0.0609 | ❌ |
| Language | 0.9721 | 0.478097 | 0.0726 | ❌ |
| Prefrontal | 0.6324 | 0.812069 | 0.0485 | ❌ |
| Motor | 1.2039 | 0.285416 | 0.0884 | ❌ |
| Parietal | 0.7250 | 0.725303 | 0.0552 | ❌ |

## 3. Post-Hoc Pairwise Effect Sizes (Cohen's d, Top 10 Pairs)

Cohen's d interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large

| Pair | Cohen's d | Magnitude |
|---|---|---|
| Emotional vs ImageVisual | -0.8571 | large |
| Abstract vs Emotional | +0.7732 | medium |
| Emotional vs Novelty | -0.7432 | medium |
| ImageVisual vs Social | +0.7420 | medium |
| ImageVisual vs Spatial | +0.7271 | medium |
| Abstract vs Social | +0.6403 | medium |
| Abstract vs Spatial | +0.6248 | medium |
| ImageVisual vs Multimodal | +0.6184 | medium |
| Novelty vs Social | +0.5996 | medium |
| Novelty vs Spatial | +0.5820 | medium |

> Bonferroni-corrected α for 78 pairs: 0.000641

## 4. Region Activation Profile by Content Type

| Content Type | Visual | Auditory | Language | Prefrontal | Motor | Parietal |
|---|---|---|---|---|---|---|
| ImageVisual | -0.0027 | -0.0047 | -0.0048 | -0.0005 | +0.0072 | +0.0065 |
| Abstract | +0.0000 | +0.0013 | +0.0037 | -0.0042 | -0.0038 | +0.0027 |
| Novelty | +0.0038 | -0.0024 | -0.0000 | -0.0017 | +0.0074 | -0.0071 |
| ThreatSafety | +0.0019 | +0.0031 | -0.0037 | +0.0020 | -0.0033 | +0.0001 |
| Narrative | -0.0039 | +0.0086 | +0.0003 | +0.0025 | -0.0064 | -0.0010 |
| Reward | +0.0042 | +0.0035 | -0.0006 | -0.0047 | -0.0036 | +0.0012 |
| Factual | +0.0017 | -0.0045 | -0.0010 | +0.0066 | -0.0030 | -0.0005 |
| TextVerbal | -0.0063 | -0.0029 | +0.0002 | +0.0026 | -0.0004 | +0.0071 |
| AudioText | +0.0012 | -0.0000 | +0.0022 | +0.0002 | -0.0002 | -0.0038 |
| Multimodal | -0.0050 | -0.0001 | +0.0000 | +0.0032 | +0.0006 | +0.0017 |
| Social | -0.0031 | +0.0079 | -0.0027 | -0.0010 | -0.0026 | +0.0025 |
| Spatial | +0.0012 | -0.0031 | +0.0055 | -0.0005 | -0.0011 | -0.0028 |
| Emotional | +0.0006 | +0.0042 | -0.0111 | -0.0017 | +0.0087 | +0.0011 |

## 5. Pearson Correlations: Region Activation vs Global Mean

| Region | r | p-value | Interpretation |
|---|---|---|---|
| Visual | +0.1654 | 0.033904 | weak ✅ |
| Auditory | +0.0475 | 0.547539 | weak ❌ |
| Language | +0.0010 | 0.989868 | weak ❌ |
| Prefrontal | -0.1668 | 0.032326 | weak ✅ |
| Motor | -0.0415 | 0.599268 | weak ❌ |
| Parietal | -0.0185 | 0.814950 | weak ❌ |

## 6. Content-Type Ranking by Region (Highest → Lowest Relative Activation)

**Visual:** Reward > Novelty > ThreatSafety > Factual > AudioText > Spatial > Emotional > Abstract > ImageVisual > Social > Narrative > Multimodal > TextVerbal
**Auditory:** Narrative > Social > Emotional > Reward > ThreatSafety > Abstract > AudioText > Multimodal > Novelty > TextVerbal > Spatial > Factual > ImageVisual
**Language:** Spatial > Abstract > AudioText > Narrative > TextVerbal > Multimodal > Novelty > Reward > Factual > Social > ThreatSafety > ImageVisual > Emotional
**Prefrontal:** Factual > Multimodal > TextVerbal > Narrative > ThreatSafety > AudioText > ImageVisual > Spatial > Social > Emotional > Novelty > Abstract > Reward
**Motor:** Emotional > Novelty > ImageVisual > Multimodal > AudioText > TextVerbal > Spatial > Social > Factual > ThreatSafety > Reward > Abstract > Narrative
**Parietal:** TextVerbal > ImageVisual > Abstract > Social > Multimodal > Reward > Emotional > ThreatSafety > Factual > Narrative > Spatial > AudioText > Novelty

## 7. PCA of 6-Region Activation Profiles

PC1 explains **34.5%** of variance | PC2 explains **26.5%** | Combined: **61.0%**

**PC1 loadings** (which regions drive the first axis):
  - Visual: +0.0544
  - Auditory: -0.3342
  - Language: -0.4705
  - Prefrontal: -0.0647
  - Motor: +0.8086
  - Parietal: +0.0771

**PC2 loadings:**
  - Visual: -0.3910
  - Auditory: +0.5623
  - Language: -0.5041
  - Prefrontal: +0.0098
  - Motor: -0.0834
  - Parietal: +0.5195

**Content-type positions in PC1×PC2 space:**

| Content Type | PC1 | PC2 |
|---|---|---|
| Abstract | -0.0049 | -0.0009 |
| AudioText | -0.0016 | -0.0050 |
| Emotional | +0.0109 | +0.0061 |
| Factual | -0.0009 | -0.0041 |
| ImageVisual | +0.0099 | +0.0022 |
| Multimodal | +0.0000 | +0.0013 |
| Narrative | -0.0088 | +0.0048 |
| Novelty | +0.0065 | -0.0086 |
| Reward | -0.0033 | -0.0000 |
| Social | -0.0035 | +0.0071 |
| Spatial | -0.0027 | -0.0077 |
| TextVerbal | +0.0005 | +0.0030 |
| ThreatSafety | -0.0022 | +0.0018 |

## 8. Contrastive Pair Analysis

Matched pairs: same content topic, different language structure. Delta = A − B.

### ThreatSafety: headline vs narrative

| Metric | Stimulus A | Stimulus B | Δ (A−B) |
|---|---|---|---|
| global_mean | -0.00251 | +0.00465 | -0.00716 |
| Visual rel | +0.01530 | -0.00656 | +0.02186 |
| Auditory rel | +0.00898 | -0.01227 | +0.02124 |
| Language rel | -0.03446 | +0.02228 | -0.05674 |
| Prefrontal rel | +0.03129 | -0.01955 | +0.05084 |
| Motor rel | -0.02406 | -0.00202 | -0.02204 |
| Parietal rel | +0.00340 | +0.01685 | -0.01345 |

### Narrative: chronological vs bullet-point

| Metric | Stimulus A | Stimulus B | Δ (A−B) |
|---|---|---|---|
| global_mean | -0.00627 | +0.00182 | -0.00809 |
| Visual rel | -0.00434 | -0.00342 | -0.00092 |
| Auditory rel | +0.00258 | +0.01102 | -0.00844 |
| Language rel | +0.01597 | -0.00146 | +0.01744 |
| Prefrontal rel | -0.01657 | +0.00315 | -0.01972 |
| Motor rel | +0.00700 | +0.00615 | +0.00085 |
| Parietal rel | -0.00491 | -0.01469 | +0.00978 |

### Factual: active vs passive voice

| Metric | Stimulus A | Stimulus B | Δ (A−B) |
|---|---|---|---|
| global_mean | +0.00495 | +0.00102 | +0.00393 |
| Visual rel | -0.00178 | -0.00584 | +0.00406 |
| Auditory rel | -0.00062 | -0.01962 | +0.01899 |
| Language rel | -0.01759 | +0.00693 | -0.02453 |
| Prefrontal rel | -0.00327 | +0.00858 | -0.01184 |
| Motor rel | +0.00576 | +0.00596 | -0.00020 |
| Parietal rel | +0.02024 | +0.00275 | +0.01749 |

## 9. Extreme Stimuli

### Top 5 Highest Global Activation

| Rank | ID | Content Type | Global Mean |
|---|---|---|---|
| 1 | m1_512_001 | TextVerbal | +0.01548 |
| 2 | m2_002 | ImageVisual | +0.01512 |
| 3 | s4_011 | Factual | +0.00988 |
| 4 | m2_010 | ImageVisual | +0.00971 |
| 5 | s5_005 | Spatial | +0.00954 |

### Bottom 5 Lowest Global Activation

| Rank | ID | Content Type | Global Mean |
|---|---|---|---|
| 1 | s5_004 | Spatial | -0.01337 |
| 2 | s3_009 | Emotional | -0.01288 |
| 3 | m3_009 | AudioText | -0.01067 |
| 4 | b3_011 | ThreatSafety | -0.01062 |
| 5 | m1_512_003 | TextVerbal | -0.01045 |

## 10. Within-Type Activation Stability (Coefficient of Variation)

Lower CV = more consistent activation within a content type.

| Content Type | Mean | SD | CV |
|---|---|---|---|
| ImageVisual | +0.00269 | 0.00565 | 2.10 |
| Abstract | +0.00171 | 0.00429 | 2.50 |
| Novelty | +0.00132 | 0.00350 | 2.66 |
| Emotional | -0.00176 | 0.00468 | 2.66 |
| Spatial | -0.00147 | 0.00580 | 3.94 |
| Social | -0.00102 | 0.00426 | 4.17 |
| ThreatSafety | +0.00102 | 0.00565 | 5.53 |
| Multimodal | -0.00062 | 0.00505 | 8.10 |
| AudioText | -0.00046 | 0.00599 | 13.15 |
| Narrative | +0.00031 | 0.00490 | 15.74 |
| TextVerbal | -0.00026 | 0.00670 | 25.61 |
| Factual | -0.00016 | 0.00458 | 28.15 |
| Reward | -0.00009 | 0.00476 | 52.75 |

## 11. Theory-Relative Interpretation (Demo Mode)

⚠️ All results are from keyword-aware SHA256 mock activations. The following interpretations
are structural/methodological — semantic validity requires real FmriEncoder weights.

**Global mean ranking (top 3):** ImageVisual, Abstract, Novelty
  - GWT/FEP predict: ThreatSafety, Novelty in top 3
  - Observed: ✅ confirmed (demo mode)

**Prefrontal activation ranking (top 3):** Factual, Multimodal, TextVerbal
  - GWT predicts: ThreatSafety highest ignition (broadest activation)
  - Observed: ❌ not confirmed (demo mode)

**Language region ranking (top 3):** Spatial, Abstract, AudioText
  - IIT/DCT predict: Narrative, Social highest language integration
  - Observed: ❌ not confirmed (demo mode)

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