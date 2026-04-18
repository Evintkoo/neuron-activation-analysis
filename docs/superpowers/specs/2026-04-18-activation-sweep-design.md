# Activation Sweep: Empirical Content-Type Ranking via TRIBE v2

**Date:** 2026-04-18
**Status:** Approved
**Depends on:** `2026-04-17-neuron-activation-research-design.md`

---

## 1. Research Goal

Empirically determine which content types, internet source formats, and language structures produce the highest activation in the TRIBE v2 fMRI brain encoder — both globally (across all 20,484 cortical vertices) and regionally (across the 6 tracked cortical regions).

---

## 2. Model Context

**TRIBE v2** (`tribe-playground/tribe-downloader`) is a 177M-parameter transformer encoder that predicts cortical surface activations from text, audio, and image inputs. It was trained on real fMRI data.

- **Text encoder:** LLaMA-3.2-3B → 6144-dim features
- **Audio encoder:** Wav2Vec2Bert → 2048-dim features
- **Image encoder:** CLIP ViT-L/14 → 2816-dim features
- **Output:** 20,484 cortical vertex activations (`[B, T, N_VERT]`)
- **Server:** Axum HTTP at `localhost:8081`, endpoint `POST /api/predict`

**Tracked brain regions** (vertex index ranges):

| Region | Vertices |
|---|---|
| visual | 0–3600 |
| auditory | 3600–6800 |
| language | 6800–10500 |
| prefrontal | 10500–14000 |
| motor | 14000–17200 |
| parietal | 17200–20484 |

The server returns `global_stats` (mean/std/min/max across all vertices) and `region_stats` (mean/std/rel_activation/peak per region) in every response. `rel_activation = (region_mean − global_mean) / global_std`.

---

## 3. Execution Strategy

**Phase A (immediate):** Build the full sweep harness and corpus in demo mode (hash-based encoders, no weights required). Validates the pipeline end-to-end.

**Phase B (after weights):** Re-run the identical sweep with real LLaMA + FmriEncoder weights. Demo mode results are clearly flagged (`demo_mode: true`) and excluded from analysis.

This means zero idle time — pipeline development and weight download happen in parallel.

---

## 4. Stimulus Corpus

**File:** `experiments/corpus/stimuli.json`
**Size:** 150 text stimuli, ~11–12 per content type across all 13 categories from the parent spec.

Each stimulus entry:
```json
{
  "id": "b3_001",
  "content_type": "ThreatSafety",
  "source_type": "breaking_news",
  "language_structure": "short_declarative_present_tense",
  "text": "..."
}
```

### 4.1 Content Type → Source Type → Language Structure Mapping

| Content Type | Internet Source | Language Structure |
|---|---|---|
| B3 ThreatSafety | Breaking news headlines, emergency alerts, health warnings | Short declarative, present-tense urgency, imperative |
| B4 Novelty | Science breakthroughs, "first-ever" announcements, anomaly reports | Surprise framing, embedded contrast, superlatives |
| S1 Narrative | Reddit personal stories, movie synopses, news long-form | Chronological, causal connectors (then/so/because), past-tense sequence |
| B1 Social | Conversation threads, theory-of-mind scenarios, interpersonal conflict | Multi-agent dialogue, perspective shifts, mental state verbs |
| S3 Emotional | High-valence posts (positive and negative), personal confessions | First-person, affect-laden verbs, exclamatory |
| M1 Text/Verbal | Wikipedia opening paragraphs at 3 lengths (32/128/512 tokens) | Neutral declarative, varied length, third-person |
| S4 Factual | Encyclopedia statements, how-to instructions, technical specs | Third-person passive, no narrative arc, present tense |
| S2 Abstract | Philosophy excerpts, math definitions, conceptual frameworks | Dense nominal, low-concreteness, nested clauses |
| S5 Spatial | Navigation instructions, room/scene descriptions, maps-as-text | Relational prepositions, cardinal directions, ordinal sequences |
| B2 Reward | 5-star product reviews, achievement announcements, congratulations | Positive superlatives, outcome framing, second-person |
| M2 Image (text) | Image alt-text descriptions, visual scene captions | Sensory-descriptive, color/shape/position vocabulary |
| M3 Audio (text) | Audio scene captions, music descriptions, sound event labels | Acoustic vocabulary, tempo/timbre/rhythm language |
| M4 Multimodal | AV scene descriptions combining visual + spoken content | Mixed sensory + linguistic vocabulary, concurrent event framing |

### 4.2 Language Structure Contrastive Pairs

Within B3, S1, and S4, include 3 matched pairs where content is held constant but structure varies:

- **B3:** Identical threat content in (a) headline format vs. (b) narrative first-person account
- **S1:** Same story in (a) chronological causal sequence vs. (b) bullet-point summary
- **S4:** Same fact as (a) declarative statement vs. (b) passive construction

These pairs enable direct attribution of activation differences to language structure rather than content.

---

## 5. Sweep Harness

**Location:** `experiments/sweep/` (new Rust binary crate)

### 5.1 Behavior

1. Read `experiments/corpus/stimuli.json`
2. Check `GET localhost:8081/health` — exit with clear error if server not running
3. For each stimulus, send `POST /api/predict` with `{"text": "...", "seq_len": 16}`
4. Collect per-stimulus: `demo_mode`, `global_stats.global_mean`, `global_stats.global_max`, and per-region `mean` + `rel_activation` for all 6 regions
5. Write `results/sweep_results.json` (full raw responses)
6. Write `results/sweep_ranked.csv` (sorted by `global_mean` descending)
7. Write `results/region_heatmap.json` (13×6 content-type × region `rel_activation` matrix)
8. Print live progress and a top-10 summary table to stdout

### 5.2 Output Schema

**`sweep_ranked.csv` columns:**
`rank, id, content_type, source_type, language_structure, demo_mode, global_mean, global_max, visual_rel, auditory_rel, language_rel, prefrontal_rel, motor_rel, parietal_rel`

**`region_heatmap.json` schema:**
```json
{
  "content_types": ["ThreatSafety", "Novelty", ...],
  "regions": ["visual", "auditory", "language", "prefrontal", "motor", "parietal"],
  "matrix": [[rel_activation per region], ...]
}
```

**`sweep_results.json`:** array of raw API responses keyed by stimulus `id`, including full `vertex_acts` for downstream analysis.

### 5.3 Demo Mode Handling

Results with `demo_mode: true` are written to output files but flagged. The ranked CSV includes a `demo_mode` column. The harness prints a warning at completion if any results are in demo mode:
```
⚠  150/150 results are in demo mode — re-run after loading real weights for valid results.
```

---

## 6. Weight Download

**Script:** `scripts/download_weights.sh`

Hits tribe-server download endpoints (server must be running):
```bash
curl -s -X POST localhost:8081/api/download/llama
curl -s -X POST localhost:8081/api/download/wav2vec
curl -s -X POST localhost:8081/api/download/clip
```

For `best.safetensors` (FmriEncoder): run `python3 convert_ckpt.py` in the tribe-playground directory. This is a prerequisite for any real-weights run — the server will not start without it.

Poll job status with `GET localhost:8081/api/jobs/:job_id` until complete.

---

## 7. Analysis Outputs

| File | Purpose |
|---|---|
| `results/sweep_ranked.csv` | Ranked stimulus table — sortable by any region or global metric |
| `results/region_heatmap.json` | 13×6 matrix for Chart.js dashboard visualization |
| `results/sweep_results.json` | Full raw results including vertex_acts for follow-on analysis |

The `region_heatmap.json` feeds directly into the existing `viz/brain_heatmap/dashboard.html` Chart.js panel.

---

## 8. Success Criteria

| Criterion | Target |
|---|---|
| Corpus coverage | All 13 content types represented, ≥10 stimuli each |
| Contrastive pairs | 3 matched pairs (B3/S1/S4), structure-only variation |
| Pipeline validates | Sweep runs end-to-end in demo mode with 0 errors |
| Real-weights result | ≥3 content types show `rel_activation > 1.0` in ≥1 region |
| Theory validation | B3/B4 rank in top 3 for `global_mean` (FEP/GWT prediction) |
| Region specificity | Language region shows highest `rel_activation` for S1/B1 types |

---

## 9. Repository Changes

```
experiments/
├── corpus/
│   └── stimuli.json              # 150 curated stimuli
├── sweep/
│   ├── Cargo.toml                # new binary crate
│   └── src/
│       └── main.rs               # sweep harness
results/
├── sweep_results.json            # (gitignored — large)
├── sweep_ranked.csv
└── region_heatmap.json
scripts/
└── download_weights.sh
Cargo.toml                        # add experiments/sweep to workspace
```
