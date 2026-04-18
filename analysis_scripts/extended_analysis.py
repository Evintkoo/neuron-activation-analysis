#!/usr/bin/env python3
"""
Extended analyses for paper update:
  1. Vertex-level analysis (top-K discriminating vertices, k-means clustering)
  2. Temporal dynamics (response onset, peak, decay across the 16 timesteps)
  3. Cross-source robustness (within-category variation across data origins)
"""
import json, csv, math, time, sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
EXT_DIR = RESULTS_DIR / "extended"
EXT_DIR.mkdir(exist_ok=True)

print("Loading data...")
csv_df = pd.read_csv(RESULTS_DIR / "sweep_ranked.csv")
print(f"  CSV: {len(csv_df)} rows")

# Load full vertex-level data (1.1 GB)
print("  Loading sweep_results.json (1.1GB)...")
t0 = time.time()
with open(RESULTS_DIR / "sweep_results.json") as f:
    raw = json.load(f)
print(f"  Loaded {len(raw)} records in {time.time()-t0:.1f}s")

# Build per-stimulus arrays
stimuli = []
for r in raw:
    if "vertex_acts" not in r: continue
    stimuli.append({
        "id": r["id"],
        "content_type": r["content_type"],
        "source_type": r["source_type"],
        "vertex_acts": np.array(r["vertex_acts"], dtype=np.float32),
    })
N = len(stimuli)
N_VERT = stimuli[0]["vertex_acts"].size
print(f"  N={N} stimuli, {N_VERT} vertices each")

# Build (N, N_VERT) matrix
print("Building activation matrix...")
X = np.stack([s["vertex_acts"] for s in stimuli])
labels_ct = np.array([s["content_type"] for s in stimuli])
ct_unique = sorted(set(labels_ct))
print(f"  X shape: {X.shape}")

REGIONS = [
    ("visual", 0, 3600),
    ("auditory", 3600, 6800),
    ("language", 6800, 10500),
    ("prefrontal", 10500, 14000),
    ("motor", 14000, 17200),
    ("parietal", 17200, 20484),
]

# ═══════════════════════════════════════════════════════════════════════════
# 1. VERTEX-LEVEL ANALYSIS — find top-K most discriminating vertices
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== VERTEX-LEVEL ANALYSIS ===")

# Per-vertex one-way ANOVA across content types
print("  Running per-vertex ANOVA across content types...")
t0 = time.time()
groups = {ct: X[labels_ct == ct] for ct in ct_unique}
F_per_vertex = np.zeros(N_VERT)
for v in range(N_VERT):
    if v % 5000 == 0:
        print(f"    vertex {v}/{N_VERT} ({time.time()-t0:.1f}s)")
    arrays = [g[:, v] for g in groups.values() if len(g) > 1]
    try:
        F, _ = sp_stats.f_oneway(*arrays)
        F_per_vertex[v] = F if not np.isnan(F) else 0
    except Exception:
        F_per_vertex[v] = 0
print(f"  Done in {time.time()-t0:.1f}s")

# Top-K most discriminating vertices
K = 100
top_k_idx = np.argsort(-F_per_vertex)[:K]
print(f"  Top {K} most discriminating vertices:")
for i, v in enumerate(top_k_idx[:10]):
    region = next(r[0] for r in REGIONS if r[1] <= v < r[2])
    print(f"    #{i+1}: vertex {v} (region={region}), F={F_per_vertex[v]:.2f}")

# Region distribution of top vertices
top_region = Counter()
for v in top_k_idx:
    region = next(r[0] for r in REGIONS if r[1] <= v < r[2])
    top_region[region] += 1
print(f"  Top-{K} vertex region distribution: {dict(top_region)}")

# Vertex k-means clustering — group vertices by their content-type response profile
print("  K-means clustering vertices by content-type response profile...")
ct_means_per_vertex = np.zeros((N_VERT, len(ct_unique)))
for ci, ct in enumerate(ct_unique):
    ct_means_per_vertex[:, ci] = X[labels_ct == ct].mean(axis=0)

# Reduce vertices to clusters
N_CLUSTERS = 8
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=5)
vertex_clusters = km.fit_predict(ct_means_per_vertex)
print(f"  Vertex cluster sizes: {np.bincount(vertex_clusters)}")

# For each cluster: which content types preferentially activate it?
cluster_profile = np.zeros((N_CLUSTERS, len(ct_unique)))
for c in range(N_CLUSTERS):
    cluster_profile[c] = ct_means_per_vertex[vertex_clusters == c].mean(axis=0)

# Cross-tab: cluster × dominant region
cluster_region_xtab = np.zeros((N_CLUSTERS, len(REGIONS)), dtype=int)
for v in range(N_VERT):
    region_i = next(i for i, r in enumerate(REGIONS) if r[1] <= v < r[2])
    cluster_region_xtab[vertex_clusters[v], region_i] += 1

# Save vertex analysis
vertex_analysis = {
    "n_vertices": N_VERT,
    "n_stimuli": N,
    "anova": {
        "F_min": float(F_per_vertex.min()),
        "F_max": float(F_per_vertex.max()),
        "F_mean": float(F_per_vertex.mean()),
        "F_p99": float(np.percentile(F_per_vertex, 99)),
        "n_significant_p001": int((F_per_vertex > sp_stats.f.ppf(0.999, len(ct_unique)-1, N - len(ct_unique))).sum()),
    },
    "top_100_vertices": [
        {
            "vertex": int(v),
            "F": float(F_per_vertex[v]),
            "region": next(r[0] for r in REGIONS if r[1] <= v < r[2]),
        } for v in top_k_idx
    ],
    "top_region_distribution": dict(top_region),
    "vertex_clusters": {
        f"cluster_{c}": {
            "size": int((vertex_clusters == c).sum()),
            "region_breakdown": {r[0]: int(cluster_region_xtab[c, i]) for i, r in enumerate(REGIONS)},
            "top_3_content_types": sorted(
                [(ct, float(cluster_profile[c, i])) for i, ct in enumerate(ct_unique)],
                key=lambda x: -x[1])[:3],
        } for c in range(N_CLUSTERS)
    },
}
with open(EXT_DIR / "vertex_analysis.json", "w") as f:
    json.dump(vertex_analysis, f, indent=2)
print(f"  Saved → results/extended/vertex_analysis.json")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL DYNAMICS — analyse the 16 timesteps in temporal_acts
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEMPORAL DYNAMICS ===")

# temporal_acts is [T][6_regions] per stimulus
print("  Extracting temporal activations...")
temp_data = []
for r in raw:
    if "temporal_acts" not in r: continue
    ta = np.array(r["temporal_acts"])  # (T, 6)
    if ta.ndim != 2 or ta.shape[1] != 6: continue
    temp_data.append({
        "id": r["id"],
        "content_type": r["content_type"],
        "temporal_acts": ta,
        "T": ta.shape[0],
    })
print(f"  N={len(temp_data)} stimuli with temporal data")
if temp_data:
    print(f"  Timesteps T: range {min(t['T'] for t in temp_data)}-{max(t['T'] for t in temp_data)}")

# Per-content-type temporal trajectory (averaged across stimuli)
T = max(t["T"] for t in temp_data) if temp_data else 0
ct_temporal = {}
for ct in ct_unique:
    ct_stims = [t for t in temp_data if t["content_type"] == ct]
    if not ct_stims: continue
    # Pad/truncate to T timesteps
    arr = np.zeros((len(ct_stims), T, 6))
    for i, s in enumerate(ct_stims):
        arr[i, :s["T"], :] = s["temporal_acts"]
    ct_temporal[ct] = arr.mean(axis=0)  # (T, 6)

# For each content type: time-to-peak per region
temporal_summary = {}
for ct, traj in ct_temporal.items():
    peaks = {}
    for ri, (rname, _, _) in enumerate(REGIONS):
        region_traj = traj[:, ri]
        if len(region_traj) > 0:
            time_to_peak = int(np.argmax(np.abs(region_traj)))
            peak_val = float(region_traj[time_to_peak])
            onset = int(np.argmax(np.abs(region_traj) > np.abs(region_traj).max() * 0.5))
            decay_rate = float(np.abs(region_traj[-1] - region_traj[time_to_peak]) / max(1, len(region_traj) - time_to_peak))
            peaks[rname] = {
                "time_to_peak": time_to_peak,
                "peak_magnitude": peak_val,
                "onset_t50": onset,
                "decay_rate": decay_rate,
            }
    temporal_summary[ct] = peaks

with open(EXT_DIR / "temporal_dynamics.json", "w") as f:
    json.dump({
        "T": int(T),
        "content_types": list(ct_temporal.keys()),
        "regions": [r[0] for r in REGIONS],
        "summary": temporal_summary,
        "mean_trajectories": {ct: traj.tolist() for ct, traj in ct_temporal.items()},
    }, f, indent=2)
print(f"  Saved → results/extended/temporal_dynamics.json")

# ═══════════════════════════════════════════════════════════════════════════
# 3. CROSS-SOURCE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== CROSS-SOURCE ROBUSTNESS ===")

# Group by (content_type, source_type)
grouped = csv_df.groupby(["content_type", "source_type"])["global_mean"].agg(
    ["count", "mean", "std"]).reset_index()

robustness = {}
for ct in ct_unique:
    sub = grouped[grouped["content_type"] == ct]
    sub = sub[sub["count"] >= 5]  # Need at least 5 stimuli to be reliable
    if len(sub) < 2: continue

    means = sub["mean"].values
    overall_mean = means.mean()
    cross_source_sd = means.std()
    within_source_sd = sub["std"].mean()
    icc = max(0.0, 1.0 - (cross_source_sd**2) / (cross_source_sd**2 + within_source_sd**2 + 1e-12))

    robustness[ct] = {
        "n_sources": int(len(sub)),
        "sources": sub["source_type"].tolist(),
        "source_means": {s: float(m) for s, m in zip(sub["source_type"], sub["mean"])},
        "cross_source_sd": float(cross_source_sd),
        "within_source_sd_avg": float(within_source_sd),
        "icc_proxy": float(icc),
    }

# Print summary
print(f"  Cross-source robustness for {len(robustness)} content types:")
for ct, info in sorted(robustness.items(), key=lambda x: -x[1]["icc_proxy"]):
    if info["n_sources"] >= 2:
        print(f"    {ct:<15} sources={info['n_sources']:<3} "
              f"cross-SD={info['cross_source_sd']:.5f} "
              f"within-SD={info['within_source_sd_avg']:.5f} "
              f"ICC≈{info['icc_proxy']:.3f}")

with open(EXT_DIR / "cross_source_robustness.json", "w") as f:
    json.dump(robustness, f, indent=2)
print(f"  Saved → results/extended/cross_source_robustness.json")

# ═══════════════════════════════════════════════════════════════════════════
# 4. COHEN'S D EFFECT MATRIX (full 13×13)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== FULL COHEN'S d MATRIX ===")
d_matrix = np.zeros((len(ct_unique), len(ct_unique)))
for i, a in enumerate(ct_unique):
    for j, b in enumerate(ct_unique):
        if i == j: continue
        ga = csv_df[csv_df["content_type"] == a]["global_mean"].values
        gb = csv_df[csv_df["content_type"] == b]["global_mean"].values
        if len(ga) < 2 or len(gb) < 2: continue
        pooled = math.sqrt(((len(ga)-1)*ga.var() + (len(gb)-1)*gb.var()) / (len(ga)+len(gb)-2))
        if pooled > 1e-12:
            d_matrix[i, j] = (ga.mean() - gb.mean()) / pooled

with open(EXT_DIR / "cohens_d_matrix.json", "w") as f:
    json.dump({"content_types": ct_unique, "matrix": d_matrix.tolist()}, f, indent=2)
print(f"  Saved → results/extended/cohens_d_matrix.json")

print("\n=== EXTENDED ANALYSIS COMPLETE ===")
print(f"  Output dir: {EXT_DIR}")
