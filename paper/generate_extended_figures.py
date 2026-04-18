#!/usr/bin/env python3
"""Generate figures for the extended analyses (vertex, temporal, cross-source, multilingual)."""

import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).parent.parent
EXT  = ROOT / "results" / "extended"
FIG  = Path(__file__).parent / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

CT_ORDER = [
    "ThreatSafety","AudioText","TextVerbal","Social","Novelty",
    "ImageVisual","Factual","Emotional","Abstract","Reward",
    "Spatial","Multimodal","Narrative",
]
PALETTE = {ct: c for ct, c in zip(CT_ORDER, [
    "#D62728","#FF7F0E","#F7B500","#2CA02C","#17BECF",
    "#9467BD","#8C564B","#E377C2","#1F77B4","#BCBD22",
    "#7F7F7F","#AEC7E8","#FFBB78",
])}
REGIONS = ["Visual","Auditory","Language","Prefrontal","Motor","Parietal"]

# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 — Vertex-level F-statistic distribution + top vertex regional dist
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 9: vertex analysis...")
vert = json.load(open(EXT / "vertex_analysis.json"))
top = vert["top_100_vertices"]
top_F = [v["F"] for v in top]
top_regions = [v["region"] for v in top]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: top-100 F values (sorted)
axes[0].plot(range(1, len(top_F)+1), sorted(top_F, reverse=True),
             color="#1F77B4", linewidth=2)
axes[0].fill_between(range(1, len(top_F)+1), sorted(top_F, reverse=True),
                     alpha=0.2, color="#1F77B4")
axes[0].axhline(13.51, color="#D62728", linestyle="--", lw=1, label="6-region ANOVA F=13.51")
axes[0].set_xlabel("Vertex rank (top 100)", fontsize=10)
axes[0].set_ylabel("F-statistic", fontsize=10)
axes[0].set_title("(a) Top-100 vertex F-statistics\n(per-vertex one-way ANOVA across content types)", fontsize=10)
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(True)

# Panel B: regional distribution of top 100 vertices
from collections import Counter
region_counts = Counter(top_regions)
region_order = ["Visual","Auditory","Language","Prefrontal","Motor","Parietal"]
counts = [region_counts.get(r.lower(), 0) for r in region_order]
colors_r = ["#D6604D","#F4A582","#FDDBC7","#92C5DE","#4393C3","#2166AC"]
bars = axes[1].bar(region_order, counts, color=colors_r, edgecolor="white")
for b, c in zip(bars, counts):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f"{c}%", ha="center", fontsize=9, fontweight="bold")
axes[1].set_ylabel("Count (out of top 100)", fontsize=10)
axes[1].set_title("(b) Cortical region distribution of\ntop-100 most discriminating vertices", fontsize=10)
axes[1].set_ylim(0, max(counts)*1.2)
axes[1].grid(axis="y")

fig.suptitle("Figure 9. Vertex-Level Analysis — Sub-Regional Discriminative Resolution",
             fontsize=10, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig9_vertex_analysis.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Figure 10 — Cross-source robustness (ICC bubble plot)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 10: cross-source robustness...")
robust = json.load(open(EXT / "cross_source_robustness.json"))
items = sorted(robust.items(), key=lambda x: -x[1]["icc_proxy"])

fig, ax = plt.subplots(figsize=(8.5, 5.5))
y = np.arange(len(items))
icc = [info["icc_proxy"] for _, info in items]
n_src = [info["n_sources"] for _, info in items]
colors = [PALETTE.get(ct, "#888") for ct, _ in items]
labels = [f"{ct}  ({info['n_sources']} sources)" for ct, info in items]

scatter = ax.scatter(icc, y, c=colors, s=[n*30 for n in n_src],
                     alpha=0.85, edgecolors="white", linewidths=1, zorder=3)
ax.hlines(y, 0.5, icc, colors=colors, alpha=0.3, lw=2)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Cross-source ICC (proxy)", fontsize=10)
ax.invert_yaxis()
ax.axvline(0.95, color="#2CA02C", linestyle=":", lw=1, label="ICC=0.95 (excellent)")
ax.axvline(0.75, color="#FF7F0E", linestyle=":", lw=1, label="ICC=0.75 (good)")
ax.set_xlim(0.5, 1.0)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Figure 10. Cross-Source Robustness of Content-Type Effects\n"
             "Same content type from different sources produces consistent activations.\n"
             "Bubble size ∝ number of sources contributing to each category.",
             fontsize=9, loc="left", pad=8)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig10_cross_source_robustness.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Figure 11 — Temporal dynamics (if available)
# ═══════════════════════════════════════════════════════════════════════════
temp_path = EXT / "temporal_dynamics_by_ct.json"
if temp_path.exists():
    print("Fig 11: temporal dynamics...")
    td = json.load(open(temp_path))
    if td.get("trajectories"):
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
        axes = axes.flatten()
        for ri, region in enumerate(REGIONS):
            ax = axes[ri]
            for ct in CT_ORDER:
                traj = td["trajectories"].get(ct)
                if not traj: continue
                arr = np.array(traj)
                if arr.ndim == 2 and arr.shape[1] >= 6:
                    ax.plot(range(arr.shape[0]), arr[:, ri],
                            color=PALETTE.get(ct, "#888"), alpha=0.7,
                            linewidth=1.2, label=ct if ri == 0 else None)
            ax.set_title(region, fontsize=10)
            ax.set_xlabel("Timestep" if ri >= 3 else "")
            ax.set_ylabel("Activation" if ri % 3 == 0 else "")
            ax.grid(True)
        axes[0].legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2,
                       bbox_to_anchor=(0, 1.5))
        fig.suptitle("Figure 11. Temporal Activation Trajectories per Cortical Region\n"
                     "Mean predicted BOLD trajectory across 16 timesteps, by content category.",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        fig.savefig(FIG / "fig11_temporal_dynamics.png")
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Figure 12 — Multilingual results (if available)
# ═══════════════════════════════════════════════════════════════════════════
ml_path = EXT / "multilingual_results.json"
if ml_path.exists():
    print("Fig 12: multilingual...")
    ml = json.load(open(ml_path))
    if ml:
        from collections import defaultdict
        by_lang = defaultdict(list)
        for r in ml:
            if r.get("global_mean") is not None:
                by_lang[r["language_name"]].append(r)

        if by_lang:
            langs = sorted(by_lang.keys())
            means = [np.mean([r["global_mean"] for r in by_lang[l]]) for l in langs]
            sds   = [np.std ([r["global_mean"] for r in by_lang[l]]) for l in langs]
            ns    = [len(by_lang[l]) for l in langs]
            colors_lang = ["#D62728","#FF7F0E","#2CA02C","#1F77B4","#9467BD",
                           "#8C564B","#E377C2","#7F7F7F","#BCBD22","#17BECF"][:len(langs)]
            order = np.argsort(means)
            langs_o = [langs[i] for i in order]
            means_o = [means[i] for i in order]
            sds_o   = [sds[i] for i in order]
            ns_o    = [ns[i] for i in order]
            colors_o = [colors_lang[i] for i in order]

            fig, ax = plt.subplots(figsize=(8, 5))
            y = np.arange(len(langs_o))
            ax.barh(y, means_o, xerr=sds_o, color=colors_o, alpha=0.85,
                    height=0.6, error_kw=dict(ecolor="#333", capsize=3))
            ax.set_yticks(y)
            ax.set_yticklabels([f"{l}  (N={n})" for l, n in zip(langs_o, ns_o)], fontsize=9)
            ax.set_xlabel("Predicted global cortical activation (mean BOLD, a.u.)", fontsize=10)
            ax.set_title("Figure 12. Cross-Linguistic Replication: TRIBE v2 Activation by Wikipedia Language\n"
                         "Mean predicted activation across random Wikipedia summaries in 10 languages.",
                         fontsize=9, loc="left", pad=8)
            ax.grid(axis="x")
            plt.tight_layout()
            fig.savefig(FIG / "fig12_multilingual.png")
            plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Figure 13 — Cohen's d full matrix (13×13 heatmap)
# ═══════════════════════════════════════════════════════════════════════════
print("Fig 13: Cohen's d matrix...")
dm = json.load(open(EXT / "cohens_d_matrix.json"))
cts = dm["content_types"]
mat = np.array(dm["matrix"])

# Reorder by CT_ORDER
order = [cts.index(ct) for ct in CT_ORDER if ct in cts]
mat_o = mat[np.ix_(order, order)]
cts_o = [cts[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 7))
vmax = max(np.abs(mat_o.max()), np.abs(mat_o.min()))
cmap = LinearSegmentedColormap.from_list("d", ["#2166AC","#92C5DE","white","#F4A582","#D6604D"], N=256)
im = ax.imshow(mat_o, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
ax.set_xticks(range(len(cts_o)))
ax.set_xticklabels(cts_o, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cts_o)))
ax.set_yticklabels(cts_o, fontsize=8)

for i in range(len(cts_o)):
    for j in range(len(cts_o)):
        v = mat_o[i, j]
        if i != j:
            text_color = "white" if abs(v) > vmax*0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

cbar = fig.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label("Cohen's d (row vs column)", fontsize=9)
ax.set_title("Figure 13. Pairwise Cohen's d Matrix Across All 13 Content Categories\n"
             "Cell (i,j) = standardised mean difference of i minus j on global activation.",
             fontsize=9, loc="left", pad=8)
plt.tight_layout()
fig.savefig(FIG / "fig13_cohens_d_matrix.png")
plt.close()

print(f"\nAll extended figures saved to {FIG}")
