#!/usr/bin/env python3
"""Generate Fig 15: Cross-model triangulation (seq_len robustness + LSA RDM)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
FIG  = Path(__file__).parent / "figures"
OUT  = ROOT / "results" / "cross_model"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

COLORS = {"4": "#2196F3", "8": "#D6604D", "16": "#4CAF50"}
CT_ORDER_SEM = [   # sorted by semantic (seq=8) mean
    "Social","Abstract","Spatial","Novelty","Reward",
    "ThreatSafety","Factual","TextVerbal","Narrative","Multimodal",
    "Emotional","ImageVisual","AudioText",
]

# ─────────────────────────────────────────────────────────────────
stab  = json.load(open(OUT / "seqlen_stability.json"))
rdm_d = json.load(open(OUT / "rdm_analysis.json"))

means_by_sl = stab["means_by_seqlen"]   # {"4": {ct: mean, ...}, ...}
rankings     = stab["rankings"]
corrs        = stab["pairwise_corrs"]

# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# ── Panel A: per-CT means across seq_len ────────────────────────
cts = CT_ORDER_SEM
y   = np.arange(len(cts))
w   = 0.26
offsets = {4: -w, 8: 0, 16: w}
for sl in [4, 8, 16]:
    vals  = [means_by_sl[str(sl)][ct] for ct in cts]
    svals = [vals]   # just means, no SE computed here
    ax1.barh(y + offsets[sl], vals, w, color=COLORS[str(sl)],
             edgecolor="white", label=f"seq_len={sl}", alpha=0.85)

ax1.axvline(0, color="black", lw=0.5, alpha=0.4)
ax1.set_yticks(y)
ax1.set_yticklabels(cts, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("Global activation (a.u.)", fontsize=9)
ax1.set_title("(a) Per-CT activation across\ntemporal integration windows", fontsize=9, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.grid(axis="x", alpha=0.3, linestyle="--")

# ── Panel B: rank-correlation matrix (seq_len pair-wise) ─────────
sl_labels = ["seq_len=4", "seq_len=8", "seq_len=16"]
n_sl = 3
rho_mat = np.ones((n_sl, n_sl))
pairs = [("4","8"), ("4","16"), ("8","16")]
pair_vals = {(a, b): corrs[f"sl{a}_vs_sl{b}"]["spearman_rho"] for a, b in pairs}
idx_map = {"4": 0, "8": 1, "16": 2}
for (a, b), rho in pair_vals.items():
    i, j = idx_map[a], idx_map[b]
    rho_mat[i, j] = rho_mat[j, i] = rho

im = ax2.imshow(rho_mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(n_sl)); ax2.set_xticklabels(sl_labels, rotation=20, ha="right", fontsize=8)
ax2.set_yticks(range(n_sl)); ax2.set_yticklabels(sl_labels, fontsize=8)
for i in range(n_sl):
    for j in range(n_sl):
        ax2.text(j, i, f"{rho_mat[i,j]:.2f}", ha="center", va="center",
                 fontsize=10, fontweight="bold",
                 color="black" if abs(rho_mat[i,j]) < 0.85 else "white")
ax2.set_title("(b) Spearman ρ between seq_len\ncategory rankings", fontsize=9, loc="left")
plt.colorbar(im, ax=ax2, shrink=0.7, label="Spearman ρ")

# ── Panel C: LSA RDM vs TRIBE RDM scatter ────────────────────────
cts_rdm   = rdm_d["cts"]
lsa_rdm   = np.array(rdm_d["bert_rdm"])   # key is "bert_rdm" in the JSON (LSA data stored there)
tribe_rdm = np.array(rdm_d["tribe_rdm"])
n = len(cts_rdm)
idx = np.triu_indices(n, k=1)
bv  = lsa_rdm[idx]
tv  = tribe_rdm[idx]
mantel = rdm_d["mantel"]

ax3.scatter(bv, tv, alpha=0.35, s=18, color="#555", edgecolors="none")
# trend line
z = np.polyfit(bv, tv, 1)
xl = np.linspace(bv.min(), bv.max(), 100)
ax3.plot(xl, np.polyval(z, xl), color="#D6604D", lw=1.5, label=f"r = {mantel['r_pearson']:.3f}")
ax3.set_xlabel("LSA (TF-IDF+SVD-300) cosine dissimilarity (per-CT centroid)", fontsize=9)
ax3.set_ylabel("TRIBE Euclidean dissimilarity (per-CT profile)", fontsize=9)
ax3.set_title("(c) Cross-encoder RDM correlation\n(LSA text-similarity vs TRIBE, Mantel test)", fontsize=9, loc="left")
ax3.legend(frameon=False, fontsize=9)
ax3.grid(alpha=0.3, linestyle="--")
p_str = f"p = {mantel['p_mantel']:.3f}" if mantel['p_mantel'] > 0.001 else "p < 0.001"
ax3.annotate(f"Pearson r = {mantel['r_pearson']:.3f}\nSpearman ρ = {mantel['rho_spearman']:.3f}\n{p_str}",
             xy=(0.04, 0.96), xycoords="axes fraction", va="top",
             fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

fig.suptitle(
    "Figure 15.  Cross-model triangulation: seq_len robustness & LSA text-similarity proxy\n"
    "Left: TRIBE activation ordering is stable across temporal context windows. "
    "Right: LSA text-similarity and TRIBE dissimilarity structures are largely independent, "
    "suggesting TRIBE captures brain-specific signal beyond text similarity.",
    fontsize=9, y=1.03
)
plt.tight_layout()
fig.savefig(FIG / "fig15_cross_model.png")
plt.close()

print("Saved fig15_cross_model.png")
print(f"  seq_len corrs: " +
      ", ".join(f"sl{a}vsss{b}={v:.3f}"
                for (a, b), v in {(a,b): corrs[f'sl{a}_vs_sl{b}']['spearman_rho']
                                   for a,b in pairs}.items()))
print(f"  Mantel: r={mantel['r_pearson']:.3f}, rho={mantel['rho_spearman']:.3f}, p={mantel['p_mantel']:.4f}")
