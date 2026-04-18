#!/usr/bin/env python3
"""Generate Fig 14: hash vs semantic LLaMA encoding comparison (full N=30/CT)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = Path(__file__).parent.parent
FIG  = Path(__file__).parent / "figures"

plt.rcParams.update({"font.family":"serif","font.size":10,"savefig.dpi":200,
                     "savefig.bbox":"tight","savefig.facecolor":"white",
                     "axes.spines.top":False,"axes.spines.right":False})

# Sorted by full semantic mean, highest first (rank 1 = least negative = most activating)
CT_ORDER = [
    "AudioText","ImageVisual","Emotional","Multimodal","Narrative",
    "TextVerbal","Factual","ThreatSafety","Reward","Novelty",
    "Spatial","Abstract","Social",
]

# Load hash-mode 3008-stim sweep
hash_sw = json.load(open(ROOT / "results/sweep_results.json"))
hash_by_ct = defaultdict(list)
for r in hash_sw:
    if r.get("global_mean") is not None and r.get("content_type"):
        hash_by_ct[r["content_type"]].append(r["global_mean"])
hash_means = {ct: float(np.mean(v)) for ct, v in hash_by_ct.items()}
hash_sds   = {ct: float(np.std(v))  for ct, v in hash_by_ct.items()}
hash_ns    = {ct: len(v) for ct, v in hash_by_ct.items()}

# Load semantic LLaMA 26-stim sweep summary
sem = json.load(open(ROOT / "results/llama_sweep/summary_by_ct.json"))
sem_means = {ct: v["global_mean"] for ct, v in sem.items()}
sem_sds   = {ct: v["global_sd"]   for ct, v in sem.items()}
sem_ns    = {ct: v["n"] for ct, v in sem.items()}

cts = [ct for ct in CT_ORDER if ct in hash_means and ct in sem_means]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2),
                                gridspec_kw={"width_ratios":[2.2, 1]})

# ── Panel A: per-CT means side-by-side ───────────────────────────────────────
y = np.arange(len(cts))
h_vals = [hash_means[ct] for ct in cts]
s_vals = [sem_means[ct]  for ct in cts]
h_sds  = [hash_sds[ct] / np.sqrt(hash_ns[ct]) for ct in cts]  # SE
s_sds  = [sem_sds[ct]  / np.sqrt(sem_ns[ct])  for ct in cts]

bw = 0.38
ax1.barh(y - bw/2, h_vals, bw, color="#9EC5E8", edgecolor="white",
         label=f"Hash encoding (N={sum(hash_ns.values())} total, ~{int(np.mean(list(hash_ns.values())))}/CT)",
         xerr=h_sds, error_kw=dict(ecolor="#555", capsize=2, alpha=0.6))
ax1.barh(y + bw/2, s_vals, bw, color="#D6604D", edgecolor="white",
         label=f"LLaMA-3.2-3B semantic (N={sum(sem_ns.values())} total, 30/CT — full sweep)",
         xerr=s_sds, error_kw=dict(ecolor="#555", capsize=2, alpha=0.6))
ax1.axvline(0, color="black", lw=0.5, alpha=0.4)
ax1.set_yticks(y); ax1.set_yticklabels(cts, fontsize=9)
ax1.invert_yaxis()
ax1.set_xlabel("Predicted global cortical activation (mean BOLD, a.u.)", fontsize=10)
ax1.set_title("(a) Per-content-type global activation: hash vs. semantic encoding", fontsize=10, loc="left")
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.grid(axis="x", alpha=0.3, linestyle="--")

# ── Panel B: spread comparison ────────────────────────────────────────────────
hash_spread = max(h_vals) - min(h_vals)
sem_spread  = max(s_vals) - min(s_vals)
ratio = sem_spread / hash_spread

ax2.bar(["Hash\n(hash)", "LLaMA\n(semantic)"],
        [hash_spread, sem_spread],
        color=["#9EC5E8", "#D6604D"], edgecolor="white", width=0.55)
for i, v in enumerate([hash_spread, sem_spread]):
    ax2.text(i, v*1.02, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_ylabel("max − min global activation across 13 CTs", fontsize=10)
ax2.set_title(f"(b) Activation spread\n(semantic/hash ratio = {ratio:.1f}×)",
              fontsize=10, loc="left")
ax2.grid(axis="y", alpha=0.3, linestyle="--")

fig.suptitle("Figure 14. Hash-Encoding vs. LLaMA-3.2-3B Semantic Encoding (Full N=30/CT)\n"
             "Semantic encoding produces a ~%.0fx wider activation spread; rank-ordering partially shifts." % ratio,
             fontsize=10, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig14_hash_vs_semantic.png")
plt.close()
print(f"Saved fig14 (hash spread = {hash_spread:.5f}, semantic spread = {sem_spread:.5f}, ratio = {ratio:.1f}x)")
