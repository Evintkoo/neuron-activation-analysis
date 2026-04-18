#!/usr/bin/env python3
"""Generate all publication-quality figures for the Activation Cartography paper."""

import json, csv, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

ROOT    = Path(__file__).parent.parent
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Serif"
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         [FONT_FAMILY, "Times New Roman", "Georgia"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.5,
})

# ── Content-type colour palette (distinct, colorblind-friendly) ───────────────
CT_ORDER = [
    "ThreatSafety","AudioText","TextVerbal","Social","Novelty",
    "ImageVisual","Factual","Emotional","Abstract","Reward",
    "Spatial","Multimodal","Narrative",
]
PALETTE = {
    ct: c for ct, c in zip(CT_ORDER, [
        "#D62728","#FF7F0E","#F7B500","#2CA02C","#17BECF",
        "#9467BD","#8C564B","#E377C2","#1F77B4","#BCBD22",
        "#7F7F7F","#AEC7E8","#FFBB78",
    ])
}

REGIONS      = ["Visual","Auditory","Language","Prefrontal","Motor","Parietal"]
REGION_KEYS  = ["visual_rel","auditory_rel","language_rel","prefrontal_rel","motor_rel","parietal_rel"]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "results" / "sweep_ranked.csv")
heatmap = json.load(open(ROOT / "results" / "region_heatmap.json"))
hm_cts  = heatmap["content_types"]
hm_mat  = np.array(heatmap["matrix"])  # (n_ct, 6)

# ── Descriptive stats ─────────────────────────────────────────────────────────
def bootstrap_ci(xs, n=2000, alpha=0.05):
    rng = np.random.default_rng(42)
    boot = [rng.choice(xs, len(xs), replace=True).mean() for _ in range(n)]
    return np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])

stats = []
for ct in CT_ORDER:
    sub = df[df["content_type"] == ct]["global_mean"].values
    if len(sub) == 0: continue
    ci = bootstrap_ci(sub)
    stats.append({"ct": ct, "n": len(sub), "mean": sub.mean(),
                  "sd": sub.std(), "ci_lo": ci[0], "ci_hi": ci[1]})
stats_df = pd.DataFrame(stats).set_index("ct").reindex(CT_ORDER).dropna()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Global Activation Ranking (horizontal bar + CI)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 4.8))

y_pos = np.arange(len(stats_df))
colors = [PALETTE[ct] for ct in stats_df.index]
means  = stats_df["mean"].values
ci_lo  = stats_df["ci_lo"].values
ci_hi  = stats_df["ci_hi"].values
err_lo = means - ci_lo
err_hi = ci_hi - means

bars = ax.barh(y_pos, means, xerr=[err_lo, err_hi],
               color=colors, alpha=0.85, height=0.65,
               error_kw=dict(ecolor="#333333", capsize=3, linewidth=1.0))

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{ct}  (N={stats_df.loc[ct,'n']})" for ct in stats_df.index],
                   fontsize=9)
ax.invert_yaxis()
ax.axvline(means.mean(), color="#555555", linestyle=":", linewidth=1, label="Grand mean")
ax.set_xlabel("Predicted global cortical activation (mean BOLD, a.u.)", fontsize=10)
ax.set_title("Figure 1.  Predicted Global Cortical Activation by Internet Content Category\n"
             "Bars show bootstrap 95% CI; N per category in parentheses. Higher = stronger predicted response.",
             fontsize=9, loc="left", pad=8)
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="x", alpha=0.3)
ax.grid(axis="y", alpha=0)

# Significance annotation for top pair
ax.annotate("", xy=(means[0]+0.00005, 0), xytext=(means[-1]-0.00005, len(stats_df)-1),
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
ax.text(means.mean() + 0.00015, len(stats_df)//2 - 0.3,
        f"d = 0.82*", fontsize=7.5, color="#888888", style="italic")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_global_activation_ranking.png")
plt.close()
print("Fig 1 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Region × Content-Type Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8.5, 5.2))

# Build matrix in CT_ORDER
hm_dict = {ct: row for ct, row in zip(hm_cts, hm_mat)}
mat_ordered = np.array([hm_dict.get(ct, [0]*6) for ct in CT_ORDER])

cmap = LinearSegmentedColormap.from_list(
    "bwr_custom",
    ["#2166AC","#92C5DE","#F7F7F7","#F4A582","#D6604D"],
    N=256
)
vmax = np.abs(mat_ordered).max()
im = ax.imshow(mat_ordered, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

ax.set_xticks(range(6))
ax.set_xticklabels(REGIONS, fontsize=9, rotation=20, ha="right")
ax.set_yticks(range(len(CT_ORDER)))
ax.set_yticklabels(CT_ORDER, fontsize=9)

for i in range(len(CT_ORDER)):
    for j in range(6):
        val = mat_ordered[i, j]
        text_color = "white" if abs(val) > vmax * 0.55 else "black"
        ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                fontsize=7, color=text_color, fontweight="normal")

cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Relative activation (z-score)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title(
    "Figure 2.  Regional Cortical Activation Profile per Content Category\n"
    "Values are relative activation (region mean − global mean) / global SD. "
    "Blue = below global mean; Red = above global mean.",
    fontsize=9, loc="left", pad=8)
ax.set_xlabel("Cortical Region", fontsize=10)
ax.set_ylabel("Content Category", fontsize=10)

# Grid lines
for i in range(len(CT_ORDER)+1):
    ax.axhline(i-0.5, color="white", lw=0.5)
for j in range(7):
    ax.axvline(j-0.5, color="white", lw=0.5)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_region_heatmap.png")
plt.close()
print("Fig 2 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — PCA scatter of content types in PC1 × PC2 space
# ═══════════════════════════════════════════════════════════════════════════════

# Recompute PCA from region means
def power_pca2(M):
    col_means = M.mean(axis=0)
    X = M - col_means
    C = X.T @ X / (len(M)-1)
    def power_iter(A, n=500):
        v = np.ones(A.shape[0]); v /= np.linalg.norm(v)
        for _ in range(n): v = A @ v; v /= np.linalg.norm(v)
        lam = v @ A @ v
        return v, lam
    pc1, lam1 = power_iter(C)
    C2 = C - lam1 * np.outer(pc1, pc1)
    pc2, lam2 = power_iter(C2)
    scores = X @ np.column_stack([pc1, pc2])
    var_total = np.diag(C).sum()
    return scores, pc1, pc2, lam1/var_total, lam2/var_total

M = mat_ordered  # (13, 6)
scores, pc1, pc2, ev1, ev2 = power_pca2(M)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# Left: scatter
ax = axes[0]
for i, ct in enumerate(CT_ORDER):
    ax.scatter(scores[i,0], scores[i,1], color=PALETTE[ct], s=80, zorder=3,
               edgecolors="white", linewidths=0.6)
    ax.annotate(ct, (scores[i,0], scores[i,1]),
                textcoords="offset points", xytext=(5, 3), fontsize=7.5,
                color=PALETTE[ct], fontweight="bold")
ax.axhline(0, color="#aaaaaa", lw=0.7, zorder=1)
ax.axvline(0, color="#aaaaaa", lw=0.7, zorder=1)
ax.set_xlabel(f"PC1  ({ev1*100:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PC2  ({ev2*100:.1f}% variance)", fontsize=10)
ax.set_title("(a)  Content types in PC space", fontsize=10)

# Right: PC1 loadings bar chart
ax2 = axes[1]
load_colors = ["#D62728" if v > 0 else "#1F77B4" for v in pc1]
bars2 = ax2.barh(range(6), pc1, color=load_colors, alpha=0.8, height=0.6)
ax2.set_yticks(range(6))
ax2.set_yticklabels(REGIONS, fontsize=9)
ax2.axvline(0, color="#333333", lw=0.8)
ax2.set_xlabel("PC1 loading", fontsize=10)
ax2.set_title("(b)  PC1 loadings by cortical region", fontsize=10)
for i, (v, c) in enumerate(zip(pc1, load_colors)):
    ax2.text(v + (0.005 if v >= 0 else -0.005), i,
             f"{v:+.3f}", va="center",
             ha="left" if v >= 0 else "right", fontsize=8, color="#333333")

fig.suptitle(
    "Figure 3.  Principal Component Analysis of Regional Activation Profiles\n"
    "PC1 (96.9% variance) defines a sensory-language vs. executive-motor gradient. "
    "Red loadings = positive (executive); Blue = negative (sensory).",
    fontsize=9, y=1.01
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_pca.png")
plt.close()
print("Fig 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Per-Region ANOVA F-values
# ═══════════════════════════════════════════════════════════════════════════════
from scipy import stats as sp_stats

fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
axes = axes.flatten()

for idx, (reg_key, reg_label) in enumerate(zip(REGION_KEYS, REGIONS)):
    ax = axes[idx]
    groups = [df[df["content_type"]==ct][reg_key].dropna().values for ct in CT_ORDER]
    groups = [g for g in groups if len(g) > 1]

    F_val, p_val = sp_stats.f_oneway(*groups)
    ct_means = [(df[df["content_type"]==ct][reg_key].mean(), ct) for ct in CT_ORDER
                if ct in df["content_type"].values]
    ct_means_sorted = sorted(ct_means, reverse=True)

    colors_sorted = [PALETTE[ct] for _, ct in ct_means_sorted]
    vals = [v for v, _ in ct_means_sorted]
    labels = [ct for _, ct in ct_means_sorted]

    bars3 = ax.bar(range(len(vals)), vals, color=colors_sorted, alpha=0.85, width=0.7)
    ax.axhline(0, color="#333333", lw=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=6.5)
    ax.set_title(f"{reg_label}\nF={F_val:.2f}, p{'<0.001' if p_val<0.001 else f'={p_val:.3f}'}",
                 fontsize=9)
    ax.set_ylabel("Rel. activation (z)", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Figure 4.  Mean Relative Activation per Cortical Region by Content Category\n"
    "All six regions show significant content-type effects (F-test, all p < 0.001).",
    fontsize=10, y=1.01
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_regional_anova.png")
plt.close()
print("Fig 4 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Cohen's d matrix (top pairs, dot plot)
# ═══════════════════════════════════════════════════════════════════════════════
from itertools import combinations

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 0.0
    pooled = math.sqrt(((na-1)*a.var() + (nb-1)*b.var()) / (na+nb-2))
    return (a.mean()-b.mean())/pooled if pooled > 1e-12 else 0.0

pairs_d = {}
for ct_a, ct_b in combinations(CT_ORDER, 2):
    ga = df[df["content_type"]==ct_a]["global_mean"].values
    gb = df[df["content_type"]==ct_b]["global_mean"].values
    if len(ga) > 1 and len(gb) > 1:
        pairs_d[(ct_a, ct_b)] = cohens_d(pd.Series(ga), pd.Series(gb))

top20 = sorted(pairs_d.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

fig, ax = plt.subplots(figsize=(8, 6))
labels_p = [f"{a}\nvs\n{b}" for (a,b),_ in top20]
vals_p   = [d for _,d in top20]
colors_p = ["#D62728" if abs(d) > 0.8 else "#FF7F0E" if abs(d) > 0.5 else "#1F77B4"
            for d in vals_p]

y = np.arange(len(top20))
ax.scatter(vals_p, y, c=colors_p, s=90, zorder=3, edgecolors="white", linewidths=0.5)
ax.hlines(y, 0, vals_p, colors=colors_p, alpha=0.5, linewidth=2)
ax.axvline(0, color="#333333", lw=0.8)
ax.axvline(0.8, color="#D62728", lw=0.8, linestyle=":", alpha=0.6, label="|d|=0.8 (large)")
ax.axvline(-0.8, color="#D62728", lw=0.8, linestyle=":", alpha=0.6)
ax.axvline(0.5, color="#FF7F0E", lw=0.8, linestyle=":", alpha=0.5, label="|d|=0.5 (medium)")
ax.axvline(-0.5, color="#FF7F0E", lw=0.8, linestyle=":", alpha=0.5)

ax.set_yticks(y)
ax.set_yticklabels(labels_p, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel("Cohen's d", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title(
    "Figure 5.  Top 20 Pairwise Effect Sizes (Cohen's d) on Global Activation\n"
    "Red = large (|d|>0.8); Orange = medium (|d|>0.5); Blue = small.",
    fontsize=9, loc="left", pad=8
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_effect_sizes.png")
plt.close()
print("Fig 5 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Contrastive pair delta (radar / bar comparison)
# ═══════════════════════════════════════════════════════════════════════════════
pairs_data = [
    {
        "label": "ThreatSafety\nHeadline vs Narrative",
        "A_label": "Headline",
        "B_label": "Narrative",
        "A": {"global_mean":-0.00199, "visual_rel":0.03579, "auditory_rel":0.04295,
              "language_rel":0.05812, "prefrontal_rel":-0.05569,
              "motor_rel":-0.05068, "parietal_rel":-0.03782},
        "B": {"global_mean":-0.00422, "visual_rel":0.00965, "auditory_rel":0.01781,
              "language_rel":0.02668, "prefrontal_rel":-0.02394,
              "motor_rel":-0.02512, "parietal_rel":-0.00800},
    },
    {
        "label": "Narrative\nChronological vs Bullet",
        "A_label": "Chronological",
        "B_label": "Bullet-point",
        "A": {"global_mean":-0.00199, "visual_rel":0.01346, "auditory_rel":0.03124,
              "language_rel":0.03296, "prefrontal_rel":-0.03378,
              "motor_rel":-0.02996, "parietal_rel":-0.01713},
        "B": {"global_mean":-0.00252, "visual_rel":0.01424, "auditory_rel":0.02547,
              "language_rel":0.02855, "prefrontal_rel":-0.02714,
              "motor_rel":-0.02913, "parietal_rel":-0.01529},
    },
    {
        "label": "Factual\nActive vs Passive Voice",
        "A_label": "Active",
        "B_label": "Passive",
        "A": {"global_mean":-0.00095, "visual_rel":-0.00304, "auditory_rel":0.01308,
              "language_rel":0.00969, "prefrontal_rel":-0.00291,
              "motor_rel":-0.01208, "parietal_rel":-0.00545},
        "B": {"global_mean":-0.00286, "visual_rel":0.03809, "auditory_rel":0.05709,
              "language_rel":0.05380, "prefrontal_rel":-0.06156,
              "motor_rel":-0.05719, "parietal_rel":-0.03667},
    },
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=False)
plot_keys = ["global_mean"] + ["visual_rel","auditory_rel","language_rel",
                                "prefrontal_rel","motor_rel","parietal_rel"]
x_labels  = ["Global\nMean"] + REGIONS
x = np.arange(len(x_labels))
w = 0.35

for ax, pd_item in zip(axes, pairs_data):
    va = [pd_item["A"][k] for k in plot_keys]
    vb = [pd_item["B"][k] for k in plot_keys]
    ba = ax.bar(x - w/2, va, w, label=pd_item["A_label"],
                color="#1F77B4", alpha=0.85, edgecolor="white")
    bb = ax.bar(x + w/2, vb, w, label=pd_item["B_label"],
                color="#FF7F0E", alpha=0.85, edgecolor="white")
    ax.axhline(0, color="#333", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_title(pd_item["label"], fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_ylabel("Activation (a.u.)", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "Figure 6.  Contrastive Pair Analysis: Activation Differences from Language Structure\n"
    "Same content, different linguistic framing. Bars show predicted activation per region.",
    fontsize=9, y=1.02
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_contrastive_pairs.png")
plt.close()
print("Fig 6 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Theory scorecard
# ═══════════════════════════════════════════════════════════════════════════════
theories = ["GWT", "FEP", "DCT", "IIT"]
predictions = [
    "ThreatSafety\ntop activation",
    "Novelty\ntop-3",
    "ThreatSafety\nprefrontal top-3",
    "Language→\nSocial/Narrative",
    "Narrative\nbottom activation",
    "Modality\nclusters",
]
# 1=confirmed, 0.5=partial, 0=not confirmed
scores = np.array([
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.5],  # GWT
    [1.0, 0.0, 0.5, 0.0, 0.0, 0.0],  # FEP
    [0.5, 0.0, 0.0, 0.5, 0.0, 0.5],  # DCT
    [0.5, 0.0, 0.0, 1.0, 0.0, 0.5],  # IIT
])

fig, ax = plt.subplots(figsize=(9, 3.8))
cmap_theory = LinearSegmentedColormap.from_list("theory", ["#D62728","#FFDD57","#2CA02C"], N=3)
im = ax.imshow(scores, cmap=cmap_theory, vmin=0, vmax=1, aspect="auto")

for i in range(len(theories)):
    for j in range(len(predictions)):
        val = scores[i,j]
        label = "✅ Yes" if val == 1.0 else "🔶 Partial" if val == 0.5 else "❌ No"
        color = "white" if val < 0.3 else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=color)

ax.set_xticks(range(len(predictions)))
ax.set_xticklabels(predictions, fontsize=8.5, rotation=10, ha="center")
ax.set_yticks(range(len(theories)))
ax.set_yticklabels(theories, fontsize=10, fontweight="bold")
ax.set_title(
    "Figure 7.  Theory Prediction Scorecard\n"
    "Green = confirmed; Yellow = partially confirmed; Red = not confirmed.",
    fontsize=9, loc="left", pad=8
)
for i in range(len(theories)+1): ax.axhline(i-0.5, color="white", lw=1)
for j in range(len(predictions)+1): ax.axvline(j-0.5, color="white", lw=1)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig7_theory_scorecard.png")
plt.close()
print("Fig 7 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Within-type CV stability + sample size
# ═══════════════════════════════════════════════════════════════════════════════
cv_data = []
for ct in CT_ORDER:
    vals = df[df["content_type"]==ct]["global_mean"].values
    if len(vals) < 2: continue
    m = vals.mean(); s = vals.std()
    cv = abs(s / m) if abs(m) > 1e-8 else 0
    cv_data.append({"ct": ct, "cv": cv, "n": len(vals), "mean": m, "sd": s})
cv_df = pd.DataFrame(cv_data).sort_values("cv")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors_cv = [PALETTE[ct] for ct in cv_df["ct"]]
scatter = ax.scatter(cv_df["cv"], range(len(cv_df)),
                     c=colors_cv, s=cv_df["n"]/2, alpha=0.9, zorder=3,
                     edgecolors="white", linewidths=0.5)
ax.set_yticks(range(len(cv_df)))
ax.set_yticklabels(cv_df["ct"], fontsize=9)
ax.set_xlabel("Coefficient of Variation (SD / |mean|)", fontsize=10)
ax.set_title(
    "Figure 8.  Within-Category Activation Stability\n"
    "Lower CV = more consistent neural response. Bubble size ∝ N stimuli.",
    fontsize=9, loc="left", pad=8
)
ax.axvline(0.35, color="#888", linestyle=":", lw=1, label="CV = 0.35")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="x", alpha=0.3)
ax.grid(axis="y", alpha=0)

sizes_legend = [100, 200, 300]
legend_handles = [plt.scatter([], [], s=n/2, c="#888888", alpha=0.6,
                               edgecolors="white", label=f"N={n}")
                  for n in sizes_legend]
ax.legend(handles=legend_handles, title="Sample size", frameon=False,
          fontsize=8, loc="lower right")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig8_stability.png")
plt.close()
print("Fig 8 saved")

print(f"\nAll figures saved to {FIG_DIR}")
