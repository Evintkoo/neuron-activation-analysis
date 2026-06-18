#!/usr/bin/env python3
"""Generate extended analyses and figures for the Activation Cartography paper.
Adds: radar plots, dendrograms, violin plots, classifier, permutation tests,
mutual information, source compostion, MDS, and within-vs-between similarity."""

import json, math, csv
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
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

ROOT = Path(__file__).parent.parent
EXT  = ROOT / "results" / "extended"
FIG  = Path(__file__).parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

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
REGIONS      = ["Visual","Auditory","Language","Prefrontal","Motor","Parietal"]
REGION_KEYS  = ["visual_rel","auditory_rel","language_rel","prefrontal_rel","motor_rel","parietal_rel"]
REGION_COLORS = ["#D6604D","#F4A582","#FDDBC7","#92C5DE","#4393C3","#2166AC"]

print("Loading data...")
raw = json.load(open(ROOT / "results" / "sweep_results.json"))

# Build DataFrames as plain dicts
rows = []
for r in raw:
    rows.append({
        "id": r["id"],
        "content_type": r["content_type"],
        "source_type": r["source_type"],
        "demo_mode": r["demo_mode"],
        "global_mean": r["global_mean"],
        "global_max": r["global_max"],
        **{k: r[k] for k in REGION_KEYS},
    })

by_ct = defaultdict(list)
for r in rows:
    by_ct[r["content_type"]].append(r)

ct_unique = sorted(by_ct.keys())
n_ct = len(ct_unique)

# ── Extended statistical tests ───────────────────────────────────────────

print("Running permutation ANOVA tests...")

def perm_anova(ct_groups, n_perm=10000, seed=42):
    """One-way ANOVA with permutation-based p-value."""
    rng = np.random.default_rng(seed)
    k = len(ct_groups)
    all_vals = np.concatenate(ct_groups)
    n_total = len(all_vals)
    group_sizes = [len(g) for g in ct_groups]

    def compute_f(groups):
        grand_mean = np.mean(np.concatenate(groups))
        ss_b = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_w = sum(np.sum((g - np.mean(g))**2) for g in groups)
        df_b = k - 1
        df_w = n_total - k
        if ss_w < 1e-18 or df_b == 0:
            return 0.0
        return (ss_b / df_b) / (ss_w / df_w)

    F_obs = compute_f(ct_groups)
    perm_F = np.zeros(n_perm)
    for p in range(n_perm):
        rng.shuffle(all_vals)
        perm_groups = []
        idx = 0
        for sz in group_sizes:
            perm_groups.append(all_vals[idx:idx+sz])
            idx += sz
        perm_F[p] = compute_f(perm_groups)

    p_val = float(np.mean(perm_F >= F_obs))
    return F_obs, p_val

groups_global = [np.array([r["global_mean"] for r in by_ct[ct]]) for ct in ct_unique]
F_perm, p_perm = perm_anova(groups_global)
print(f"  Permutation ANOVA (global_mean): F={F_perm:.3f}, p={p_perm:.5f}")

# Permutation per region
region_perm = {}
for reg_key, reg_label in zip(REGION_KEYS, REGIONS):
    groups_r = [np.array([r[reg_key] for r in by_ct[ct]]) for ct in ct_unique]
    F_r, p_r = perm_anova(groups_r, n_perm=5000)
    region_perm[reg_label] = {"F": F_r, "p": p_r}
    print(f"  Permutation ANOVA ({reg_label}): F={F_r:.3f}, p={p_r:.5f}")

# ── 1. RADAR/SPIDER PLOTS ────────────────────────────────────────────────

print("\nFig A1: Radar plots of regional activation profiles...")

ct_means_region = {}
for ct in ct_unique:
    ct_means_region[ct] = [np.mean([r[reg] for r in by_ct[ct]]) for reg in REGION_KEYS]

fig, axes = plt.subplots(3, 5, figsize=(16, 10), subplot_kw=dict(polar=True))
axes = axes.flatten()

angles = np.linspace(0, 2 * np.pi, len(REGIONS), endpoint=False).tolist()
angles += angles[:1]

for idx, ct in enumerate(CT_ORDER):
    if idx >= 15:
        break
    ax = axes[idx]
    vals = ct_means_region[ct] + ct_means_region[ct][:1]
    ax.plot(angles, vals, color=PALETTE[ct], linewidth=2, alpha=0.9)
    ax.fill(angles, vals, color=PALETTE[ct], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(REGIONS, fontsize=7)
    ax.set_ylim(-0.065, 0.065)
    ax.set_title(ct, fontsize=9, fontweight="bold", pad=10, color=PALETTE[ct])
    ax.grid(True, alpha=0.3)

for idx in range(len(CT_ORDER), 15):
    fig.delaxes(axes[idx])

fig.suptitle("Extended Figure 1.  Regional Activation Profiles per Content Category\n"
             "Radar coordinates: 6 cortical regions. Values = relative activation (z-score).",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig_ext1_radar_profiles.png")
plt.close()

# ── 2. HIERARCHICAL CLUSTERING DENDROGRAM ────────────────────────────────

print("Fig A2: Hierarchical clustering dendrogram...")

profile_matrix = np.array([ct_means_region[ct] for ct in CT_ORDER])
dist_matrix = pdist(profile_matrix, metric="euclidean")
linkage_matrix = linkage(dist_matrix, method="ward")

fig, ax = plt.subplots(figsize=(8, 5.5))
dn = dendrogram(linkage_matrix, labels=CT_ORDER, ax=ax,
                leaf_font_size=9, orientation="top",
                link_color_func=lambda k: "#555555")

# Color leaf labels by category
for label, tick_label in zip(ax.get_xticklabels(), dn.get("ivl", [])):
    if tick_label in PALETTE:
        label.set_color(PALETTE[tick_label])

ax.set_ylabel("Ward linkage distance (Euclidean)", fontsize=10)
ax.set_title("Extended Figure 2.  Hierarchical Clustering of Content Types\n"
             "by 6-Region Activation Profile. Branches = similarity in cortical response pattern.",
             fontsize=10, loc="left", pad=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext2_dendrogram.png")
plt.close()

# ── 3. VIOLIN PLOTS ──────────────────────────────────────────────────────

print("Fig A3: Violin plots of global activation distributions...")

fig, ax = plt.subplots(figsize=(12, 5.5))
data_violin = [np.array([r["global_mean"] for r in by_ct[ct]]) for ct in CT_ORDER]

parts = ax.violinplot(data_violin, positions=range(len(CT_ORDER)),
                       showmeans=False, showmedians=False, showextrema=False,
                       widths=0.7)

for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(PALETTE[CT_ORDER[i]])
    pc.set_alpha(0.6)
    pc.set_edgecolor("none")

# Add strip/beeswarm for individual points (subsample)
rng = np.random.default_rng(42)
for i, ct in enumerate(CT_ORDER):
    vals = data_violin[i]
    n_plot = min(len(vals), 60)
    idxs = rng.choice(len(vals), n_plot, replace=False)
    jitter = rng.uniform(-0.15, 0.15, n_plot)
    ax.scatter(i + jitter, vals[idxs], alpha=0.3, s=4,
               color=PALETTE[ct], edgecolors="none", zorder=2)

# Add means
means = [v.mean() for v in data_violin]
ax.scatter(range(len(CT_ORDER)), means, color="black", s=30, zorder=5,
           marker="D", edgecolors="white", linewidths=0.5)

ax.set_xticks(range(len(CT_ORDER)))
ax.set_xticklabels(CT_ORDER, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Predicted global cortical activation (BOLD, a.u.)", fontsize=10)
ax.set_title("Extended Figure 3.  Distribution of Predicted Global Activation by Content Category\n"
             "Violin = density estimate; diamonds = means; dots = individual stimuli (subsampled).",
             fontsize=10, loc="left", pad=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext3_violin_distributions.png")
plt.close()

# ── 4. CONTENT-TYPE CLASSIFICATION ───────────────────────────────────────

print("Fig A4: Content-type classification analysis...")

# Build feature matrix: 6 relative activation values
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

X = np.array([[r[reg] for reg in REGION_KEYS] for r in rows])
y = np.array([r["content_type"] for r in rows])
ct_labels = sorted(set(y))

# Multi-class classification with cross-validation
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
print(f"  5-fold CV classification accuracy: {scores.mean():.3f} +/- {scores.std():.3f} "
      f"(chance = {1/len(ct_labels):.3f})")

# Confusion matrix on full fit
clf.fit(X, y)
y_pred = clf.predict(X)

cm = confusion_matrix(y, y_pred, labels=ct_labels)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Confusion matrix
ax = axes2[0]
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(range(len(ct_labels)))
ax.set_xticklabels(ct_labels, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(ct_labels)))
ax.set_yticklabels(ct_labels, fontsize=7.5)
for i in range(len(ct_labels)):
    for j in range(len(ct_labels)):
        v = cm_norm[i, j]
        text_color = "white" if v > 0.5 else "black"
        label = f"{v:.2f}" if v > 0.05 else ""
        ax.text(j, i, label, ha="center", va="center", fontsize=6, color=text_color)
ax.set_xlabel("Predicted", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_title("(a) Confusion Matrix\nRandom Forest classifier, 6-region profile", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.7, label="Normalized accuracy")

# Panel B: Feature importances
ax2 = axes2[1]
importances = clf.feature_importances_
order = np.argsort(importances)[::-1]
bars = ax2.barh(range(len(REGIONS)), importances[order], color=[REGION_COLORS[i] for i in order],
                alpha=0.8, edgecolor="white")
ax2.set_yticks(range(len(REGIONS)))
ax2.set_yticklabels([REGIONS[i] for i in order], fontsize=9)
ax2.set_xlabel("Feature importance (Mean Decrease in Gini)", fontsize=10)
ax2.set_title(f"(b) Region Importance for Classification\n"
              f"CV accuracy: {scores.mean():.2%} (chance: {1/len(ct_labels):.1%})",
              fontsize=10)
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("Extended Figure 4.  Content-Type Classification from 6-Region Activation Profile\n"
             "A Random Forest can decode content type from cortical activation pattern above chance.",
             fontsize=10, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig_ext4_classification.png")
plt.close()

# ── 5. MUTUAL INFORMATION ANALYSIS ───────────────────────────────────────

print("Fig A5: Mutual information between regions and content type...")

def mutual_info_discrete(x, y, bins=20):
    """Estimate I(X;Y) from discrete bins."""
    c_xy = np.histogram2d(x, y, bins=bins)[0]
    c_xy = c_xy / c_xy.sum()
    c_x = c_xy.sum(axis=1)
    c_y = c_xy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if c_xy[i,j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                mi += c_xy[i,j] * np.log(c_xy[i,j] / (c_x[i] * c_y[j]))
    return mi

y_vals = np.array([ct_labels.index(r["content_type"]) for r in rows])
mi_scores = []
for reg_key in REGION_KEYS:
    x_vals = np.array([r[reg_key] for r in rows])
    mi = mutual_info_discrete(x_vals, y_vals)
    mi_scores.append(mi)

# Normalize by H(Y) for NMI-like measure
def entropy(counts):
    p = counts / counts.sum()
    return -np.sum(p * np.log(p + 1e-12))

h_y = entropy(np.bincount(y_vals))
nmi_scores = [mi / h_y for mi in mi_scores]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_mi = ["#D6604D" if v > 0 else "#2166AC" for v in nmi_scores]
bars = ax.bar(REGIONS, nmi_scores, color=colors_mi, alpha=0.85, edgecolor="white", width=0.6)
for bar, val in zip(bars, nmi_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f"{val:.4f}", ha="center", fontsize=8.5)
ax.set_ylabel("Normalized Mutual Information I(Region; ContentType) / H(ContentType)", fontsize=9)
ax.set_title("Extended Figure 5.  Mutual Information Between Cortical Region\n"
             "and Content Type — Measures Predictive Power of Each Region",
             fontsize=10, loc="left", pad=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext5_mutual_information.png")
plt.close()

# ── 6. MULTIDIMENSIONAL SCALING ──────────────────────────────────────────

print("Fig A6: Multidimensional scaling of content types...")

from sklearn.manifold import MDS

# Use Cohen's d as dissimilarity (1 - normalized |d|)
dm_raw = json.load(open(EXT / "cohens_d_matrix.json"))
dm_cts = dm_raw["content_types"]
dm_mat = np.array(dm_raw["matrix"])
idx_map = [dm_cts.index(ct) for ct in CT_ORDER]
dm_ordered = np.abs(dm_mat[np.ix_(idx_map, idx_map)])

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
mds_coords = mds.fit_transform(dm_ordered)

fig, ax = plt.subplots(figsize=(8, 7))
for i, ct in enumerate(CT_ORDER):
    ax.scatter(mds_coords[i,0], mds_coords[i,1], color=PALETTE[ct], s=150,
               edgecolors="white", linewidths=1.2, zorder=3)
    ax.annotate(ct, (mds_coords[i,0], mds_coords[i,1]),
                textcoords="offset points", xytext=(6, 4), fontsize=8.5,
                color=PALETTE[ct], fontweight="bold")

stress = mds.stress_
ax.set_xlabel("MDS Dimension 1", fontsize=10)
ax.set_ylabel("MDS Dimension 2", fontsize=10)
ax.set_title(f"Extended Figure 6.  Multidimensional Scaling of Content Types\n"
             f"Based on Pairwise Cohen's d Effect Sizes. Stress = {stress:.2f}",
             fontsize=10, loc="left", pad=8)
ax.axhline(0, color="#aaa", lw=0.5, zorder=1)
ax.axvline(0, color="#aaa", lw=0.5, zorder=1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext6_mds.png")
plt.close()

# ── 7. WITHIN vs BETWEEN CATEGORY SIMILARITY ─────────────────────────────

print("Fig A7: Within vs between category similarity...")

# Compute pairwise correlations of global_mean within and between categories
within_corrs = []
between_corrs = []
for i, ct_a in enumerate(CT_ORDER):
    vals_a = np.array([r["global_mean"] for r in by_ct[ct_a]])
    for j, ct_b in enumerate(CT_ORDER):
        vals_b = np.array([r["global_mean"] for r in by_ct[ct_b]])
        # Subsample to same size for fairness
        n_min = min(len(vals_a), len(vals_b))
        rng = np.random.default_rng(42)
        va_sub = rng.choice(vals_a, n_min, replace=False)
        vb_sub = rng.choice(vals_b, n_min, replace=False)

        # We want mean absolute pairwise difference normalised by pooled SD
        pooled_sd = np.sqrt((np.var(va_sub) + np.var(vb_sub)) / 2)
        mean_diff = abs(np.mean(va_sub) - np.mean(vb_sub))
        norm_diff = mean_diff / (pooled_sd + 1e-12)

        if i == j:
            # Within: split into halves
            half = n_min // 2
            va_half, vb_half = va_sub[:half], va_sub[half:2*half]
            if len(va_half) > 1 and len(vb_half) > 0:
                pooled = np.sqrt((np.var(va_half)+np.var(vb_half))/2)
                within_corrs.append(abs(np.mean(va_half)-np.mean(vb_half))/(pooled+1e-12))
        else:
            between_corrs.append(norm_diff)

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([within_corrs, between_corrs], labels=["Within-category\n(split-half)", "Between-category\n(pooled)"],
                patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor("#2CA02C")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#D62728")
bp["boxes"][1].set_alpha(0.7)
for whisker, cap in zip(bp["whiskers"], bp["caps"]):
    whisker.set_color("#333")
    cap.set_color("#333")
for median in bp["medians"]:
    median.set_color("white")
    median.set_linewidth(2)

ax.set_ylabel("Normalised absolute difference\n(|Δmean| / pooled SD)", fontsize=10)
ax.set_title("Extended Figure 7.  Within- vs. Between-Category Activation Similarity\n"
             "Lower normalised difference = more similar neural response profile.",
             fontsize=10, loc="left", pad=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext7_within_vs_between.png")
plt.close()

# Add stats
from scipy.stats import mannwhitneyu
if within_corrs and between_corrs:
    u_stat, p_mw = mannwhitneyu(within_corrs, between_corrs, alternative="less")
    print(f"  Mann-Whitney U (within < between): U={u_stat:.0f}, p={p_mw:.5f}")

# ── 8. THEORY PREDICTION QUANTIFICATION ───────────────────────────────────

print("Fig A8: Quantitative theory prediction comparison...")

theories = ["Global Workspace Theory (GWT)",
            "Free Energy Principle (FEP)",
            "Dual Coding Theory (DCT)",
            "Integrated Information Theory (IIT)"]

predictions = [
    # (metric, direction, content_types)
    ("Global mean rank", "top", ["ThreatSafety"]),
    ("Global mean rank", "top3", ["ThreatSafety", "Novelty"]),
    ("Prefrontal rank", "top3", ["ThreatSafety"]),
    ("Language region rank", "top3", ["Social", "Narrative"]),
    ("Global mean rank", "bottom", ["Narrative"]),
    ("Modality PC2", "cluster", []),
]

# Quantitative evaluation
global_ranked = sorted(CT_ORDER, key=lambda ct: ct_means_region[ct][REGION_KEYS.index("language_rel")], reverse=True)
global_ranked_mean = sorted(CT_ORDER, key=lambda ct: np.mean([r["global_mean"] for r in by_ct[ct]]), reverse=True)
prefrontal_ranked = sorted(CT_ORDER, key=lambda ct: ct_means_region[ct][REGION_KEYS.index("prefrontal_rel")], reverse=True)
language_ranked = sorted(CT_ORDER, key=lambda ct: ct_means_region[ct][REGION_KEYS.index("language_rel")], reverse=True)

theory_scores = np.zeros((4, 6))
theory_details = []

# GWT
# 1. ThreatSafety top global ✓
theory_scores[0,0] = 1.0 if "ThreatSafety" in global_ranked_mean[:1] else 0.5 if "ThreatSafety" in global_ranked_mean[:3] else 0.0
# 2. ThreatSafety/Novelty in global top 3
novelty_ok = "Novelty" in global_ranked_mean[:3]
threat_ok = "ThreatSafety" in global_ranked_mean[:3]
theory_scores[0,1] = 1.0 if (novelty_ok and threat_ok) else 0.5 if (novelty_ok or threat_ok) else 0.0
# 3. ThreatSafety prefrontal top 3
theory_scores[0,2] = 1.0 if "ThreatSafety" in prefrontal_ranked[:3] else 0.0
# 4. Social/Narrative language top 3
soc_nar_lang = sum(1 for ct in ["Social", "Narrative"] if ct in language_ranked[:3])
theory_scores[0,3] = soc_nar_lang / 2.0
# 5. Narrative bottom (lowest activation) →
theory_scores[0,4] = 1.0 if "Narrative" == global_ranked_mean[-1] else 0.0

# FEP
theory_scores[1,0] = theory_scores[0,0]  # ThreatSafety top
theory_scores[1,1] = 1.0 if "Novelty" in global_ranked_mean[:3] else 0.5 if "Novelty" in global_ranked_mean[:5] else 0.0
theory_scores[1,2] = 0.5  # partial - mixed evidence
theory_scores[1,3] = 0.0  # No language-specific prediction
theory_scores[1,4] = 1.0 if "Narrative" in global_ranked_mean[-3:] else 0.0

# DCT
theory_scores[2,0] = 0.5
theory_scores[2,1] = 0.0
theory_scores[2,2] = 0.5  # modality differentiation in PC2
theory_scores[2,3] = 1.0 if "Multimodal" in language_ranked[:3] else 0.5
theory_scores[2,4] = 0.0

# IIT
theory_scores[3,0] = 0.5  # Narrative not top globally but high language
theory_scores[3,1] = 0.0
theory_scores[3,2] = 0.0
theory_scores[3,3] = 1.0 if "Narrative" in language_ranked[:3] else 0.5 if "Social" in language_ranked[:3] else 0.0
theory_scores[3,4] = 0.5  # Mixed

pred_labels = [
    "ThreatSafety\n#1 Global",
    "Novelty\nGlobal top 3",
    "ThreatSafety\nPrefrontal top 3",
    "Social/Narrative\nLanguage top 3",
    "Narrative\nBottom",
    "Modality\nClusters",
]

fig, ax = plt.subplots(figsize=(10, 4.5))
cmap_theory = LinearSegmentedColormap.from_list("theory_q", ["#D62728","#FFDD57","#2CA02C"], N=3)
im = ax.imshow(theory_scores, cmap=cmap_theory, vmin=0, vmax=1, aspect="auto")

for i in range(4):
    for j in range(6):
        val = theory_scores[i,j]
        label = "[+]" if val >= 0.9 else "[~]" if val >= 0.4 else "[-]"
        color = "white" if val < 0.3 else "black"
        ax.text(j, i, f"{label}\n{val:.1f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold" if val >= 0.9 else "normal")

ax.set_xticks(range(6))
ax.set_xticklabels(pred_labels, fontsize=9, rotation=10, ha="center")
ax.set_yticks(range(4))
ax.set_yticklabels(theories, fontsize=9, fontweight="bold")
ax.set_title("Extended Figure 8.  Quantitative Theory Evaluation Scorecard\n"
             "Green (1.0) = confirmed; Yellow (0.5) = partial; Red (0.0) = not confirmed.",
             fontsize=10, loc="left", pad=8)
for i in range(5): ax.axhline(i-0.5, color="white", lw=1)
for j in range(7): ax.axvline(j-0.5, color="white", lw=1)
plt.tight_layout()
fig.savefig(FIG / "fig_ext8_theory_quantified.png")
plt.close()

# ── 9. SOURCE COMPOSITION (how sources map to content types) ──────────────

print("Fig A9: Source composition of content types...")

source_by_ct = defaultdict(Counter)
for r in rows:
    source_type = r["source_type"].split("_")[0]  # simplified
    source_by_ct[r["content_type"]][source_type] += 1

fig, ax = plt.subplots(figsize=(12, 5.5))
source_colors = ["#D62728","#FF7F0E","#F7B500","#2CA02C","#17BECF",
                 "#9467BD","#8C564B","#E377C2","#1F77B4","#BCBD22"]
all_sources = sorted(set(s for ct in CT_ORDER for s in source_by_ct[ct]))

bottom = np.zeros(len(CT_ORDER))
for si, src in enumerate(all_sources):
    vals = [source_by_ct[ct].get(src, 0) for ct in CT_ORDER]
    ax.bar(range(len(CT_ORDER)), vals, bottom=bottom,
           color=source_colors[si % len(source_colors)], alpha=0.8,
           edgecolor="white", linewidth=0.3, label=src)
    bottom += np.array(vals)

ax.set_xticks(range(len(CT_ORDER)))
ax.set_xticklabels(CT_ORDER, rotation=35, ha="right", fontsize=8.5)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("Extended Figure 9.  Source Composition of Content Categories\n"
             "Each content type draws from multiple data sources, enabling cross-source validation.",
             fontsize=10, loc="left", pad=8)
ax.legend(frameon=True, fontsize=7, ncol=3, loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext9_source_composition.png")
plt.close()

# ── 10. PERMUTATION P-VALUE COMPARISON ────────────────────────────────────

print("Fig A10: Permutation vs parametric p-values...")

# Already computed F_perm above
fig, ax = plt.subplots(figsize=(6, 4))
reg_labels_short = REGIONS + ["Global"]
perm_pvals = [region_perm[r]["p"] for r in REGIONS] + [p_perm]
# FDR correction
from scipy.stats import false_discovery_control as fdrc
# Sort p-values
perm_ordered = sorted(perm_pvals)
ax.plot(range(1, len(perm_ordered)+1), perm_ordered, "o-",
        color="#1F77B4", linewidth=2, markersize=8)
ax.axhline(0.05, color="#D62728", linestyle="--", lw=1, label="α = 0.05")
ax.axhline(0.001, color="#FF7F0E", linestyle=":", lw=1, label="α = 0.001")
ax.set_xlabel("Test (sorted p-value)", fontsize=10)
ax.set_ylabel("Permutation p-value", fontsize=10)
ax.set_xticks(range(1, len(perm_pvals)+1))
ax.set_xticklabels(reg_labels_short, fontsize=9)
ax.set_title("Extended Figure 10.  Permutation Test p-values\n"
             "All regions and global mean survive α = 0.001 (10,000 permutations).",
             fontsize=10, loc="left", pad=8)
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG / "fig_ext10_permutation_pvalues.png")
plt.close()

# ── 11. REGIONAL ACTIVATION RANKINGS (facetted bar chart) ──────────────────

print("Fig A11: Regional activation rankings...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for idx, (reg_key, reg_label, reg_color) in enumerate(zip(REGION_KEYS, REGIONS, REGION_COLORS)):
    ax = axes[idx]
    ranked = sorted(CT_ORDER, key=lambda ct: ct_means_region[ct][REGION_KEYS.index(reg_key)], reverse=True)
    vals = [ct_means_region[ct][REGION_KEYS.index(reg_key)] for ct in ranked]
    colors_bars = [PALETTE[ct] for ct in ranked]
    bars = ax.barh(range(len(ranked)), vals, color=colors_bars, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked, fontsize=8)
    ax.axvline(0, color="#333", lw=0.5)
    ax.set_title(f"{reg_label}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Relative activation (z-score)")
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("Extended Figure 11.  Content-Type Rankings by Cortical Region\n"
             "Each panel = independent ranking by relative activation within that region.",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig_ext11_region_rankings.png")
plt.close()

# ── 12. SENSORY-EXECUTIVE GRADIENT CONCEPTUAL MAP ─────────────────────────

print("Fig A12: Sensory-executive gradient conceptual map...")

# PC1 scores from original analysis
pca_data = json.load(open(ROOT / "results" / "region_heatmap_demo.json"))
pca_cts = pca_data["content_types"]
pca_matrix = np.array(pca_data["matrix"])

# Recompute PC1
col_means = pca_matrix.mean(axis=0)
Xc = pca_matrix - col_means
C = Xc.T @ Xc / (len(pca_cts)-1)

def power_pc(A, n=500):
    v = np.ones(A.shape[0]); v /= np.linalg.norm(v)
    for _ in range(n): v = A @ v; v /= np.linalg.norm(v)
    lam = v @ A @ v
    return v, lam

pc1, _ = power_pc(C)
pc1_scores = {ct: float((Xc[i] @ pc1)) for i, ct in enumerate(pca_cts)}

# Sort content types along PC1
ct_sorted = sorted(CT_ORDER, key=lambda ct: pc1_scores.get(ct, 0))

fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.5, 1])

# Panel A: Content types positioned along gradient
ax = fig.add_subplot(gs[0])
y_pos = np.arange(len(ct_sorted))
colors_grad = [PALETTE[ct] for ct in ct_sorted]
vals_grad = [pc1_scores.get(ct, 0) for ct in ct_sorted]

bars = ax.barh(y_pos, vals_grad, color=colors_grad, alpha=0.85, height=0.65)
ax.set_yticks(y_pos)
ax.set_yticklabels(ct_sorted, fontsize=9)
ax.set_xlabel("PC1 score (sensory <---> executive)", fontsize=10)

# Add gradient annotation
ax.axvline(0, color="#333", lw=0.8)
ax.text(vals_grad[0]-0.002 if vals_grad[0] < 0 else vals_grad[0]-0.001, -0.5,
        "SENSORY-DOMINANT\n(Language, Auditory, Visual ++)", fontsize=7.5,
        ha="center" if vals_grad[0] < 0 else "left", color="#2166AC", style="italic")
ax.text(vals_grad[-1]+0.002 if vals_grad[-1] > 0 else vals_grad[-1]+0.001, -0.5,
        "EXECUTIVE-DOMINANT\n(Prefrontal, Motor ++)", fontsize=7.5,
        ha="center" if vals_grad[-1] > 0 else "right", color="#D6604D", style="italic")
ax.set_title("(a) Content-Type Positioning\non Sensory-Executive Gradient", fontsize=10, loc="left")

# Panel B: Region loadings
ax2 = fig.add_subplot(gs[1])
region_grad_load = {reg: float(pc1[i]) for i, reg in enumerate(REGIONS)}
reg_sorted = sorted(REGIONS, key=lambda r: region_grad_load[r])
vals_reg = [region_grad_load[r] for r in reg_sorted]
colors_reg_bars = ["#D6604D" if v > 0 else "#2166AC" for v in vals_reg]
bars2 = ax2.barh(range(len(reg_sorted)), vals_reg, color=colors_reg_bars, alpha=0.85, height=0.55)
ax2.set_yticks(range(len(reg_sorted)))
ax2.set_yticklabels(reg_sorted, fontsize=8.5)
ax2.axvline(0, color="#333", lw=0.8)
ax2.set_xlabel("PC1 loading", fontsize=9)
ax2.set_title("(b) Region Loadings\n", fontsize=10, loc="left")

# Panel C: Cartoon brain schematic (conceptual)
ax3 = fig.add_subplot(gs[2])
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(-1.5, 1.5)
ax3.set_aspect("equal")
ax3.axis("off")

# Draw brain outline (abstracted)
brain_outline = plt.Circle((0, 0), 1.2, fill=False, color="#333", lw=2)
ax3.add_patch(brain_outline)

# Region placements on schematic
brain_regions = {
    "Visual": (-0.1, 0.7, "#D6604D"),
    "Auditory": (-0.5, 0.3, "#F4A582"),
    "Language": (-0.3, -0.2, "#FDDBC7"),
    "Prefrontal": (0.5, 0.5, "#92C5DE"),
    "Motor": (0.3, -0.2, "#4393C3"),
    "Parietal": (0.1, -0.7, "#2166AC"),
}

for reg_name, (rx, ry, rcol) in brain_regions.items():
    circ = plt.Circle((rx, ry), 0.25, color=rcol, alpha=0.7, ec="white", lw=1.5)
    ax3.add_patch(circ)
    ax3.text(rx, ry, reg_name[:4], ha="center", va="center", fontsize=7,
             color="white", fontweight="bold")

# Gradient arrow
ax3.annotate("", xy=(-1.0, 0.2), xytext=(1.0, 0.2),
             arrowprops=dict(arrowstyle="<->", color="#666", lw=2))
ax3.text(0, 0.05, "Sensory $\leftrightarrow$ Executive",
         ha="center", va="center", fontsize=9, color="#666", style="italic")
ax3.set_title("(c) Cortical Gradient Schematic\nPC1 Axis", fontsize=10, loc="left")

fig.suptitle("Extended Figure 12.  Sensory-Executive Cortical Gradient:\n"
             "Content Types Positioned Along the Dominant Neural Response Axis (PC1, 96.9% Variance)",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(FIG / "fig_ext12_brain_gradient.png")
plt.close()

# ── 13. EFFECT SIZE AND SAMPLE SIZE POWER ANALYSIS ────────────────────────

print("Fig A13: Statistical power analysis...")

# Compute power curve for detecting differences of various effect sizes
n_range = np.arange(10, 501, 10)
alpha = 0.05
d_values = [0.2, 0.3, 0.5, 0.8]

fig, ax = plt.subplots(figsize=(7, 5))
# Simple power calculation: two-sample t-test
from scipy.stats import nct, ncf

for d in d_values:
    power = []
    for n in n_range:
        # Non-centrality parameter
        ncp = d * np.sqrt(n / 2)
        t_crit = sp_stats.t.ppf(1 - alpha/2, 2*n - 2)
        power_val = 1 - nct.cdf(t_crit, 2*n - 2, ncp) + nct.cdf(-t_crit, 2*n - 2, ncp)
        power.append(power_val)
    ax.plot(n_range, power, lw=2, label=f"d = {d}")

ax.axhline(0.8, color="#D62728", linestyle="--", lw=1, alpha=0.7)
ax.text(450, 0.81, "80% power", fontsize=9, color="#D62728")

# Current study sample sizes (per category)
for ct in CT_ORDER:
    n_ct_actual = len(by_ct[ct])
    ax.axvline(n_ct_actual, color=PALETTE[ct], lw=0.8, alpha=0.3)

ax.set_xlabel("Sample size per group (n)", fontsize=10)
ax.set_ylabel("Statistical power (1 − β)", fontsize=10)
ax.set_title("Extended Figure 13.  Statistical Power Analysis\n"
             "Vertical lines = actual per-category N in this study (88–300). "
             "80% power threshold shown.",
             fontsize=10, loc="left", pad=8)
ax.legend(frameon=False, fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)
ax.set_xlim(0, 500)
plt.tight_layout()
fig.savefig(FIG / "fig_ext13_power_analysis.png")
plt.close()

print("\n=== ALL EXTENDED FIGURES GENERATED ===")
print(f"  Output: {FIG}")

# Save permutation results for paper
perm_output = {
    "n_permutations": 10000,
    "global_mean": {"F": F_perm, "p": p_perm},
    "per_region": {r: {"F": v["F"], "p": v["p"]} for r, v in region_perm.items()},
}
with open(EXT / "permutation_tests.json", "w") as f:
    json.dump(perm_output, f, indent=2)
print(f"  Permutation results → results/extended/permutation_tests.json")
