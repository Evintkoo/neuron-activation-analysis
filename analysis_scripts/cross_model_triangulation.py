#!/usr/bin/env python3
"""
Cross-model triangulation:
  Part A — TRIBE seq_len robustness (seq_len=4, 8, 16)
  Part B — BERT (distilbert-base-uncased) text-embedding proxy
  Part C — Cross-encoder RDM (Mantel correlation)

Usage:
    python3 analysis_scripts/cross_model_triangulation.py [seqsweep|bert|rdm|all]
"""
import json, os, sys, time, urllib.request, random
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT    = Path(__file__).parent.parent
OUT     = ROOT / "results" / "cross_model"
LLAMA_8 = ROOT / "results" / "llama_sweep" / "sweep_results.json"
CORPUS  = ROOT / "experiments" / "corpus" / "stimuli_llama_subset.json"
OUT.mkdir(parents=True, exist_ok=True)

BASE    = os.environ.get("TRIBE_URL", "http://localhost:8081")
REGIONS = ["visual","auditory","language","prefrontal","motor","parietal"]
random.seed(42)


# ─────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────
def predict(text, seq_len=8, timeout=180):
    body = json.dumps({"text": text, "seq_len": seq_len}).encode()
    req  = urllib.request.Request(
        f"{BASE}/api/predict", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def flatten(resp):
    rs = resp.get("region_stats", {})
    gs = resp.get("global_stats", {})
    row = {"global_mean": gs.get("global_mean"),
           "global_max":  gs.get("global_max"),
           "demo_mode":   resp.get("demo_mode", True),
           "elapsed_ms":  resp.get("elapsed_ms")}
    for reg in REGIONS:
        row[f"{reg}_rel"] = rs.get(reg, {}).get("rel_activation")
    return row


def per_ct_means(records):
    by_ct = defaultdict(list)
    for r in records:
        if r.get("global_mean") is not None and r.get("content_type"):
            by_ct[r["content_type"]].append(r["global_mean"])
    return {ct: float(np.mean(v)) for ct, v in by_ct.items()}


# ─────────────────────────────────────────────────────────────────
# Part A — seq_len robustness sweeps
# ─────────────────────────────────────────────────────────────────
def sweep_seqlen(seq_len: int):
    out_path = OUT / f"seqlen_{seq_len}_results.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        ok = [r for r in existing if "global_mean" in r]
        if len(ok) >= 390:
            print(f"[seq{seq_len}] already complete ({len(ok)} records) — skipping", flush=True)
            return existing

    corpus = json.load(open(CORPUS))
    print(f"[seq{seq_len}] {len(corpus)} stimuli, seq_len={seq_len}", flush=True)
    records, t0 = [], time.time()
    for i, s in enumerate(corpus):
        ts   = time.time()
        resp = predict(s["text"], seq_len=seq_len)
        el   = time.time() - ts
        rec  = {"id": s["id"], "content_type": s.get("content_type"), "seq_len": seq_len}
        if "error" in resp:
            rec["error"] = resp["error"]
            print(f"[seq{seq_len}] ERR {i+1}/{len(corpus)} {el:.1f}s {resp['error'][:50]}", flush=True)
        else:
            rec.update(flatten(resp))
            if (i+1) % 10 == 0 or i < 3:
                rate = (i+1) / (time.time()-t0)
                eta  = (len(corpus)-(i+1)) / rate
                print(f"[seq{seq_len}] {i+1:3}/{len(corpus)} {s.get('content_type'):<14} "
                      f"{el:.1f}s gm={rec['global_mean']:+.5f}  eta={eta:.0f}s", flush=True)
        records.append(rec)
        if (i+1) % 20 == 0 or i == len(corpus)-1:
            json.dump(records, open(out_path, "w"), indent=2)
    print(f"[seq{seq_len}] saved → {out_path}", flush=True)
    return records


# ─────────────────────────────────────────────────────────────────
# Part B — BERT embedding proxy
# ─────────────────────────────────────────────────────────────────
def bert_embeddings():
    """
    Compute text embeddings using TF-IDF + LSA (Latent Semantic Analysis, 300 dims).
    This is a pure-sklearn approach that avoids PyTorch/Metal issues on macOS while
    still providing a meaningful, model-independent text similarity baseline for the
    cross-encoder RDM comparison.
    """
    out_path = OUT / "bert_embeddings.json"
    if out_path.exists():
        print("[lsa] embeddings already computed — loading", flush=True)
        return json.load(open(out_path))

    print("[lsa] computing TF-IDF + LSA (300-dim) embeddings — no PyTorch needed …", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    corpus = json.load(open(CORPUS))
    texts  = [s["text"] for s in corpus]

    # TF-IDF (unigrams + bigrams, sublinear tf, min_df=2)
    t0 = time.time()
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                          min_df=2, max_features=50_000)
    X = vec.fit_transform(texts)
    print(f"[lsa] TF-IDF: {X.shape[0]} docs × {X.shape[1]} features  ({time.time()-t0:.1f}s)", flush=True)

    # SVD to 300 dims (≈ LSA)
    svd = TruncatedSVD(n_components=300, random_state=42)
    E   = normalize(svd.fit_transform(X))        # L2-normalised
    print(f"[lsa] LSA: {E.shape}  explained_var={svd.explained_variance_ratio_.sum():.3f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    results = []
    for i, s in enumerate(corpus):
        results.append({"id": s["id"], "content_type": s.get("content_type"),
                        "embedding": E[i].tolist()})

    json.dump(results, open(out_path, "w"))
    print(f"[lsa] saved → {out_path}  ({time.time()-t0:.1f}s)", flush=True)
    return results


def bert_per_ct_centroids(embed_records):
    """Mean embedding per content type."""
    by_ct = defaultdict(list)
    for r in embed_records:
        if r.get("embedding") and r.get("content_type"):
            by_ct[r["content_type"]].append(r["embedding"])
    return {ct: np.mean(v, axis=0) for ct, v in by_ct.items()}


# ─────────────────────────────────────────────────────────────────
# Part C — Cross-encoder RDM (Representational Dissimilarity Matrix)
# ─────────────────────────────────────────────────────────────────
def compute_rdm(vectors, metric="cosine"):
    """vectors: dict {ct: array}. Returns (cts, matrix)."""
    from scipy.spatial.distance import cdist
    cts  = sorted(vectors.keys())
    mat  = np.vstack([vectors[ct] for ct in cts])
    rdm  = cdist(mat, mat, metric=metric)
    return cts, rdm


def mantel_test(rdm_a, rdm_b, n_perms=10_000, seed=42):
    """Mantel test: correlation between upper-triangles of two RDMs."""
    from scipy.stats import pearsonr, spearmanr
    rng = np.random.default_rng(seed)
    n   = rdm_a.shape[0]
    idx = np.triu_indices(n, k=1)
    va  = rdm_a[idx]
    vb  = rdm_b[idx]
    r_obs, _  = pearsonr(va, vb)
    rho_obs,_ = spearmanr(va, vb)
    # Permutation test on rows/cols of rdm_b
    perm_rs = []
    for _ in range(n_perms):
        perm  = rng.permutation(n)
        vbp   = rdm_b[np.ix_(perm, perm)][idx]
        rp, _ = pearsonr(va, vbp)
        perm_rs.append(rp)
    perm_rs = np.array(perm_rs)
    p_val = float(np.mean(perm_rs >= r_obs))
    return {"r_pearson": float(r_obs), "rho_spearman": float(rho_obs), "p_mantel": p_val}


# ─────────────────────────────────────────────────────────────────
# Main analysis & output
# ─────────────────────────────────────────────────────────────────
def run_rdm_analysis(bert_records, tribe_records_8):
    """Build and compare RDMs for BERT and TRIBE."""
    from scipy.stats import spearmanr, pearsonr

    # BERT RDM (cosine distance between per-CT mean embeddings)
    bert_cents = bert_per_ct_centroids(bert_records)
    cts_b, bert_rdm = compute_rdm(bert_cents, metric="cosine")

    # TRIBE RDM (Euclidean distance between per-CT mean activation vectors)
    # Use full region profile: [global_mean, visual_rel, auditory_rel, language_rel,
    #                           prefrontal_rel, motor_rel, parietal_rel]
    tribe_vecs = {}
    by_ct = defaultdict(list)
    for r in tribe_records_8:
        if r.get("global_mean") is not None and r.get("content_type"):
            vec = [r.get("global_mean", 0)] + [r.get(f"{reg}_rel", 0) or 0 for reg in REGIONS]
            by_ct[r["content_type"]].append(vec)
    for ct, vecs in by_ct.items():
        tribe_vecs[ct] = np.mean(vecs, axis=0)

    cts_t, tribe_rdm = compute_rdm(tribe_vecs, metric="euclidean")

    # Align to same CT order
    assert set(cts_b) == set(cts_t), "CT mismatch between BERT and TRIBE"
    cts = sorted(cts_b)
    b_ord = [cts_b.index(ct) for ct in cts]
    t_ord = [cts_t.index(ct) for ct in cts]
    bert_rdm_a  = bert_rdm[np.ix_(b_ord, b_ord)]
    tribe_rdm_a = tribe_rdm[np.ix_(t_ord, t_ord)]

    mantel = mantel_test(bert_rdm_a, tribe_rdm_a)
    result = {
        "cts": cts,
        "bert_rdm":  bert_rdm_a.tolist(),
        "tribe_rdm": tribe_rdm_a.tolist(),
        "mantel":    mantel,
    }
    json.dump(result, open(OUT / "rdm_analysis.json", "w"), indent=2)
    print(f"[rdm] Mantel r={mantel['r_pearson']:.3f}, rho={mantel['rho_spearman']:.3f}, "
          f"p={mantel['p_mantel']:.4f} ({10_000} perms)", flush=True)
    return result


def summarise_seqlen_stability(records_by_seqlen, cts):
    """Rank correlation between seq_len conditions."""
    from scipy.stats import spearmanr, pearsonr
    means = {}
    for sl, recs in records_by_seqlen.items():
        m = per_ct_means(recs)
        means[sl] = [m[ct] for ct in cts]
    sls = sorted(means.keys())
    corrs = {}
    for i, sl_a in enumerate(sls):
        for sl_b in sls[i+1:]:
            r, p    = pearsonr(means[sl_a], means[sl_b])
            rho, rp = spearmanr(means[sl_a], means[sl_b])
            corrs[f"sl{sl_a}_vs_sl{sl_b}"] = {"pearson_r": float(r), "spearman_rho": float(rho),
                                                "pearson_p": float(p), "spearman_p": float(rp)}
    rankings = {}
    for sl, m in means.items():
        ranked = sorted(cts, key=lambda ct: means[sl][cts.index(ct)])
        rankings[str(sl)] = ranked
    result = {"means_by_seqlen": {str(k): dict(zip(cts, v)) for k, v in means.items()},
              "rankings":        rankings,
              "pairwise_corrs":  corrs}
    json.dump(result, open(OUT / "seqlen_stability.json", "w"), indent=2)
    for k, v in corrs.items():
        print(f"[seqlen] {k}: Pearson r={v['pearson_r']:.3f}, Spearman rho={v['spearman_rho']:.3f}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = set(sys.argv[1:]) if len(sys.argv) > 1 else {"all"}

    # ── Part A: seq_len sweeps
    if "all" in args or "seqsweep" in args:
        recs_4  = sweep_seqlen(4)
        recs_16 = sweep_seqlen(16)
        recs_8  = json.load(open(LLAMA_8))
        cts_all = sorted(set(r["content_type"] for r in recs_8 if r.get("content_type")))
        summarise_seqlen_stability({4: recs_4, 8: recs_8, 16: recs_16}, cts_all)

    # ── Part B+C: BERT embeddings + RDM
    if "all" in args or "bert" in args or "rdm" in args:
        recs_8   = json.load(open(LLAMA_8))
        bert_recs = bert_embeddings()
        run_rdm_analysis(bert_recs, recs_8)

    print("DONE", flush=True)
