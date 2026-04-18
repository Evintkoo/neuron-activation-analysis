#!/usr/bin/env python3
"""Run the three pending extension sweeps once the tribe-server is up:

  (a) Multilingual sweep — uses pre-fetched results/extended/multilingual_corpus.json
  (b) Temporal-dynamics sweep — 12 stimuli per content type, captures temporal_acts
  (c) LLaMA semantic replication — full pass over experiments/corpus/stimuli_llama_subset.json

All requests go to /api/predict on http://localhost:8081. Results land in
results/extended/ and results/llama_sweep/ respectively. Each sweep writes
incrementally so a crash does not lose prior progress.
"""
import json, os, sys, time, urllib.request, random
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent
EXT  = ROOT / "results" / "extended"
LLAMA_OUT = ROOT / "results" / "llama_sweep"
EXT.mkdir(parents=True, exist_ok=True)
LLAMA_OUT.mkdir(parents=True, exist_ok=True)

BASE = os.environ.get("TRIBE_URL", "http://localhost:8081")
random.seed(42)

REGIONS = ["visual","auditory","language","prefrontal","motor","parietal"]


SEQ_LEN = int(os.environ.get("TRIBE_SEQ_LEN", "8"))
TIMEOUT = int(os.environ.get("TRIBE_TIMEOUT", "180"))


def predict(text, seq_len=SEQ_LEN, timeout=TIMEOUT):
    body = json.dumps({"text": text, "seq_len": seq_len}).encode()
    req = urllib.request.Request(
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
    return {
        "global_mean": gs.get("global_mean"),
        "global_max":  gs.get("global_max"),
        "visual_rel":     rs.get("visual",{}).get("rel_activation"),
        "auditory_rel":   rs.get("auditory",{}).get("rel_activation"),
        "language_rel":   rs.get("language",{}).get("rel_activation"),
        "prefrontal_rel": rs.get("prefrontal",{}).get("rel_activation"),
        "motor_rel":      rs.get("motor",{}).get("rel_activation"),
        "parietal_rel":   rs.get("parietal",{}).get("rel_activation"),
        "demo_mode":      resp.get("demo_mode", True),
        "elapsed_ms":     resp.get("elapsed_ms"),
    }


# ───────────────────────────────────────────────────────────────────────────
def sweep_multilingual():
    corpus = json.load(open(EXT / "multilingual_corpus.json"))
    cap_per_lang = int(os.environ.get("MULTI_PER_LANG", "0"))
    if cap_per_lang > 0:
        seen = defaultdict(int)
        kept = []
        for s in corpus:
            if seen[s["language"]] < cap_per_lang:
                kept.append(s); seen[s["language"]] += 1
        corpus = kept
    print(f"[multi] {len(corpus)} stimuli across {len(set(s['language'] for s in corpus))} languages seq_len={SEQ_LEN}", flush=True)
    results, t0 = [], time.time()
    out_path = EXT / "multilingual_results.json"
    for i, s in enumerate(corpus):
        ts = time.time()
        resp = predict(s["text"])
        elapsed = time.time() - ts
        rec = {"id": s["id"], "language": s["language"],
               "language_name": s["language_name"], "source": s["source"]}
        if "error" in resp:
            rec["error"] = resp["error"]
            print(f"[multi] ERR {i+1}/{len(corpus)} {s['language']} {elapsed:.1f}s {resp['error'][:60]}", flush=True)
        else:
            rec.update(flatten(resp))
            print(f"[multi] OK  {i+1}/{len(corpus)} {s['language']} {elapsed:.1f}s gm={rec['global_mean']:+.5f}", flush=True)
        results.append(rec)
        if (i+1) % 5 == 0 or i == len(corpus)-1:
            json.dump(results, open(out_path, "w"), indent=2)
    dt = time.time() - t0
    ok = sum(1 for r in results if "global_mean" in r)
    print(f"[multi] saved → {out_path} ok={ok}/{len(corpus)} total={dt:.0f}s", flush=True)
    summarise_multilingual(results)


def summarise_multilingual(results):
    by_lang = defaultdict(list)
    for r in results:
        if r.get("global_mean") is not None:
            by_lang[r["language"]].append(r)
    summary = {}
    for code, recs in by_lang.items():
        gms = np.array([r["global_mean"] for r in recs])
        lrs = np.array([r["language_rel"] for r in recs if r.get("language_rel") is not None])
        summary[code] = {
            "name": recs[0]["language_name"],
            "n": len(recs),
            "global_mean": float(gms.mean()),
            "global_sd":   float(gms.std()),
            "language_rel": float(lrs.mean()) if len(lrs) else None,
        }
    json.dump(summary, open(EXT / "multilingual_summary.json", "w"), indent=2)

    try:
        from scipy import stats as sp
        groups = [[r["global_mean"] for r in recs] for recs in by_lang.values() if len(recs) > 1]
        if len(groups) >= 2:
            F, p = sp.f_oneway(*groups)
            anova = {"F": float(F), "p": float(p), "k": len(groups),
                     "n_total": sum(len(g) for g in groups)}
            json.dump(anova, open(EXT / "multilingual_anova.json", "w"), indent=2)
            print(f"[multi] cross-lang ANOVA F={F:.3f}, p={p:.4f}")
    except Exception as e:
        print(f"[multi] ANOVA skipped: {e}")


# ───────────────────────────────────────────────────────────────────────────
def sweep_temporal():
    master = json.load(open(ROOT / "experiments/corpus/stimuli_master.json"))
    by_ct = defaultdict(list)
    for s in master:
        by_ct[s["content_type"]].append(s)
    print(f"[temp] master corpus {len(master)} stimuli, {len(by_ct)} content types")

    SAMPLE_SIZE = int(os.environ.get("TEMP_SAMPLE", "12"))
    records, t0 = [], time.time()
    out_path = EXT / "temporal_records.json"
    for ct, items in by_ct.items():
        random.shuffle(items)
        for s in items[:SAMPLE_SIZE]:
            ts = time.time()
            resp = predict(s["text"])
            el = time.time() - ts
            if "error" in resp:
                print(f"[temp] ERR {ct} {el:.1f}s {resp['error'][:60]}", flush=True)
                continue
            ta = resp.get("temporal_acts", [])
            if not ta:
                print(f"[temp] no temporal_acts {ct} {el:.1f}s", flush=True)
                continue
            records.append({
                "id": s["id"],
                "content_type": ct,
                "source_type": s.get("source_type"),
                "temporal_acts": ta,
                "global_mean": resp.get("global_stats", {}).get("global_mean", 0),
            })
        json.dump(records, open(out_path, "w"), indent=2)
        print(f"[temp] {ct:<14} cumulative={len(records)} elapsed={time.time()-t0:.0f}s", flush=True)

    by_ct_arr = defaultdict(list)
    T_max = 0
    for r in records:
        arr = np.array(r["temporal_acts"])
        if arr.ndim == 2 and arr.shape[1] == 6:
            by_ct_arr[r["content_type"]].append(arr)
            T_max = max(T_max, arr.shape[0])

    trajectories = {}
    for ct, arrs in by_ct_arr.items():
        padded = np.zeros((len(arrs), T_max, 6))
        for i, a in enumerate(arrs):
            padded[i, :a.shape[0]] = a
        trajectories[ct] = padded.mean(axis=0).tolist()

    json.dump({"regions": REGIONS, "T_steps": T_max, "trajectories": trajectories},
              open(EXT / "temporal_dynamics_by_ct.json", "w"), indent=2)
    print(f"[temp] saved {len(by_ct_arr)} CTs → temporal_dynamics_by_ct.json")


# ───────────────────────────────────────────────────────────────────────────
def sweep_llama_semantic():
    corpus = json.load(open(ROOT / "experiments/corpus/stimuli_llama_subset.json"))
    per_ct = int(os.environ.get("LLAMA_PER_CT", "0"))
    if per_ct > 0:
        seen = defaultdict(int)
        kept = []
        random.shuffle(corpus)
        for s in corpus:
            ct = s.get("content_type")
            if seen[ct] < per_ct:
                kept.append(s); seen[ct] += 1
        corpus = kept
    cap = int(os.environ.get("LLAMA_CAP", str(len(corpus))))
    corpus = corpus[:cap]
    print(f"[llama] {len(corpus)} stimuli, semantic LLaMA encoding seq_len={SEQ_LEN}", flush=True)
    out_path = LLAMA_OUT / "sweep_results.json"
    records, t0 = [], time.time()
    by_ct_count = defaultdict(int)
    for i, s in enumerate(corpus):
        ts = time.time()
        resp = predict(s["text"])
        el = time.time() - ts
        rec = {
            "id": s["id"],
            "content_type": s.get("content_type"),
            "source_type": s.get("source_type"),
        }
        if "error" in resp:
            rec["error"] = resp["error"]
            print(f"[llama] ERR {i+1}/{len(corpus)} {s.get('content_type')} {el:.1f}s {resp['error'][:60]}", flush=True)
        else:
            rec.update(flatten(resp))
            by_ct_count[s.get("content_type")] += 1
            if (i+1) % 5 == 0 or i < 5:
                print(f"[llama] OK  {i+1}/{len(corpus)} {s.get('content_type'):<14} {el:.1f}s gm={rec['global_mean']:+.5f}", flush=True)
        records.append(rec)
        if (i+1) % 10 == 0 or i == len(corpus)-1:
            json.dump(records, open(out_path, "w"), indent=2)
            dt = time.time() - t0
            ok = sum(1 for r in records if "global_mean" in r)
            rate = (i+1) / dt if dt > 0 else 0
            eta = (len(corpus) - (i+1)) / rate if rate > 0 else 0
            print(f"[llama] CHK {i+1}/{len(corpus)} ok={ok} elapsed={dt:.0f}s eta={eta:.0f}s", flush=True)
    print(f"[llama] saved → {out_path}", flush=True)
    summarise_llama(records)


def summarise_llama(records):
    by_ct = defaultdict(list)
    for r in records:
        if r.get("global_mean") is not None and r.get("content_type"):
            by_ct[r["content_type"]].append(r)
    summary = {}
    for ct, recs in by_ct.items():
        gms = np.array([r["global_mean"] for r in recs])
        summary[ct] = {
            "n": len(recs),
            "global_mean": float(gms.mean()),
            "global_sd":   float(gms.std()),
            "language_rel": float(np.mean([r["language_rel"] for r in recs if r.get("language_rel") is not None])),
        }
    json.dump(summary, open(LLAMA_OUT / "summary_by_ct.json", "w"), indent=2)
    try:
        from scipy import stats as sp
        groups = [[r["global_mean"] for r in recs] for recs in by_ct.values() if len(recs) > 1]
        if len(groups) >= 2:
            F, p = sp.f_oneway(*groups)
            anova = {"F": float(F), "p": float(p), "k": len(groups),
                     "n_total": sum(len(g) for g in groups)}
            json.dump(anova, open(LLAMA_OUT / "anova.json", "w"), indent=2)
            print(f"[llama] CT ANOVA F={F:.3f}, p={p:.4f}")
    except Exception as e:
        print(f"[llama] ANOVA skipped: {e}")


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    sel = set(args)
    if "all" in sel or sel & {"multi", "multilingual"}:
        sweep_multilingual()
    if "all" in sel or sel & {"temp", "temporal"}:
        sweep_temporal()
    if "all" in sel or sel & {"llama", "semantic"}:
        sweep_llama_semantic()
    print("DONE", flush=True)
