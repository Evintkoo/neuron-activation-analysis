#!/usr/bin/env python3
"""
Two extensions:
  1. Temporal dynamics: re-sweep ~120 stimuli (10/category) capturing temporal_acts
  2. Multilingual replication: fetch + sweep Spanish/French/German/Mandarin Wikipedia content
"""
import json, time, urllib.request, urllib.parse, random, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
EXT  = ROOT / "results" / "extended"
EXT.mkdir(exist_ok=True)
random.seed(42)

BASE = "http://localhost:8081"

def predict(text):
    body = json.dumps({"text": text, "seq_len": 16}).encode()
    req = urllib.request.Request(f"{BASE}/api/predict", data=body,
                                  headers={"Content-Type":"application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def get_text(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent":"research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  WARN {url[:60]}: {e}", file=__import__('sys').stderr)
        return ""

def get_json(url):
    raw = get_text(url)
    try: return json.loads(raw) if raw else {}
    except: return {}

def clean(t): return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', str(t))).strip()[:800]

# ═══════════════════════════════════════════════════════════════════════════
# 1. TEMPORAL DYNAMICS — re-sweep stratified subsample
# ═══════════════════════════════════════════════════════════════════════════
print("=== TEMPORAL DYNAMICS SWEEP ===")
master = json.load(open(ROOT / "experiments/corpus/stimuli_master.json"))
by_ct = {}
for s in master: by_ct.setdefault(s["content_type"], []).append(s)
print(f"  Master corpus: {sum(len(v) for v in by_ct.values())} stimuli, {len(by_ct)} types")

# Take 12 per content type
SAMPLE_SIZE = 12
temporal_records = []
for ct, items in by_ct.items():
    random.shuffle(items)
    for s in items[:SAMPLE_SIZE]:
        resp = predict(s["text"])
        if "error" in resp:
            print(f"  ERROR {s['id']}: {resp['error']}")
            continue
        ta = resp.get("temporal_acts", [])
        if not ta:
            continue
        temporal_records.append({
            "id": s["id"],
            "content_type": ct,
            "source_type": s["source_type"],
            "temporal_acts": ta,  # [T][6]
            "global_mean": resp.get("global_stats",{}).get("global_mean", 0),
        })
    print(f"  {ct:<15} captured {len(temporal_records)} cumulative")

print(f"  Total: {len(temporal_records)} temporal records")
with open(EXT / "temporal_records.json", "w") as f:
    json.dump(temporal_records, f, indent=2)
print(f"  Saved → results/extended/temporal_records.json")

# Analyze temporal trajectories per content type
print("\n  Analyzing temporal patterns per content type...")
import numpy as np
REGIONS = ["visual","auditory","language","prefrontal","motor","parietal"]
ct_traj = {}
for ct in by_ct:
    recs = [r for r in temporal_records if r["content_type"] == ct]
    if not recs: continue
    arrays = []
    for r in recs:
        arr = np.array(r["temporal_acts"])
        if arr.ndim == 2 and arr.shape[1] == 6:
            arrays.append(arr)
    if not arrays: continue
    T_max = max(a.shape[0] for a in arrays)
    padded = np.zeros((len(arrays), T_max, 6))
    for i, a in enumerate(arrays):
        padded[i, :a.shape[0]] = a
    mean_traj = padded.mean(axis=0)  # (T, 6)
    ct_traj[ct] = mean_traj.tolist()
    print(f"    {ct:<15} N={len(recs)} T={T_max}")

with open(EXT / "temporal_dynamics_by_ct.json", "w") as f:
    json.dump({
        "regions": REGIONS,
        "T_steps": T_max,
        "trajectories": ct_traj,
    }, f, indent=2)
print(f"  Saved → results/extended/temporal_dynamics_by_ct.json")

# ═══════════════════════════════════════════════════════════════════════════
# 2. MULTILINGUAL REPLICATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== MULTILINGUAL REPLICATION ===")

# Fetch Wikipedia random summaries in 5 languages
langs = {
    "es": "Spanish",   "fr": "French",   "de": "German",
    "it": "Italian",   "pt": "Portuguese",
    "ja": "Japanese",  "zh": "Mandarin", "ar": "Arabic",
    "ru": "Russian",   "ko": "Korean",
}

multilingual_stimuli = []
N_PER_LANG = 30
for lang_code, lang_name in langs.items():
    print(f"  Fetching {lang_name} ({lang_code})...")
    fetched = 0
    for _ in range(N_PER_LANG * 2):  # try double to allow filtering
        if fetched >= N_PER_LANG: break
        d = get_json(f"https://{lang_code}.wikipedia.org/api/rest_v1/page/random/summary")
        extract = clean(d.get("extract", ""))
        if len(extract) > 80:
            multilingual_stimuli.append({
                "id": f"ml_{lang_code}_{fetched:04d}",
                "language": lang_code,
                "language_name": lang_name,
                "source": f"wikipedia_{lang_code}",
                "text": extract,
            })
            fetched += 1
        time.sleep(0.25)
    print(f"    {lang_name}: {fetched} stimuli")

print(f"  Total multilingual: {len(multilingual_stimuli)}")

# Save corpus
with open(EXT / "multilingual_corpus.json", "w") as f:
    json.dump(multilingual_stimuli, f, indent=2, ensure_ascii=False)

# Sweep through TRIBE
print("\n  Running multilingual sweep...")
multilingual_results = []
for i, s in enumerate(multilingual_stimuli):
    resp = predict(s["text"])
    if "error" in resp:
        print(f"    ERROR {s['id']}: {resp['error']}")
        continue
    rs = resp.get("region_stats", {})
    gs = resp.get("global_stats", {})
    multilingual_results.append({
        "id": s["id"],
        "language": s["language"],
        "language_name": s["language_name"],
        "source": s["source"],
        "global_mean": gs.get("global_mean"),
        "global_max": gs.get("global_max"),
        "visual_rel": rs.get("visual",{}).get("rel_activation"),
        "auditory_rel": rs.get("auditory",{}).get("rel_activation"),
        "language_rel": rs.get("language",{}).get("rel_activation"),
        "prefrontal_rel": rs.get("prefrontal",{}).get("rel_activation"),
        "motor_rel": rs.get("motor",{}).get("rel_activation"),
        "parietal_rel": rs.get("parietal",{}).get("rel_activation"),
        "demo_mode": resp.get("demo_mode", True),
    })
    if (i+1) % 50 == 0:
        print(f"    [{i+1}/{len(multilingual_stimuli)}]")

print(f"  Completed {len(multilingual_results)} multilingual predictions")
with open(EXT / "multilingual_results.json", "w") as f:
    json.dump(multilingual_results, f, indent=2)

# Per-language summary
import numpy as np
print("\n  Per-language summary:")
print(f"    {'Language':<12} {'N':>3} {'Mean':>10} {'SD':>10} {'Lang_rel':>9}")
lang_summary = {}
for code, name in langs.items():
    sub = [r for r in multilingual_results if r["language"] == code and r.get("global_mean") is not None]
    if not sub: continue
    gms = np.array([r["global_mean"] for r in sub])
    lrs = np.array([r["language_rel"] for r in sub if r["language_rel"] is not None])
    lang_summary[code] = {
        "name": name,
        "n": len(sub),
        "global_mean": float(gms.mean()),
        "global_sd": float(gms.std()),
        "language_rel": float(lrs.mean()) if len(lrs) else None,
    }
    print(f"    {name:<12} {len(sub):>3} {gms.mean():>+.5f} {gms.std():>.5f} {lrs.mean() if len(lrs) else 0:>+.3f}")

with open(EXT / "multilingual_summary.json", "w") as f:
    json.dump(lang_summary, f, indent=2)
print(f"  Saved → results/extended/multilingual_summary.json")

# ANOVA across languages
from scipy import stats as sp_stats
groups = [[r["global_mean"] for r in multilingual_results
           if r["language"] == c and r["global_mean"] is not None]
          for c in langs]
groups = [g for g in groups if len(g) > 1]
if len(groups) >= 2:
    F, p = sp_stats.f_oneway(*groups)
    print(f"\n  Cross-language ANOVA: F={F:.3f}, p={p:.4f}")
    with open(EXT / "multilingual_anova.json", "w") as f:
        json.dump({"F": float(F), "p": float(p), "k": len(groups),
                   "n_total": sum(len(g) for g in groups)}, f, indent=2)

print("\n=== EXTENSIONS COMPLETE ===")
