#!/usr/bin/env python3
"""
Fetch proper research datasets for the fMRI activation sweep corpus.
Uses HuggingFace datasets-server API, direct parquet downloads, and
public dataset URLs. Requires: pandas, pyarrow.
"""
import json, time, re, sys, random, urllib.request, urllib.parse, io
from pathlib import Path
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "experiments" / "corpus" / "stimuli_proper.json"
DELAY = 0.3
random.seed(42)

results = []
counts  = Counter()

def add(id_prefix, ct, source, lang_struct, text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'<[^>]+>', '', text)
    if len(text) < 40: return
    text = text[:1000]
    n = counts[ct] + 1
    counts[ct] = n
    results.append({
        "id": f"{id_prefix}_{n:05d}",
        "content_type": ct,
        "source_type": source,
        "language_structure": lang_struct,
        "text": text,
    })

def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "brain-encoding-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        print(f"  WARN {url[:70]}: {e}", file=sys.stderr)
        return b""

def get_json(url):
    raw = get(url)
    try: return json.loads(raw)
    except: return {}

def hf_rows(dataset, config="default", split="train", offset=0, limit=200):
    url = (f"https://datasets-server.huggingface.co/rows"
           f"?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
           f"&split={urllib.parse.quote(split)}&offset={offset}&limit={limit}")
    d = get_json(url)
    return [r["row"] for r in d.get("rows", [])]

def hf_parquet(dataset, config="default", split="train"):
    """Download parquet file from HuggingFace and return as DataFrame."""
    info_url = (f"https://datasets-server.huggingface.co/parquet"
                f"?dataset={urllib.parse.quote(dataset)}")
    info = get_json(info_url)
    files = [f for f in info.get("parquet_files", [])
             if f.get("config") == config and f.get("split") == split]
    if not files:
        files = [f for f in info.get("parquet_files", [])
                 if f.get("split") == split]
    if not files:
        files = info.get("parquet_files", [])[:1]
    if not files:
        print(f"  No parquet for {dataset}", file=sys.stderr)
        return pd.DataFrame()
    url = files[0]["url"]
    raw = get(url)
    if not raw: return pd.DataFrame()
    try:
        buf = io.BytesIO(raw)
        return pq.read_table(buf).to_pandas()
    except Exception as e:
        print(f"  Parquet read error {dataset}: {e}", file=sys.stderr)
        return pd.DataFrame()

def clean_text(t):
    if not isinstance(t, str): return ""
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', t)).strip()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ThreatSafety + Novelty — AG News (fancyzhx/ag_news)
#    labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
# ═══════════════════════════════════════════════════════════════════════════════

THREAT_KWORDS = {"kill","dead","attack","crash","fire","flood","earthquake","storm",
                 "killed","deaths","injured","explosion","war","crisis","emergency",
                 "recall","warning","threat","danger","collapse","shooting","bombing",
                 "outbreak","murder","arrested","accused","disaster","hurricane","typhoon",
                 "hostage","kidnap","terror","nuclear","chemical","toxic","evacuate","trapped"}
NOVEL_KWORDS  = {"discover","breakthrough","first","new","novel","launch","announce",
                 "scientists","researchers","study","found","reveals","develops","creates",
                 "artificial","quantum","gene","space","universe","climate","record",
                 "unprecedented","never","invented","cure","vaccine","treatment"}

print("=== AG News → ThreatSafety + Novelty ===")
for offset in range(0, 3000, 200):
    rows = hf_rows("fancyzhx/ag_news", config="default", split="test", offset=offset, limit=200)
    if not rows: break
    for r in rows:
        text = clean_text(r.get("text", ""))
        if len(text) < 40: continue
        lc = text.lower()
        words = set(re.findall(r'\b\w+\b', lc))
        label = r.get("label", -1)
        if label == 0 and words & THREAT_KWORDS:  # World + threat keywords
            add("ag_b3", "ThreatSafety", "ag_news_world", "short_declarative_present_tense", text)
        elif label == 3 and words & NOVEL_KWORDS:  # Sci/Tech + novelty keywords
            add("ag_b4", "Novelty", "ag_news_scitech", "surprise_framing", text)
    time.sleep(DELAY)
    if counts["ThreatSafety"] >= 200 and counts["Novelty"] >= 200: break

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Narrative — HellaSwag (Rowan/hellaswag) contexts
# ═══════════════════════════════════════════════════════════════════════════════

print("=== HellaSwag → Narrative ===")
seen_ctx = set()
for offset in range(0, 4000, 200):
    rows = hf_rows("Rowan/hellaswag", config="default", split="train", offset=offset, limit=200)
    if not rows: break
    for r in rows:
        ctx = clean_text(r.get("ctx", ""))
        ctx_a = clean_text(r.get("ctx_a", ""))
        activity = clean_text(r.get("activity_label", ""))
        text = ctx if len(ctx) > 80 else ctx_a
        if len(text) < 60: continue
        key = text[:60].lower()
        if key in seen_ctx: continue
        seen_ctx.add(key)
        if activity:
            text = f"[{activity}] {text}"
        add("hs_s1", "Narrative", "hellaswag_activity", "chronological_causal", text)
    time.sleep(DELAY)
    if counts["Narrative"] >= 200: break

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Narrative (continued) — TinyStories (roneneldan/TinyStories)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== TinyStories → Narrative ===")
rows = hf_rows("roneneldan/TinyStories", config="default", split="train", offset=0, limit=500)
random.shuffle(rows)
for r in rows:
    text = clean_text(r.get("text", ""))
    # Take first paragraph only (keep stories concise)
    paras = [p.strip() for p in text.split('\n') if len(p.strip()) > 60]
    if paras:
        add("ts_s1", "Narrative", "tinystories", "chronological_causal", paras[0][:700])
    if counts["Narrative"] >= 350: break
time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Abstract — MultiNLI premises (nyu-mll/multi_nli)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== MultiNLI → Abstract ===")
ABST_GENRES = {"government", "slate", "telephone", "travel", "fiction"}
for offset in range(0, 2000, 200):
    rows = hf_rows("nyu-mll/multi_nli", config="default", split="validation_matched",
                   offset=offset, limit=200)
    if not rows: break
    for r in rows:
        premise = clean_text(r.get("premise", ""))
        genre = r.get("genre", "")
        if len(premise) < 60: continue
        # Prefer government/academic genres for abstract content
        ls = "conditional_argument" if "government" in genre else "dense_nominal_nested"
        add("mn_s2", "Abstract", f"multinli_{genre}", ls, premise)
    time.sleep(DELAY)
    if counts["Abstract"] >= 200: break

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Abstract (continued) — SciQ support passages (allenai/sciq)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== SciQ → Abstract ===")
for offset in range(0, 2000, 200):
    rows = hf_rows("allenai/sciq", config="default", split="train",
                   offset=offset, limit=200)
    if not rows: break
    for r in rows:
        support = clean_text(r.get("support", ""))
        question = clean_text(r.get("question", ""))
        answer = clean_text(r.get("correct_answer", ""))
        if len(support) > 60:
            add("sc_s2", "Abstract", "sciq_support", "formal_definition", support)
        elif len(question) > 40 and len(answer) > 10:
            add("sc_s4", "Factual", "sciq_qa", "declarative_third_person",
                f"{question} {answer}.")
    time.sleep(DELAY)
    if counts["Abstract"] >= 350: break

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Factual — Natural Questions (simplified subset)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== Natural Questions → Factual ===")
rows = hf_rows("google-research-datasets/natural_questions",
               config="default", split="validation", offset=0, limit=200)
for r in rows:
    q = clean_text(r.get("question", {}).get("text", "") if isinstance(r.get("question"), dict)
                   else r.get("question_text", ""))
    # Extract short answers
    annotations = r.get("annotations", [])
    short_ans = ""
    if annotations:
        ann = annotations[0] if isinstance(annotations, list) and annotations else {}
        sas = ann.get("short_answers", [])
        if sas and isinstance(sas, list) and sas[0]:
            sa = sas[0]
            if isinstance(sa, dict) and sa.get("text"):
                short_ans = sa["text"][0] if isinstance(sa["text"], list) else str(sa["text"])
    if q and short_ans:
        add("nq_s4", "Factual", "natural_questions", "declarative_third_person",
            f"{q} {short_ans}.")
time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Emotional — GoEmotions via parquet (google-research-datasets/go_emotions)
#    27 emotion labels; we want high-arousal: joy, anger, fear, disgust, surprise, sadness
# ═══════════════════════════════════════════════════════════════════════════════

print("=== GoEmotions → Emotional (parquet) ===")
EMOTIONAL_LABELS = {2, 10, 14, 16, 25, 26, 27}  # anger, fear, joy, love, sadness, surprise, neutral
df_ge = hf_parquet("google-research-datasets/go_emotions", config="simplified", split="train")
if not df_ge.empty and "text" in df_ge.columns:
    df_ge = df_ge[df_ge["text"].str.len() > 40].sample(min(500, len(df_ge)), random_state=42)
    for _, row in df_ge.iterrows():
        text = clean_text(str(row["text"]))
        labels = row.get("labels", [])
        if isinstance(labels, list) and labels:
            add("ge_s3", "Emotional", "go_emotions_reddit", "first_person_affective", text)
        elif len(text) > 40:
            add("ge_s3", "Emotional", "go_emotions_reddit", "first_person_affective", text)
        if counts["Emotional"] >= 200: break
time.sleep(DELAY)

# Fallback: dair-ai/emotion
if counts["Emotional"] < 100:
    print("  Trying dair-ai/emotion parquet...")
    df_em = hf_parquet("dair-ai/emotion", config="split", split="train")
    if not df_em.empty and "text" in df_em.columns:
        for _, row in df_em.sample(min(300, len(df_em)), random_state=42).iterrows():
            add("da_s3", "Emotional", "dair_ai_emotion", "first_person_affective",
                clean_text(str(row["text"])))
            if counts["Emotional"] >= 200: break

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Reward — Yelp reviews (5-star) via parquet
# ═══════════════════════════════════════════════════════════════════════════════

print("=== Yelp 5-star → Reward (parquet) ===")
df_yelp = hf_parquet("Yelp/yelp_review_full", config="yelp_review_full", split="test")
if df_yelp.empty:
    df_yelp = hf_parquet("Yelp/yelp_review_full", config="default", split="test")
if not df_yelp.empty and "text" in df_yelp.columns:
    label_col = "label" if "label" in df_yelp.columns else "stars"
    five_star = df_yelp[df_yelp[label_col] == 4] if label_col == "label" else df_yelp[df_yelp[label_col] == 5]
    if five_star.empty:
        five_star = df_yelp.nlargest(500, label_col)
    sample = five_star.sample(min(300, len(five_star)), random_state=42)
    for _, row in sample.iterrows():
        text = clean_text(str(row["text"]))
        if len(text) > 60:
            add("yl_b2", "Reward", "yelp_5star_review", "positive_superlative", text[:600])
        if counts["Reward"] >= 200: break
time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Social — DailyDialog via direct download
# ═══════════════════════════════════════════════════════════════════════════════

print("=== DailyDialog → Social ===")
raw = get("https://huggingface.co/datasets/daily_dialog/resolve/main/data/train.zip")
if raw:
    import zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                if "dialogues_train.txt" in name or "train/dialogues" in name:
                    content = z.read(name).decode("utf-8", errors="replace")
                    dialogues = [line.strip() for line in content.split('\n') if line.strip()]
                    random.shuffle(dialogues)
                    for dial in dialogues:
                        turns = [t.strip() for t in dial.split("__eou__") if t.strip()]
                        if len(turns) >= 2:
                            text = " | ".join(turns[:4])
                            add("dd_b1", "Social", "daily_dialog", "multi_agent_dialogue", text)
                        if counts["Social"] >= 200: break
                    break
    except Exception as e:
        print(f"  DailyDialog zip error: {e}", file=sys.stderr)
time.sleep(DELAY)

# Fallback Social: MultiNLI fiction genre (interpersonal)
if counts["Social"] < 80:
    print("  Social fallback: MultiNLI fiction genre...")
    for offset in range(0, 3000, 200):
        rows = hf_rows("nyu-mll/multi_nli", config="default", split="validation_matched",
                       offset=offset, limit=200)
        if not rows: break
        for r in rows:
            if r.get("genre") in ("fiction", "telephone"):
                text = clean_text(r.get("premise", ""))
                if len(text) > 60:
                    add("mn_b1", "Social", "multinli_fiction", "multi_agent_dialogue", text)
        if counts["Social"] >= 200: break
        time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. ImageVisual — COCO Captions (Karpathy test split JSON)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== COCO Captions → ImageVisual ===")
coco_urls = [
    "https://huggingface.co/datasets/HuggingFaceM4/COCO/resolve/main/coco_karpathy_test.json",
    "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k_test.json",
]
for url in coco_urls:
    raw = get(url)
    if not raw: continue
    try:
        data = json.loads(raw)
        items = data if isinstance(data, list) else data.get("annotations", data.get("images", []))
        random.shuffle(items)
        for item in items:
            caption = item.get("caption", "") or item.get("raw", "")
            if isinstance(caption, list): caption = caption[0]
            caption = clean_text(str(caption))
            if len(caption) > 20:
                add("co_m2", "ImageVisual", "coco_karpathy", "sensory_descriptive", caption)
            if counts["ImageVisual"] >= 200: break
        if counts["ImageVisual"] >= 100: break
    except:
        pass
    time.sleep(DELAY)

# Fallback: Flickr30k via HuggingFace rows
if counts["ImageVisual"] < 80:
    print("  ImageVisual fallback: flickr30k...")
    df_fl = hf_parquet("nlphuji/flickr30k", config="default", split="test")
    if not df_fl.empty:
        for col in ["raw", "caption", "sentences_raw"]:
            if col in df_fl.columns:
                for _, row in df_fl.sample(min(300, len(df_fl)), random_state=42).iterrows():
                    cap = row[col]
                    if isinstance(cap, list): cap = cap[0]
                    cap = clean_text(str(cap))
                    if len(cap) > 20:
                        add("fl_m2", "ImageVisual", "flickr30k", "sensory_descriptive", cap)
                    if counts["ImageVisual"] >= 200: break
                break

# ═══════════════════════════════════════════════════════════════════════════════
# 11. AudioText — AudioCaps (GitHub CSV)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== AudioCaps → AudioText ===")
for split_name in ["test", "val", "train"]:
    raw = get(f"https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/{split_name}.csv")
    if not raw: continue
    lines = raw.decode("utf-8", errors="replace").strip().split('\n')
    random.shuffle(lines[1:])
    for line in lines[1:]:
        parts = line.split(',', 2)
        if len(parts) >= 3:
            caption = clean_text(parts[-1].strip('"'))
            if len(caption) > 20:
                add("ac_m3", "AudioText", "audiocaps", "acoustic_descriptive", caption)
        if counts["AudioText"] >= 200: break
    if counts["AudioText"] >= 100: break
    time.sleep(DELAY)

# Fallback: Clotho dataset descriptions
if counts["AudioText"] < 80:
    print("  AudioText fallback: Clotho dataset...")
    df_cl = hf_parquet("benjaminbeilharz/clotho-dataset", config="default", split="test")
    if df_cl.empty:
        df_cl = hf_parquet("clotho-dataset", config="default", split="evaluation")
    if not df_cl.empty:
        for col in ["caption_1", "caption", "captions"]:
            if col in df_cl.columns:
                for _, row in df_cl.sample(min(300, len(df_cl)), random_state=42).iterrows():
                    cap = row[col]
                    if isinstance(cap, list): cap = cap[0]
                    cap = clean_text(str(cap))
                    if len(cap) > 20:
                        add("cl_m3", "AudioText", "clotho", "acoustic_descriptive", cap)
                    if counts["AudioText"] >= 200: break
                break

# ═══════════════════════════════════════════════════════════════════════════════
# 12. Multimodal — ActivityNet Captions (GitHub JSON)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== ActivityNet Captions → Multimodal ===")
raw = get("https://raw.githubusercontent.com/ranjaykrishna/densevid_eval/master/data/val_1.json",
          timeout=30)
if raw:
    try:
        data = json.loads(raw)
        items = list(data.items()) if isinstance(data, dict) else []
        random.shuffle(items)
        for vid_id, info in items[:500]:
            sentences = info.get("sentences", [])
            if len(sentences) >= 2:
                text = " ".join(sentences[:3])
                add("an_m4", "Multimodal", "activitynet_captions", "simultaneous_av", text)
            if counts["Multimodal"] >= 200: break
    except: pass
time.sleep(DELAY)

# Fallback: MSR-VTT descriptions
if counts["Multimodal"] < 80:
    print("  Multimodal fallback: MSR-VTT...")
    df_vtt = hf_parquet("AlexZigma/msr-vtt", config="default", split="test")
    if not df_vtt.empty:
        for col in ["caption", "sentence", "captions"]:
            if col in df_vtt.columns:
                for _, row in df_vtt.sample(min(300, len(df_vtt)), random_state=42).iterrows():
                    cap = row[col]
                    if isinstance(cap, list): cap = cap[0]
                    cap = clean_text(str(cap))
                    if len(cap) > 20:
                        add("vt_m4", "Multimodal", "msr_vtt", "simultaneous_av", cap)
                    if counts["Multimodal"] >= 200: break
                break

# ═══════════════════════════════════════════════════════════════════════════════
# 13. Spatial — Region captions from Visual Genome (via HuggingFace)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== Visual Genome regions → Spatial ===")
df_vg = hf_parquet("visual_genome", config="region_descriptions_v1_2_0", split="train")
if df_vg.empty:
    df_vg = hf_parquet("merve/vg-captions", config="default", split="train")
if not df_vg.empty:
    for col in ["phrase", "region_description", "description"]:
        if col in df_vg.columns:
            sample = df_vg.sample(min(500, len(df_vg)), random_state=42)
            for _, row in sample.iterrows():
                text = clean_text(str(row[col]))
                if len(text) > 20 and any(w in text.lower() for w in
                    ["left","right","top","bottom","above","below","behind","front",
                     "next to","near","inside","outside","on the","in the","at the"]):
                    add("vg_s5", "Spatial", "visual_genome_regions", "positional_relational", text)
                if counts["Spatial"] >= 200: break
            break
time.sleep(DELAY)

# Fallback spatial: GeoNames descriptions
if counts["Spatial"] < 80:
    print("  Spatial fallback: NaturalQuestions with spatial keywords...")
    rows = hf_rows("google-research-datasets/natural_questions", config="default",
                   split="validation", offset=0, limit=500)
    SPATIAL_KW = {"located","north","south","east","west","left","right","above","below",
                  "between","behind","adjacent","near","distance","kilometers","miles"}
    for r in rows:
        q = r.get("question", {})
        text = clean_text(q.get("text", "") if isinstance(q, dict) else str(q))
        if text and set(re.findall(r'\b\w+\b', text.lower())) & SPATIAL_KW:
            add("nq_s5", "Spatial", "natural_questions_spatial", "positional_relational", text)
        if counts["Spatial"] >= 200: break

# ═══════════════════════════════════════════════════════════════════════════════
# 14. TextVerbal — BookCorpus-style via Wikipedia extracts (various lengths)
# ═══════════════════════════════════════════════════════════════════════════════

print("=== Wikipedia random → TextVerbal (varied lengths) ===")
for batch in range(8):
    url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    d = get_json(url)
    extract = clean_text(d.get("extract", ""))
    if len(extract) > 50:
        # Short version (32-token equivalent)
        sents = re.split(r'(?<=[.!?])\s+', extract)
        if sents:
            add("wp_m1_s", "TextVerbal", "wikipedia_random_short",
                "neutral_declarative_32tok", sents[0][:200])
        # Medium (128-token equivalent)
        if len(sents) >= 2:
            add("wp_m1_m", "TextVerbal", "wikipedia_random_medium",
                "neutral_declarative_128tok", " ".join(sents[:3])[:500])
        # Long (512-token equivalent)
        add("wp_m1_l", "TextVerbal", "wikipedia_random_long",
            "neutral_declarative_512tok", extract[:1000])
    time.sleep(0.3)

# Also: wikitext via HuggingFace
df_wt = hf_parquet("wikitext", config="wikitext-103-raw-v1", split="test")
if not df_wt.empty and "text" in df_wt.columns:
    for _, row in df_wt[df_wt["text"].str.len() > 100].sample(
            min(100, len(df_wt)), random_state=42).iterrows():
        add("wt_m1", "TextVerbal", "wikitext103", "neutral_declarative_512tok",
            clean_text(str(row["text"]))[:800])
        if counts["TextVerbal"] >= 100: break

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

# Deduplicate
seen = set()
deduped = []
for s in results:
    key = s["text"][:80].lower()
    if key not in seen:
        seen.add(key)
        deduped.append(s)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(deduped, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Saved {len(deduped)} stimuli to {OUT}")
print(f"\nBreakdown:")
final_counts = Counter(s["content_type"] for s in deduped)
for ct, n in sorted(final_counts.items(), key=lambda x: -x[1]):
    print(f"  {ct:<18} {n:>5}")
