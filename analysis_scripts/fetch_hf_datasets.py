#!/usr/bin/env python3
"""
Fetch thin content types using HuggingFace datasets library.
Targets: Social, Factual, Spatial, TextVerbal
"""
import json, random, re
from pathlib import Path
from collections import Counter
from datasets import load_dataset

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "experiments" / "corpus" / "stimuli_boost.json"
random.seed(42)

results = []

def clean(t):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', str(t))).strip()[:900]

def add(prefix, ct, src, ls, text):
    text = clean(text)
    if len(text) >= 40:
        results.append({"id": f"{prefix}_{len(results):05d}", "content_type": ct,
                        "source_type": src, "language_structure": ls, "text": text})

by_ct = lambda ct: sum(1 for x in results if x["content_type"] == ct)

# ── SOCIAL ────────────────────────────────────────────────────────────────────

print("=== Social: EmpatheticDialogues ===")
try:
    ds = load_dataset("facebook/empathetic_dialogues", split="train", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(500, len(ds)))))
    for r in rows:
        text = clean(r.get("utterance", ""))
        ctx = clean(r.get("context", ""))
        if ctx: text = f"{ctx}: {text}"
        add("ed_b1", "Social", "empathetic_dialogues", "multi_agent_dialogue", text)
        if by_ct("Social") >= 250: break
    print(f"  Social: {by_ct('Social')}")
except Exception as e:
    print(f"  WARN: {e}")

print("=== Social: social_i_qa ===")
try:
    ds = load_dataset("allenai/social_i_qa", split="train", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(400, len(ds)))))
    for r in rows:
        ctx = clean(r.get("context", ""))
        q   = clean(r.get("question", ""))
        ans = clean(r.get("answerA", ""))
        text = f"{ctx} {q} {ans}".strip() if ctx else f"{q} {ans}".strip()
        add("si_b1", "Social", "social_iqa", "theory_of_mind", text)
        if by_ct("Social") >= 300: break
    print(f"  Social: {by_ct('Social')}")
except Exception as e:
    print(f"  WARN: {e}")

# ── FACTUAL ───────────────────────────────────────────────────────────────────

print("=== Factual: TriviaQA ===")
try:
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation",
                      trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(400, len(ds)))))
    for r in rows:
        q = clean(r.get("question", ""))
        ans = r.get("answer", {})
        a = clean(ans.get("value", "") if isinstance(ans, dict) else str(ans))
        if q and a:
            add("tq_s4", "Factual", "triviaqa_wikipedia", "declarative_third_person", f"{q} {a}.")
        if by_ct("Factual") >= 200: break
    print(f"  Factual: {by_ct('Factual')}")
except Exception as e:
    print(f"  WARN: {e}")

print("=== Factual: SciQ ===")
try:
    ds = load_dataset("allenai/sciq", split="train", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(400, len(ds)))))
    for r in rows:
        q = clean(r.get("question", ""))
        a = clean(r.get("correct_answer", ""))
        if q and a:
            add("sc_s4", "Factual", "sciq_qa", "declarative_third_person", f"{q} {a}.")
        if by_ct("Factual") >= 300: break
    print(f"  Factual: {by_ct('Factual')}")
except Exception as e:
    print(f"  WARN: {e}")

# ── SPATIAL ───────────────────────────────────────────────────────────────────

print("=== Spatial: bAbI tasks (story-based spatial) ===")
try:
    ds = load_dataset("facebook/babi_qa", type="en", task_no="17",
                      split="train", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(300, len(ds)))))
    for r in rows:
        story = clean(" ".join(r.get("story", {}).get("text", [])))
        if len(story) > 40:
            add("ba_s5", "Spatial", "babi_spatial", "positional_relational", story)
        if by_ct("Spatial") >= 150: break
    print(f"  Spatial: {by_ct('Spatial')}")
except Exception as e:
    print(f"  WARN: {e}")

print("=== Spatial: GeoQA+ ===")
try:
    ds = load_dataset("Geometry3K/geometry3k", split="train", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(300, len(ds)))))
    for r in rows:
        text = clean(r.get("problem_text", r.get("text", "")))
        if len(text) > 40:
            add("gq_s5", "Spatial", "geometry3k", "positional_relational", text)
        if by_ct("Spatial") >= 200: break
    print(f"  Spatial: {by_ct('Spatial')}")
except Exception as e:
    print(f"  WARN: {e}")

print("=== Spatial: NLVR2 (spatial language reasoning) ===")
try:
    ds = load_dataset("lil-lab/nlvr", split="train", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42).select(range(min(400, len(ds)))))
    for r in rows:
        text = clean(r.get("sentence", r.get("text", "")))
        if len(text) > 20 and any(w in text.lower() for w in
            ["left","right","above","below","top","bottom","next","near","beside","between"]):
            add("nl_s5", "Spatial", "nlvr_spatial", "positional_relational", text)
        if by_ct("Spatial") >= 250: break
    print(f"  Spatial: {by_ct('Spatial')}")
except Exception as e:
    print(f"  WARN: {e}")

# ── TEXTVERBAL ────────────────────────────────────────────────────────────────

print("=== TextVerbal: WikiText-103 ===")
try:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="test", trust_remote_code=True)
    rows = list(ds.shuffle(seed=42))
    for r in rows:
        text = clean(r.get("text", ""))
        if len(text) < 100: continue
        # Short
        sents = re.split(r'(?<=[.!?])\s+', text)
        add("wt_m1_s", "TextVerbal", "wikitext103_short",  "neutral_declarative_32tok",  sents[0][:200])
        add("wt_m1_m", "TextVerbal", "wikitext103_medium", "neutral_declarative_128tok", " ".join(sents[:3])[:500])
        add("wt_m1_l", "TextVerbal", "wikitext103_long",   "neutral_declarative_512tok", text[:800])
        if by_ct("TextVerbal") >= 250: break
    print(f"  TextVerbal: {by_ct('TextVerbal')}")
except Exception as e:
    print(f"  WARN: {e}")

print("=== TextVerbal: BookCorpus (pg19) ===")
try:
    ds = load_dataset("deepmind/pg19", split="test", trust_remote_code=True,
                      streaming=True)
    for i, r in enumerate(ds):
        text = clean(r.get("text", ""))
        if len(text) > 200:
            words = text.split()
            mid = len(words) // 3
            chunk = " ".join(words[mid:mid+100])
            add("pg_m1", "TextVerbal", "pg19_books", "neutral_declarative_512tok", chunk)
        if by_ct("TextVerbal") >= 300 or i >= 30: break
    print(f"  TextVerbal: {by_ct('TextVerbal')}")
except Exception as e:
    print(f"  WARN: {e}")

# ── REWARD BOOST ─────────────────────────────────────────────────────────────

print("=== Reward: Amazon Reviews (5-star) ===")
try:
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                      "raw_review_All_Beauty", split="full",
                      trust_remote_code=True, streaming=True)
    for i, r in enumerate(ds):
        if r.get("rating", 0) == 5.0:
            text = clean(r.get("text", ""))
            if len(text) > 50:
                add("az_b2", "Reward", "amazon_reviews_5star", "positive_superlative", text[:600])
        if by_ct("Reward") >= 300 or i >= 2000: break
    print(f"  Reward: {by_ct('Reward')}")
except Exception as e:
    print(f"  WARN: {e}")

# ── SAVE ──────────────────────────────────────────────────────────────────────

final_counts = Counter(x["content_type"] for x in results)
print(f"\n{'='*50}")
print(f"Total boost: {len(results)}")
for ct, n in sorted(final_counts.items(), key=lambda x: -x[1]):
    print(f"  {ct:<18} {n}")

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUT}")
