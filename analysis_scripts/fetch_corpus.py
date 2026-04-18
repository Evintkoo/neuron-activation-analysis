#!/usr/bin/env python3
"""
Fetch real internet content across all 13 content types.
Sources: Wikipedia, arXiv, HackerNews, Reddit (JSON), BBC/Reuters RSS, PubMed
No API keys required — all public endpoints.
"""
import json, time, random, re, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "experiments" / "corpus" / "stimuli_fetched.json"

DELAY = 0.4  # seconds between requests — be polite

def get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "neuron-activation-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  WARN fetch {url[:80]}: {e}", file=sys.stderr)
        return ""

def get_json(url):
    raw = get(url)
    try: return json.loads(raw)
    except: return {}

def clean(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[\d+\]', '', text)          # remove citation markers
    text = re.sub(r'<[^>]+>', '', text)           # strip HTML
    return text[:1200]  # cap length

# ── Counters ──────────────────────────────────────────────────────────────────
results = []
counts  = {}

def add(id_prefix, content_type, source_type, language_structure, text):
    text = clean(text)
    if len(text) < 40: return
    idx = counts.get(content_type, 0) + 1
    counts[content_type] = idx
    results.append({
        "id":                 f"{id_prefix}_{idx:04d}",
        "content_type":       content_type,
        "source_type":        source_type,
        "language_structure": language_structure,
        "text":               text,
    })
    print(f"  [{content_type:<15}] {id_prefix}_{idx:04d}: {text[:60]}…")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. WIKIPEDIA — Factual, Abstract, Spatial, TextVerbal
# ═══════════════════════════════════════════════════════════════════════════════

def wiki_extract(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    d = get_json(url)
    return d.get("extract", "")

def wiki_random(n=60):
    url = f"https://en.wikipedia.org/api/rest_v1/page/random/summary"
    entries = []
    for _ in range(n):
        d = get_json(url)
        entries.append(d)
        time.sleep(DELAY)
    return entries

def wiki_category(category, n=30):
    url = (f"https://en.wikipedia.org/w/api.php?action=query&list=categorymembers"
           f"&cmtitle=Category:{urllib.parse.quote(category)}&cmlimit={n}&format=json&cmtype=page")
    d = get_json(url)
    titles = [m["title"] for m in d.get("query",{}).get("categorymembers",[])]
    results_list = []
    for t in titles[:n]:
        extract = wiki_extract(t)
        if extract:
            results_list.append((t, extract))
        time.sleep(DELAY)
    return results_list

print("=== Wikipedia: Factual (short declarative) ===")
factual_categories = [
    "Chemical_elements", "Countries_by_GDP", "Astronomical_objects",
    "Historic_earthquakes", "Centenarians", "Rivers_of_Europe",
]
for cat in factual_categories:
    for title, extract in wiki_category(cat, n=12):
        # First sentence only → short factual
        sent = re.split(r'(?<=[.!?])\s+', extract)[0]
        add("wk_s4", "Factual", "wikipedia_first_sentence", "declarative_third_person", sent)

print("\n=== Wikipedia: Abstract (philosophy, mathematics, physics) ===")
abstract_topics = [
    "Gödel's incompleteness theorems", "Quantum superposition", "Emergence",
    "Epistemic_injustice", "Determinism", "Self-reference", "Holism",
    "Philosophy of mind", "Formal_language", "Type theory",
    "Information entropy", "Computability theory", "Ontology",
    "Modal logic", "Falsifiability", "Reductionism",
]
for topic in abstract_topics:
    extract = wiki_extract(topic)
    if extract:
        # Take first 2 sentences — dense conceptual
        sents = re.split(r'(?<=[.!?])\s+', extract)
        text = " ".join(sents[:2])
        add("wk_s2", "Abstract", "wikipedia_philosophy_math", "dense_nominal_nested", text)
    time.sleep(DELAY)

print("\n=== Wikipedia: Spatial (geography, anatomy, architecture) ===")
spatial_topics = [
    "Amazon_River", "Hippocampus", "Great_Barrier_Reef", "Mariana_Trench",
    "Cerebral_cortex", "Sahara", "Mount_Everest", "Tectonic_plate",
    "Retina", "Antarctic_ice_sheet", "English_Channel", "Corpus_callosum",
    "Nile_River", "Thyroid_gland", "Mediterranean_Sea", "Spinal_cord",
    "Tibetan_Plateau", "Rift_Valley", "Louvre", "Aorta",
]
for topic in spatial_topics:
    extract = wiki_extract(topic)
    if extract:
        sents = re.split(r'(?<=[.!?])\s+', extract)
        text = " ".join(sents[:2])
        add("wk_s5", "Spatial", "wikipedia_geography_anatomy", "positional_relational", text)
    time.sleep(DELAY)

print("\n=== Wikipedia: TextVerbal (long-form, varied lengths) ===")
long_topics = [
    "World_War_II", "Climate_change", "Artificial_intelligence",
    "Human_genome", "Black_hole", "COVID-19_pandemic",
    "French_Revolution", "Theory_of_relativity", "Evolution",
    "International_Space_Station", "CRISPR", "Roman_Empire",
]
for topic in long_topics:
    extract = wiki_extract(topic)
    if extract:
        add("wk_m1", "TextVerbal", "wikipedia_long_form", "neutral_declarative_512tok", extract)
    time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ARXIV — Abstract (scientific), Novelty (breakthroughs)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== arXiv: Abstract + Novelty ===")

def arxiv_search(query, max_results=25, category=""):
    cat_filter = f"+AND+cat:{category}" if category else ""
    url = (f"https://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}"
           f"{cat_filter}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending")
    raw = get(url)
    if not raw: return []
    try:
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall("atom:entry", ns):
            title   = (entry.findtext("atom:title",   "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            entries.append((title, summary))
        return entries
    except:
        return []

# Abstract: theory-heavy papers
for query, cat in [
    ("quantum information entropy", "quant-ph"),
    ("topological manifolds algebraic", "math.AT"),
    ("consciousness neural correlates", "q-bio.NC"),
    ("language model emergent capabilities", "cs.CL"),
    ("causal inference counterfactual", "stat.ML"),
]:
    for title, abstract in arxiv_search(query, max_results=8, category=cat):
        if abstract:
            add("ax_s2", "Abstract", "arxiv_abstract", "formal_definition", abstract[:800])
    time.sleep(DELAY)

# Novelty: recent surprising findings
for query in [
    "surprising unexpected discovery breakthrough",
    "first ever observation novel phenomenon",
    "anomalous unexpected experimental result",
    "unprecedented new approach outperforms",
]:
    for title, abstract in arxiv_search(query, max_results=10):
        if abstract and any(w in abstract.lower() for w in
                            ["first", "novel", "unprecedented", "surprising", "unexpected", "outperform"]):
            add("ax_b4", "Novelty", "arxiv_breakthrough", "superlative_first_ever", abstract[:600])
    time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HACKERNEWS — Novelty (tech), Abstract (technical discussion)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== HackerNews: Novelty + Abstract ===")

def hn_top(n=200):
    ids = get_json("https://hacker-news.firebaseio.com/v0/topstories.json") or []
    return ids[:n]

def hn_item(item_id):
    return get_json(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json")

hn_ids = hn_top(300)
random.shuffle(hn_ids)
hn_novel = 0; hn_abst = 0
for item_id in hn_ids:
    if hn_novel >= 25 and hn_abst >= 20: break
    item = hn_item(item_id)
    if not item: continue
    title = item.get("title", "")
    text  = item.get("text", "") or ""
    text  = clean(re.sub(r'<[^>]+>', ' ', text))

    # Novelty: "show hn", launch posts, "I built", breakthroughs
    if hn_novel < 25 and any(w in title.lower() for w in
                              ["first", "new", "launch", "announce", "break", "discover",
                               "never", "world", "fastest", "largest", "open source"]):
        body = f"{title}. {text}" if text else title
        if len(body) > 40:
            add("hn_b4", "Novelty", "hackernews_announcement", "surprise_framing", body[:600])
            hn_novel += 1

    # Abstract: Ask HN, technical deep-dives
    elif hn_abst < 20 and title.startswith("Ask HN:") and text and len(text) > 100:
        add("hn_s2", "Abstract", "hackernews_discussion", "conditional_argument", text[:600])
        hn_abst += 1

    time.sleep(DELAY * 0.5)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. REDDIT (public JSON API) — Social, Narrative, Emotional, Reward
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Reddit: Social, Narrative, Emotional, Reward ===")

def reddit_posts(subreddit, sort="hot", limit=40, time_filter="month"):
    url = (f"https://www.reddit.com/r/{subreddit}/{sort}.json"
           f"?limit={limit}&t={time_filter}")
    d = get_json(url)
    posts = []
    for child in d.get("data", {}).get("children", []):
        post = child.get("data", {})
        title = post.get("title", "")
        body  = post.get("selftext", "")
        score = post.get("score", 0)
        posts.append({"title": title, "body": body, "score": score})
    return posts

# Social — AITA, relationship_advice
for sub in ["AmItheAsshole", "relationship_advice", "socialskills"]:
    for post in reddit_posts(sub, sort="top", limit=30):
        text = post["body"] if len(post["body"]) > 80 else post["title"]
        if len(text) > 80:
            add("rd_b1", "Social", f"reddit_{sub}", "multi_agent_dialogue", text[:700])
    time.sleep(DELAY)

# Narrative — tifu, personalfinance, survivorship stories
for sub in ["tifu", "TrueOffMyChest", "confession"]:
    for post in reddit_posts(sub, sort="top", limit=30):
        if len(post["body"]) > 150:
            add("rd_s1", "Narrative", f"reddit_{sub}", "chronological_causal", post["body"][:700])
    time.sleep(DELAY)

# Emotional — r/happy, r/sad, r/offmychest
for sub in ["happy", "sad", "offmychest", "grief"]:
    for post in reddit_posts(sub, sort="top", limit=25):
        text = post["body"] if len(post["body"]) > 80 else post["title"]
        if len(text) > 80:
            add("rd_s3", "Emotional", f"reddit_{sub}", "first_person_affective", text[:600])
    time.sleep(DELAY)

# Reward — success stories, MadeMeSmile
for sub in ["MadeMeSmile", "wholesomememes", "GetMotivated", "CasualConversation"]:
    for post in reddit_posts(sub, sort="top", limit=25):
        text = post["body"] if len(post["body"]) > 80 else post["title"]
        if len(text) > 40:
            add("rd_b2", "Reward", f"reddit_{sub}", "positive_outcome", text[:600])
    time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BBC / REUTERS RSS — ThreatSafety, Novelty
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== BBC/Reuters RSS: ThreatSafety, Novelty ===")

def parse_rss(url, max_items=40):
    raw = get(url)
    if not raw: return []
    try:
        root = ET.fromstring(raw)
        items = []
        for item in root.iter("item"):
            title       = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            description = re.sub(r'<[^>]+>', '', description)
            items.append((title, description))
        return items[:max_items]
    except:
        return []

threat_keywords = [
    "kill","dead","attack","crash","fire","flood","earthquake","storm","arrest",
    "killed","deaths","injured","explosion","war","crisis","emergency","recall",
    "warning","threat","danger","collapse","shooting","bombing","outbreak",
]
novel_keywords  = [
    "first","new","discover","launch","announce","breakthrough","record",
    "scientists","researchers","study","found","reveals","develops","creates",
]

rss_feeds = [
    ("https://feeds.bbci.co.uk/news/rss.xml",                "bbc_world"),
    ("https://feeds.bbci.co.uk/news/science_and_environment/rss.xml", "bbc_science"),
    ("https://feeds.bbci.co.uk/news/health/rss.xml",         "bbc_health"),
    ("https://feeds.bbci.co.uk/news/technology/rss.xml",     "bbc_tech"),
    ("https://rss.nytimes.com/services/xml/rss/nyt/World.xml","nyt_world"),
    ("https://rss.nytimes.com/services/xml/rss/nyt/Science.xml","nyt_science"),
    ("https://feeds.reuters.com/reuters/topNews",             "reuters_top"),
]

for feed_url, source in rss_feeds:
    items = parse_rss(feed_url, max_items=50)
    for title, desc in items:
        text = f"{title}. {desc}" if desc else title
        text = clean(text)
        if len(text) < 40: continue
        tl = text.lower()
        if any(k in tl for k in threat_keywords):
            add("rss_b3", "ThreatSafety", source, "short_declarative_present_tense", text[:500])
        elif any(k in tl for k in novel_keywords):
            add("rss_b4", "Novelty",      source, "surprise_framing",                text[:500])
    time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PUBMED ABSTRACTS — Abstract, Factual (medical), Novelty (medical breakthroughs)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== PubMed: Abstract + Factual (medical) ===")

def pubmed_search(query, max_results=20):
    search_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                  f"?db=pubmed&term={urllib.parse.quote(query)}&retmax={max_results}&retmode=json")
    d = get_json(search_url)
    ids = d.get("esearchresult", {}).get("idlist", [])
    if not ids: return []
    fetch_url = (f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                 f"?db=pubmed&id={','.join(ids)}&retmode=xml&rettype=abstract")
    raw = get(fetch_url)
    abstracts = []
    try:
        root = ET.fromstring(raw)
        for art in root.findall(".//PubmedArticle"):
            abstract_el = art.find(".//AbstractText")
            if abstract_el is not None and abstract_el.text:
                abstracts.append(abstract_el.text.strip())
    except:
        pass
    return abstracts

for query in [
    "neuroscience fMRI brain activation language",
    "cognitive neuroscience emotional processing amygdala",
    "randomized controlled trial clinical outcome",
    "novel therapeutic approach treatment efficacy",
]:
    for abstract in pubmed_search(query, max_results=15):
        if len(abstract) > 80:
            if any(w in abstract.lower() for w in ["novel","first","new","demonstrate","show"]):
                add("pm_b4", "Novelty", "pubmed_abstract", "academic_breakthrough", abstract[:700])
            else:
                add("pm_s2", "Abstract", "pubmed_abstract", "formal_definition", abstract[:700])
    time.sleep(1.0)  # PubMed rate limit

# ═══════════════════════════════════════════════════════════════════════════════
# 7. WIKIPEDIA "Current Events" — ThreatSafety, Novelty (live news)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Wikipedia Current Events ===")

def wiki_current_events():
    url = "https://en.wikipedia.org/w/api.php?action=parse&page=Portal:Current_events&prop=wikitext&format=json"
    d = get_json(url)
    wikitext = d.get("parse", {}).get("wikitext", {}).get("*", "")
    lines = [l.strip("*# ") for l in wikitext.split("\n") if l.strip().startswith("*")]
    return [re.sub(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', r'\1', l) for l in lines if len(l) > 40]

for line in wiki_current_events()[:60]:
    ll = line.lower()
    if any(k in ll for k in threat_keywords):
        add("ce_b3", "ThreatSafety", "wikipedia_current_events", "short_declarative_present_tense", line[:400])
    elif any(k in ll for k in novel_keywords):
        add("ce_b4", "Novelty",      "wikipedia_current_events", "surprise_framing",                line[:400])

# ═══════════════════════════════════════════════════════════════════════════════
# 8. GUTENBERG — Narrative (classic literature excerpts)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== Project Gutenberg: Narrative ===")

gutenberg_books = [
    ("84",   "Frankenstein"),
    ("1342", "Pride and Prejudice"),
    ("11",   "Alice in Wonderland"),
    ("74",   "Tom Sawyer"),
    ("2701", "Moby Dick"),
    ("1661", "Sherlock Holmes"),
    ("98",   "A Tale of Two Cities"),
    ("36",   "The War of the Worlds"),
    ("2554", "Crime and Punishment"),
    ("1400", "Great Expectations"),
]

def gutenberg_excerpt(book_id, offset_chars=2000, length=700):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    raw = get(url)
    if not raw: return ""
    # Skip header
    start = raw.find("*** START OF")
    if start > 0:
        raw = raw[start+30:]
    raw = re.sub(r'\s+', ' ', raw)
    # Take a chunk from the middle (avoid header/footer)
    mid = len(raw) // 3
    chunk = raw[mid + offset_chars: mid + offset_chars + length]
    return chunk.strip()

for book_id, title in gutenberg_books:
    for offset in [0, 3000, 6000]:
        excerpt = gutenberg_excerpt(book_id, offset_chars=offset)
        if len(excerpt) > 100:
            add("gt_s1", "Narrative", f"gutenberg_{title.lower().replace(' ','_')}",
                "chronological_causal", excerpt)
    time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. OPEN STREET MAP NOMINATIM — Spatial (place descriptions)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== OpenStreetMap: Spatial ===")

places = [
    "Eiffel Tower, Paris", "Times Square, New York", "Mount Fuji, Japan",
    "Sydney Opera House", "Grand Canyon, Arizona", "Colosseum, Rome",
    "Amazon rainforest", "Sahara Desert", "Nile Delta", "Manhattan",
    "Tokyo Station", "London Bridge", "Golden Gate Bridge",
    "Yellowstone National Park", "Machu Picchu", "Great Wall of China",
    "Niagara Falls", "Stonehenge", "Angkor Wat", "Kilimanjaro",
]

def nominatim_describe(place):
    url = (f"https://nominatim.openstreetmap.org/search"
           f"?q={urllib.parse.quote(place)}&format=json&limit=1&addressdetails=1")
    results = get_json(url)
    if not results: return ""
    r = results[0]
    parts = []
    addr  = r.get("address", {})
    disp  = r.get("display_name", "")
    lat, lon = r.get("lat",""), r.get("lon","")
    if disp:
        parts.append(f"{place} is located at {disp}.")
    if lat and lon:
        ns = "N" if float(lat) >= 0 else "S"
        ew = "E" if float(lon) >= 0 else "W"
        parts.append(f"Coordinates: {abs(float(lat)):.4f}°{ns}, {abs(float(lon)):.4f}°{ew}.")
    if addr.get("country"):
        parts.append(f"Country: {addr['country']}.")
    return " ".join(parts)

for place in places:
    desc = nominatim_describe(place)
    if desc:
        add("osm_s5", "Spatial", "openstreetmap_nominatim", "positional_relational", desc)
    time.sleep(1.1)  # Nominatim rate limit: 1 req/sec

# ═══════════════════════════════════════════════════════════════════════════════
# 10. AUDIO/IMAGE/MULTIMODAL — descriptive captions from Wikipedia commons
# ═══════════════════════════════════════════════════════════════════════════════

print("\n=== WikiCommons captions: M2/M3/M4 ===")

def wiki_image_captions(category, n=20):
    url = (f"https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers"
           f"&cmtitle=Category:{urllib.parse.quote(category)}&cmlimit={n}&cmtype=file&format=json")
    d = get_json(url)
    captions = []
    for m in d.get("query",{}).get("categorymembers",[]):
        title = m["title"].replace("File:", "").replace(".jpg","").replace(".png","").replace("_"," ")
        if len(title) > 15:
            captions.append(title)
    return captions

visual_cats = [
    "Landscape_photographs", "Aerial_photographs", "Abstract_art",
    "Portrait_photography", "Macro_photography", "Night_photography",
]
for cat in visual_cats:
    for cap in wiki_image_captions(cat, n=15):
        add("wc_m2", "ImageVisual", "wikimedia_commons", "sensory_descriptive",
            f"A photograph: {cap}.")
    time.sleep(DELAY)

audio_topics = [
    ("bird song recording", "AudioText", "nature_recording", "acoustic_descriptive"),
    ("orchestral symphony performance", "AudioText", "music_description", "spectral_descriptive"),
    ("crowd noise stadium atmosphere", "AudioText", "crowd_recording", "acoustic_descriptive"),
    ("thunder storm rain audio", "AudioText", "nature_recording", "acoustic_descriptive"),
]
audio_templates = [
    "A high-fidelity audio recording of {topic}, characterized by its distinctive tonal qualities and spatial presence.",
    "Field recording: {topic}. The soundscape is rich with layered frequencies and dynamic range.",
    "An immersive audio capture of {topic}, with clearly defined foreground and background acoustic elements.",
]
for query, ct, src, ls in audio_topics:
    for template in audio_templates:
        text = template.format(topic=query)
        add("aud_m3", ct, src, ls, text)

# Multimodal descriptions from Wikipedia "film" and "documentary" articles
mm_topics = [
    "Planet_Earth_(TV_series)", "The_Blue_Planet", "March_of_the_Penguins",
    "Koyaanisqatsi", "Baraka_(film)", "Samsara_(2011_film)",
]
for topic in mm_topics:
    extract = wiki_extract(topic)
    if extract:
        add("wk_m4", "Multimodal", "wikipedia_documentary",
            "simultaneous_av", extract[:600])
    time.sleep(DELAY)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Saved {len(results)} stimuli to {OUT}")
print(f"\nBreakdown by content type:")
for ct, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {ct:<18} {n:>4}")
