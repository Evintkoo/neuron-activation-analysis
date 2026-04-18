# Activation Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an empirical sweep harness that sends 150 curated text stimuli to the TRIBE v2 server, collects cortical activation stats per stimulus, and produces ranked output and a region heatmap.

**Architecture:** A new Rust binary crate `experiments/sweep` reads `experiments/corpus/stimuli.json`, POSTs each stimulus to `localhost:8081/api/predict`, and writes three output files: `results/sweep_results.json` (full raw), `results/sweep_ranked.csv` (sorted by global_mean), and `results/region_heatmap.json` (13×6 content-type × region matrix). The binary runs in both demo mode and real-weights mode; demo results are flagged in output.

**Tech Stack:** Rust, `ureq 2` (blocking HTTP), `serde`/`serde_json`, manual CSV writing; no async runtime needed.

---

## File Map

| File | Role |
|------|------|
| `Cargo.toml` (workspace root) | Add `"experiments/sweep"` member |
| `experiments/sweep/Cargo.toml` | Binary crate manifest |
| `experiments/sweep/src/main.rs` | All sweep logic: types, HTTP, ranking, output, main |
| `experiments/corpus/stimuli.json` | 150 curated text stimuli across 13 content types |
| `results/.gitignore` | Ignore `sweep_results.json` (large); track CSV and heatmap |
| `scripts/download_weights.sh` | Hit tribe-server download endpoints for model weights |

---

### Task 1: Scaffold — crate skeleton + workspace registration

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `experiments/sweep/Cargo.toml`
- Create: `experiments/sweep/src/main.rs` (stub)

- [ ] **Step 1: Write the failing test**

```rust
// experiments/sweep/src/main.rs
fn main() {}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder_compiles() {
        assert!(true);
    }
}
```

- [ ] **Step 2: Create `experiments/sweep/Cargo.toml`**

```toml
[package]
name = "sweep"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "sweep"
path = "src/main.rs"

[dependencies]
serde      = { version = "1", features = ["derive"] }
serde_json = "1"
ureq       = { version = "2", features = ["json"] }
```

- [ ] **Step 3: Add `"experiments/sweep"` to workspace root `Cargo.toml`**

Change:
```toml
members = ["experiments", "analysis", "runner", "viz/builder"]
```
To:
```toml
members = ["experiments", "analysis", "runner", "viz/builder", "experiments/sweep"]
```

- [ ] **Step 4: Run test to verify it compiles**

```bash
cargo test -p sweep
```
Expected: `test placeholder_compiles ... ok`

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml experiments/sweep/Cargo.toml experiments/sweep/src/main.rs
git commit -m "feat: scaffold sweep binary crate"
```

---

### Task 2: Stimulus corpus JSON

**Files:**
- Create: `experiments/corpus/stimuli.json`

- [ ] **Step 1: Create `experiments/corpus/` and write `stimuli.json`**

```bash
mkdir -p experiments/corpus
```

Write `experiments/corpus/stimuli.json` — 150 entries, each:
```json
{ "id": "...", "content_type": "...", "source_type": "...", "language_structure": "...", "text": "..." }
```

Full corpus:

```json
[
  {"id":"b3_001","content_type":"ThreatSafety","source_type":"breaking_news","language_structure":"short_declarative_present_tense","text":"Evacuation order issued for coastal residents as Category 4 hurricane makes landfall."},
  {"id":"b3_002","content_type":"ThreatSafety","source_type":"health_advisory","language_structure":"imperative_urgent","text":"CDC issues urgent health advisory: outbreak of novel respiratory illness reported in three states. Avoid crowded indoor spaces immediately."},
  {"id":"b3_003","content_type":"ThreatSafety","source_type":"infrastructure_alert","language_structure":"short_declarative_present_tense","text":"Structural failure detected in bridge used by 50,000 commuters daily. Emergency closure in effect."},
  {"id":"b3_004","content_type":"ThreatSafety","source_type":"emergency_alert","language_structure":"imperative_urgent","text":"Active shooter reported at downtown office complex. Shelter in place immediately. Do not approach windows."},
  {"id":"b3_005","content_type":"ThreatSafety","source_type":"breaking_news","language_structure":"short_declarative_present_tense","text":"Wildfire jumps containment lines and threatens suburban neighborhoods. Mandatory evacuations expanded to Zone C."},
  {"id":"b3_006","content_type":"ThreatSafety","source_type":"product_recall","language_structure":"imperative_urgent","text":"Recall issued for infant formula linked to bacterial contamination. Stop use immediately and check lot numbers on the base of the can."},
  {"id":"b3_007","content_type":"ThreatSafety","source_type":"infrastructure_alert","language_structure":"short_declarative_present_tense","text":"Power grid failure across northeastern region. Hospitals switching to backup generators. Outages expected to last 48 hours."},
  {"id":"b3_008","content_type":"ThreatSafety","source_type":"emergency_alert","language_structure":"imperative_urgent","text":"Flash flood warning in effect until midnight. Do not attempt to cross flooded roads. Turn around, don't drown."},
  {"id":"b3_009","content_type":"ThreatSafety","source_type":"security_breach","language_structure":"short_declarative_present_tense","text":"Security breach detected at major national bank. Customer account data potentially compromised. Change passwords now."},
  {"id":"b3_010","content_type":"ThreatSafety","source_type":"breaking_news","language_structure":"short_declarative_present_tense","text":"Train derailment near chemical plant triggers hazmat response. Area residents warned to stay indoors and seal windows."},
  {"id":"b3_011","content_type":"ThreatSafety","source_type":"emergency_alert","language_structure":"imperative_urgent","text":"Tornado touchdown confirmed 2 miles north of city center. Seek shelter now. Do not wait until you see the funnel cloud."},
  {"id":"b3_012","content_type":"ThreatSafety","source_type":"emergency_alert","language_structure":"short_declarative_present_tense","text":"Emergency: gas leak detected in downtown district. Large-scale evacuation underway. Do not use elevators or light switches."},

  {"id":"b4_001","content_type":"Novelty","source_type":"science_news","language_structure":"surprise_framing","text":"Scientists announce first successful room-temperature superconductor with independent verification from three separate laboratories."},
  {"id":"b4_002","content_type":"Novelty","source_type":"archaeology","language_structure":"superlative_first_ever","text":"Archaeologists uncover intact 3,000-year-old city beneath Saharan dunes — the largest ever found, containing an estimated 20,000 undisturbed rooms."},
  {"id":"b4_003","content_type":"Novelty","source_type":"ai_research","language_structure":"unexpected_contrast","text":"AI system spontaneously develops novel mathematical proof no human has previously constructed, then refuses to explain its reasoning."},
  {"id":"b4_004","content_type":"Novelty","source_type":"astronomy","language_structure":"surprise_framing","text":"Astronomers detect repeating radio signal from a source less than 100 light-years away. Origin unknown. Pattern does not match any known natural phenomenon."},
  {"id":"b4_005","content_type":"Novelty","source_type":"biology","language_structure":"unexpected_contrast","text":"First documented case of a deep-sea organism consuming plastic as its primary food source and converting it to organic compounds."},
  {"id":"b4_006","content_type":"Novelty","source_type":"medical_research","language_structure":"accidental_discovery","text":"Researchers accidentally discover compound that reverses cellular aging markers in mice by 40%. The compound was being tested for an unrelated purpose."},
  {"id":"b4_007","content_type":"Novelty","source_type":"neurotechnology","language_structure":"superlative_first_ever","text":"Brain-computer interface allows paralyzed patient to write 90 words per minute using thought alone — three times faster than previous records."},
  {"id":"b4_008","content_type":"Novelty","source_type":"ocean_exploration","language_structure":"surprise_framing","text":"Ocean trench expedition reveals bioluminescent ecosystem of entirely unknown species, covering an area the size of France, never previously observed by science."},
  {"id":"b4_009","content_type":"Novelty","source_type":"physics","language_structure":"unexpected_contrast","text":"Physicists observe particle that appears to travel backward in time under controlled laboratory conditions. Three independent teams replicated the result."},
  {"id":"b4_010","content_type":"Novelty","source_type":"technology","language_structure":"superlative_first_ever","text":"Company announces battery that charges to 80% capacity in under 30 seconds and holds charge for six months. Independent tests confirmed the claims."},
  {"id":"b4_011","content_type":"Novelty","source_type":"wildlife","language_structure":"superlative_first_ever","text":"First-ever video footage captured of giant squid hunting prey in its natural deep-sea habitat, revealing behavior that contradicts 150 years of scientific assumption."},
  {"id":"b4_012","content_type":"Novelty","source_type":"linguistics","language_structure":"unexpected_contrast","text":"Linguists discover isolated tribe in the Amazon whose language has a grammatical structure impossible under every current theory of universal grammar."},

  {"id":"s1_001","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"I was driving home from work when my phone rang. It was my sister, crying. She told me dad had collapsed. I turned the car around and drove three hours straight, not stopping once. When I finally got to the hospital, he was already in surgery. Then the doctor came out, and everything changed."},
  {"id":"s1_002","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"We had been friends for twenty years before the argument. It started over something small — who had said what at dinner three months ago — and then it wasn't small anymore. She packed a bag, walked out, and I haven't heard from her since. That was March. It is now November."},
  {"id":"s1_003","content_type":"Narrative","source_type":"business_news","language_structure":"chronological_causal","text":"The startup launched in April with twelve employees and $400,000 in seed funding. By June they had signed their first enterprise contract. By September they had grown to 80 people. Then the audit happened, and within a week the founding team was gone."},
  {"id":"s1_004","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"He trained two years for the marathon. Every morning at 5 a.m., rain or cold. He lost 40 pounds, quit drinking, rebuilt his life around the race. On mile 23 his knee gave out. He finished anyway, crawling across the line."},
  {"id":"s1_005","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"She applied to 47 jobs over six months. Each rejection came as the same form email. Then one Tuesday afternoon her phone rang with an unknown number. She almost didn't answer. But she did."},
  {"id":"s1_006","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"The letter arrived on a Thursday. It had been mailed three weeks earlier from an address she didn't recognize. Inside was a photograph of her grandmother taken in 1943, with a name written on the back she had never heard before."},
  {"id":"s1_007","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"My son said his first word at 18 months. Then he stopped talking entirely. The doctors took six months to give us the diagnosis. The next four years were the hardest of my life. Last week, he stood up in front of his class and gave a presentation."},
  {"id":"s1_008","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"They met on a delayed train during a blizzard and talked for five hours. Then the train arrived, they went their separate ways, and neither had thought to ask for a phone number. That was 12 years ago. Last Saturday, they got married."},
  {"id":"s1_009","content_type":"Narrative","source_type":"business_news","language_structure":"chronological_causal","text":"The company had been in the family for three generations. My grandfather built it, my father ran it for 30 years, and when I took over we had 200 employees. Two years later the factory burned down. Then I made the decision that cost me everything."},
  {"id":"s1_010","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"She had been clean for three years when she relapsed. It started with a single drink at a work event. By the next morning she was in a motel she didn't remember booking. The recovery counselor she called that afternoon would later become her husband."},
  {"id":"s1_011","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"The old man arrived at the immigration desk with one suitcase and forty dollars. He spoke no English and had a cousin's address on a piece of paper. Forty years later, his granddaughter graduated from Yale."},
  {"id":"s1_012","content_type":"Narrative","source_type":"reddit_personal_story","language_structure":"chronological_causal","text":"I submitted the manuscript to 12 publishers over 18 months. Every rejection stung. The 13th response was different. It started with: 'We would like to offer you a contract.' I read it four times before I believed it."},

  {"id":"b1_001","content_type":"Social","source_type":"reddit_observation","language_structure":"theory_of_mind","text":"She knew he was lying. Not because of what he said, but because of what he didn't. He had always gestured with his hands when telling a story, and right now his hands were perfectly still."},
  {"id":"b1_002","content_type":"Social","source_type":"twitter","language_structure":"multi_agent_dialogue","text":"Mom: Are you okay? Me: I'm fine. Mom: Are you sure? Me: Yes. (Neither of us believed it.)"},
  {"id":"b1_003","content_type":"Social","source_type":"twitter","language_structure":"implied_subtext","text":"'I'm not angry,' she said, in the tone she used specifically when she was angry."},
  {"id":"b1_004","content_type":"Social","source_type":"reddit_observation","language_structure":"theory_of_mind","text":"He pretended not to see her across the restaurant. She pretended not to see him pretending. They had been doing this for months."},
  {"id":"b1_005","content_type":"Social","source_type":"workplace_observation","language_structure":"implied_subtext","text":"The team lead sent an email at 11 p.m. with the subject line 'Quick question.' Everyone who received it knew it was not a quick question."},
  {"id":"b1_006","content_type":"Social","source_type":"customer_support","language_structure":"multi_agent_dialogue","text":"User: this is urgent. Support: I understand this feels urgent. User: no, it IS urgent. Support: I hear that you're frustrated. User: [leaves 1-star review]"},
  {"id":"b1_007","content_type":"Social","source_type":"twitter","language_structure":"implied_subtext","text":"She said 'we should get coffee sometime' and he made the mistake of taking out his phone and asking when she was free."},
  {"id":"b1_008","content_type":"Social","source_type":"reddit_observation","language_structure":"implied_subtext","text":"When he said 'do whatever you want,' she understood that there was exactly one acceptable answer."},
  {"id":"b1_009","content_type":"Social","source_type":"reddit_observation","language_structure":"theory_of_mind","text":"The apology came three years after the incident. It was specific, remorseful, and asked for nothing. She accepted it. She did not forgive him."},
  {"id":"b1_010","content_type":"Social","source_type":"reddit_observation","language_structure":"theory_of_mind","text":"They had the same argument every Thanksgiving. This year both of them knew it was coming, both of them started it anyway, and both of them felt wronged by the outcome."},
  {"id":"b1_011","content_type":"Social","source_type":"twitter","language_structure":"implied_subtext","text":"He could tell from the length of her reply — six words — exactly how much trouble he was in."},
  {"id":"b1_012","content_type":"Social","source_type":"reddit_observation","language_structure":"implied_subtext","text":"'It's not about the dishes,' she said. They both knew she was right. It had never been about the dishes."},

  {"id":"s3_001","content_type":"Emotional","source_type":"reddit_personal","language_structure":"first_person_positive","text":"My daughter called today just to say she loves me. Not for any reason. Just because. I have been smiling for six hours straight."},
  {"id":"s3_002","content_type":"Emotional","source_type":"reddit_personal","language_structure":"first_person_positive","text":"After 11 years of trying, the adoption agency called this morning. We have a daughter. I cannot stop shaking. I cannot stop crying. I cannot stop smiling."},
  {"id":"s3_003","content_type":"Emotional","source_type":"twitter","language_structure":"first_person_positive","text":"I got the job. I got the job. I got the job. Three years of working toward this and I got the job."},
  {"id":"s3_004","content_type":"Emotional","source_type":"reddit_personal","language_structure":"first_person_negative","text":"I have been sitting in this parking lot for 45 minutes because I can't bring myself to go inside and face people. I don't know what's wrong with me today."},
  {"id":"s3_005","content_type":"Emotional","source_type":"twitter","language_structure":"metaphorical_negative","text":"The grief doesn't get smaller. You just get bigger around it. That's the only thing anyone got right about this."},
  {"id":"s3_006","content_type":"Emotional","source_type":"twitter","language_structure":"first_person_negative","text":"Some nights I open my phone 30 times and close it without doing anything, just to have something to hold."},
  {"id":"s3_007","content_type":"Emotional","source_type":"reddit_personal","language_structure":"mixed_valence","text":"Today was my last day at the company I spent 14 years building. They held a goodbye party. I laughed at the speeches and cried on the drive home."},
  {"id":"s3_008","content_type":"Emotional","source_type":"reddit_personal","language_structure":"mixed_valence","text":"We put the dog to sleep this morning. He was 15. The house is unbearably quiet and I keep looking at his bed."},
  {"id":"s3_009","content_type":"Emotional","source_type":"reddit_personal","language_structure":"mixed_valence","text":"She called to say she was proud of me. My mother. Who has never once said that before. I didn't know what to do with it so I said thank you and hung up and then fell apart."},
  {"id":"s3_010","content_type":"Emotional","source_type":"reddit_personal","language_structure":"mixed_valence","text":"The wedding was perfect. I barely remember any of it. I kept thinking: don't forget this, don't forget this. I forgot almost all of it."},
  {"id":"s3_011","content_type":"Emotional","source_type":"reddit_personal","language_structure":"mixed_valence","text":"I found a voicemail from my dad from four years before he died. I listened to it twice and could not bring myself to listen a third time."},
  {"id":"s3_012","content_type":"Emotional","source_type":"reddit_personal","language_structure":"suppressed_negative","text":"My best friend told me she has cancer. I smiled and said we'd get through it together. Then I went to the bathroom and couldn't breathe for five minutes."},

  {"id":"m1_32_001","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_32tok","text":"The mitochondria are membrane-bound organelles found in most eukaryotic cells that generate adenosine triphosphate through cellular respiration."},
  {"id":"m1_32_002","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_32tok","text":"Quantum entanglement is a physical phenomenon in which two particles become correlated such that measurement of one instantly influences the other regardless of distance."},
  {"id":"m1_32_003","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_32tok","text":"The French Revolution began in 1789 with the convocation of the Estates-General and ended in 1799 with Napoleon Bonaparte's coup d'état."},
  {"id":"m1_32_004","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_32tok","text":"Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll."},

  {"id":"m1_128_001","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_128tok","text":"The Internet of Things refers to the network of physical devices, vehicles, appliances, and other objects embedded with sensors, software, and connectivity enabling them to collect and exchange data. As of 2024, there are estimated to be over 15 billion connected IoT devices worldwide, ranging from smart home appliances and wearable fitness trackers to industrial sensors and autonomous vehicles. The primary challenges in IoT deployment include security vulnerabilities, data privacy concerns, interoperability between devices from different manufacturers, and the sheer volume of data generated requiring efficient processing infrastructure."},
  {"id":"m1_128_002","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_128tok","text":"The amygdala is an almond-shaped cluster of neurons located deep within the medial temporal lobe of the brain's cerebrum. It plays a primary role in the processing of memory, decision-making, and emotional responses, particularly fear and aggression. The amygdala receives sensory information from multiple brain areas and is closely linked to the hippocampus, which is involved in memory formation. Damage to the amygdala in humans has been associated with impairment in recognizing fear expressions and making trustworthiness judgments about strangers."},
  {"id":"m1_128_003","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_128tok","text":"Supply chain management is the coordination of all activities involved in sourcing, procuring, producing, and delivering products or services to end consumers. Effective supply chain management requires balancing cost efficiency with resilience, particularly in the face of disruptions such as natural disasters, geopolitical instability, or pandemic-related shutdowns. Modern supply chains rely heavily on data analytics, real-time tracking, and automation to optimize inventory levels, reduce lead times, and forecast demand. The COVID-19 pandemic exposed significant vulnerabilities in global supply chains reliant on single-source suppliers."},
  {"id":"m1_128_004","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_128tok","text":"Dark matter is a hypothetical form of matter that does not emit, absorb, or reflect electromagnetic radiation, making it invisible to telescopes across all wavelengths. It is estimated to constitute approximately 27% of the total mass-energy content of the universe. Evidence for its existence comes primarily from gravitational effects on visible matter, the rotation curves of galaxies, gravitational lensing, and large-scale structure formation. Despite decades of research, no direct detection of dark matter particles has been confirmed, and its fundamental nature remains one of the most significant unsolved problems in modern physics."},

  {"id":"m1_512_001","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_512tok","text":"Climate change refers to long-term shifts in global temperatures and weather patterns. While natural processes have always influenced Earth's climate, scientific consensus holds that since the mid-20th century, human activities — primarily the burning of fossil fuels — have become the dominant driver of observed climate change. The atmospheric concentration of carbon dioxide has increased from approximately 280 parts per million in pre-industrial times to over 420 parts per million as of 2024, a level not seen in at least 800,000 years as evidenced by ice core records. This accumulation traps outgoing infrared radiation, a phenomenon known as the greenhouse effect, leading to a rise in average global surface temperature of approximately 1.2 degrees Celsius above pre-industrial baselines. The consequences of this warming are manifold and increasingly well-documented: polar ice sheets are losing mass at accelerating rates, sea levels are rising at approximately 3.7 millimeters per year, extreme weather events including heat waves, heavy precipitation, droughts, and tropical cyclones are increasing in frequency and intensity, and ecosystem disruptions — including coral bleaching, shifts in species ranges, and altered migration patterns — are being observed across every biome. International efforts to limit warming, including the Paris Agreement signed by 196 parties in 2015, aim to hold global average temperature increase to well below 2 degrees Celsius above pre-industrial levels and pursue efforts to limit warming to 1.5 degrees Celsius. Achieving these targets requires rapid and far-reaching transitions in energy, land use, transport, and industry, with net-zero greenhouse gas emissions required by mid-century under most modeled pathways."},
  {"id":"m1_512_002","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_512tok","text":"The human immune system is a complex network of cells, tissues, organs, and signaling molecules that defends the body against pathogens, foreign substances, and aberrant cells including cancers. It is conventionally divided into two branches: the innate immune system, which provides rapid, nonspecific responses to infection, and the adaptive immune system, which generates highly specific, long-lasting defenses tailored to particular pathogens. The innate system's first line of defense comprises physical barriers — skin, mucous membranes, cilia — supplemented by chemical barriers such as lysozyme in saliva and low gastric pH. When pathogens breach these barriers, innate immune cells including macrophages, neutrophils, dendritic cells, and natural killer cells are rapidly recruited to sites of infection through chemical signals called cytokines and chemokines. These cells can directly destroy pathogens through phagocytosis and the release of antimicrobial compounds, and they present pathogen-derived antigens to T lymphocytes to initiate adaptive responses. The adaptive immune response involves two main cell types: B cells, which differentiate into plasma cells producing antigen-specific antibodies, and T cells, which include cytotoxic T cells that kill infected cells and helper T cells that coordinate both humoral and cell-mediated immunity. A critical feature of the adaptive immune system is immunological memory: after an initial encounter with a pathogen, a population of memory B and T cells persists for years or decades, enabling faster and stronger responses upon subsequent exposure — the principle underlying vaccination."},
  {"id":"m1_512_003","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_512tok","text":"CRISPR-Cas9 is a molecular tool adapted from a natural bacterial immune system that enables precise editing of DNA sequences in living organisms. In bacteria, CRISPR (Clustered Regularly Interspaced Short Palindromic Repeats) arrays store fragments of viral DNA as a form of genetic memory, allowing the Cas9 protein to recognize and cut matching sequences in future viral infections. Researchers Jennifer Doudna and Emmanuelle Charpentier, awarded the Nobel Prize in Chemistry in 2020, demonstrated in 2012 that this system could be reprogrammed to cut any target DNA sequence by designing a guide RNA matching the desired target. The implications for medicine, agriculture, and basic research have proven profound. In clinical medicine, CRISPR-based therapies have entered human trials for sickle cell disease, beta-thalassemia, certain cancers, and inherited blindness, with early results showing durable corrections in affected patients. In agriculture, CRISPR has been used to develop disease-resistant crops, remove allergens from foods, and improve yield and nutritional profiles without the regulatory hurdles associated with traditional transgenic GMO approaches. In basic research, CRISPR knockout screens across entire genomes have accelerated the identification of gene functions at a scale previously impossible. The technology raises significant ethical questions regarding germline editing — modifications to embryos that would be heritable — that have prompted international scientific moratoriums and ongoing regulatory debate. The birth of gene-edited twins in China in 2018 by researcher He Jiankui, widely condemned by the scientific community, illustrated the urgency of establishing clear ethical and regulatory frameworks before heritable human genome editing proceeds."},
  {"id":"m1_512_004","content_type":"TextVerbal","source_type":"wikipedia","language_structure":"neutral_declarative_512tok","text":"The history of the internet begins with ARPANET, a packet-switching network funded by the United States Department of Defense's Advanced Research Projects Agency, which first connected computers at UCLA, Stanford Research Institute, UC Santa Barbara, and the University of Utah in 1969. The development of TCP/IP protocols in the 1970s by Vint Cerf and Bob Kahn established the technical foundation for interconnecting heterogeneous networks — the 'network of networks' that would become the internet. The 1980s saw the expansion of internet access to universities and research institutions worldwide, while the introduction of DNS in 1984 replaced numerical IP addresses with human-readable domain names. The decisive transition to the modern internet came with Tim Berners-Lee's invention of the World Wide Web at CERN in 1989-1991: the combination of HTML, HTTP, and URLs created a hyperlinked document system that dramatically lowered the barrier to publishing and accessing information. The subsequent commercialization of the internet through the 1990s — marked by the Netscape IPO in 1995 and the dot-com boom — transformed it from an academic and government tool into a mass medium. By 2024, over 5.4 billion people had internet access, representing approximately 67% of the global population. The concentration of internet infrastructure and attention among a small number of platform companies — Google, Meta, Amazon, Microsoft, Apple — has prompted ongoing debate about market power, data privacy, content moderation, and the internet's relationship to democracy and public discourse."},

  {"id":"s4_001","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."},
  {"id":"s4_002","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The Great Wall of China stretches approximately 21,196 kilometers including all branches and segments."},
  {"id":"s4_003","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The human body contains approximately 37 trillion cells, with red blood cells constituting the largest share."},
  {"id":"s4_004","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"Mount Everest stands 8,848.86 meters above sea level, as measured in 2020."},
  {"id":"s4_005","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The speed of light in a vacuum is exactly 299,792,458 meters per second."},
  {"id":"s4_006","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"Napoleon Bonaparte was born on August 15, 1769, in Ajaccio, Corsica."},
  {"id":"s4_007","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The Eiffel Tower was completed in 1889 and stands 330 meters tall including its antenna."},
  {"id":"s4_008","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The Pacific Ocean covers approximately 165 million square kilometers, roughly 46% of Earth's water surface."},
  {"id":"s4_009","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"Carbon has an atomic number of 6 and a standard atomic weight of 12.011."},
  {"id":"s4_010","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The United States Declaration of Independence was signed on August 2, 1776, though dated July 4."},
  {"id":"s4_011","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"A human red blood cell has a lifespan of approximately 120 days before being removed by the spleen."},
  {"id":"s4_012","content_type":"Factual","source_type":"encyclopedia","language_structure":"declarative_third_person","text":"The Amazon River discharges approximately 209,000 cubic meters of water per second into the Atlantic Ocean."},

  {"id":"s2_001","content_type":"Abstract","source_type":"philosophy","language_structure":"dense_nominal_nested","text":"The ontological argument posits that the concept of a maximally great being entails its necessary existence in all possible worlds, since a being that exists necessarily is greater than one that exists contingently."},
  {"id":"s2_002","content_type":"Abstract","source_type":"philosophy","language_structure":"dense_nominal_nested","text":"The hard problem of consciousness asks why and how physical processes in the brain give rise to subjective qualitative experience — the felt sense of what it is like to see red or feel pain — which appears irreducible to functional description."},
  {"id":"s2_003","content_type":"Abstract","source_type":"philosophy","language_structure":"dense_nominal_nested","text":"Zeno's paradox of Achilles and the tortoise demonstrates that motion, when decomposed into infinite subdivisions of space, appears logically impossible despite being empirically continuous, suggesting a gap between mathematical description and physical reality."},
  {"id":"s2_004","content_type":"Abstract","source_type":"philosophy","language_structure":"conditional_argument","text":"If determinism is true and all events are causally necessitated by prior states of the universe, then moral responsibility — which presupposes the ability to have done otherwise — is incoherent without a compatibilist account of freedom."},
  {"id":"s2_005","content_type":"Abstract","source_type":"philosophy","language_structure":"dense_nominal_nested","text":"The Gettier problem challenges the classical analysis of knowledge as justified true belief by constructing cases where a belief is both justified and true yet intuitively does not constitute knowledge, suggesting the tripartite analysis is insufficient."},
  {"id":"s2_006","content_type":"Abstract","source_type":"philosophy","language_structure":"dense_nominal_nested","text":"Wittgenstein's private language argument suggests that a language intelligible only to its single user is impossible, because meaning requires the possibility of public checking — without a criterion for correct application, there is no distinction between using a sign correctly and merely thinking one is using it correctly."},
  {"id":"s2_007","content_type":"Abstract","source_type":"mathematics","language_structure":"formal_definition","text":"In category theory, a functor is a structure-preserving map between categories that assigns to each object and morphism in the source category a corresponding object and morphism in the target category, in a way that preserves identity morphisms and composition."},
  {"id":"s2_008","content_type":"Abstract","source_type":"mathematics","language_structure":"paradox_statement","text":"The Banach-Tarski paradox demonstrates that it is theoretically possible to decompose a three-dimensional ball into a finite number of non-measurable pieces and reassemble them, using only rigid motions, into two balls identical to the original."},
  {"id":"s2_009","content_type":"Abstract","source_type":"physics","language_structure":"conditional_argument","text":"The principle of superposition holds that a quantum system exists in all possible states simultaneously until a measurement collapses its wave function to a single eigenstate, suggesting that measurement itself — and the role of the observer — is constitutive of physical reality."},
  {"id":"s2_010","content_type":"Abstract","source_type":"philosophy","language_structure":"hypothetical_framing","text":"Rawls's veil of ignorance asks us to design principles of justice without knowing our position in society — our class, race, sex, or natural talents — arguing that this epistemic constraint produces maximally fair distributive outcomes."},
  {"id":"s2_011","content_type":"Abstract","source_type":"ai_philosophy","language_structure":"problem_statement","text":"The frame problem in artificial intelligence asks how a reasoning system can efficiently represent what does not change when an action is taken, without having to explicitly enumerate all the facts that remain unchanged after each event."},
  {"id":"s2_012","content_type":"Abstract","source_type":"philosophy","language_structure":"dialectical_structure","text":"Hegelian dialectic describes intellectual and historical progress as a process in which a thesis encounters its antithesis, generating a contradiction that is resolved through a synthesis that preserves, negates, and elevates both prior positions."},

  {"id":"s5_001","content_type":"Spatial","source_type":"navigation_instructions","language_structure":"directional_relational","text":"To reach the library from here, head north on Main Street for two blocks, turn left at the traffic light, pass the post office on your right, and the library will be the red brick building at the corner."},
  {"id":"s5_002","content_type":"Spatial","source_type":"architectural_description","language_structure":"positional_relational","text":"The kitchen is directly north of the living room, separated by a load-bearing wall. The master bedroom is located above the kitchen and shares its east wall with the second bathroom."},
  {"id":"s5_003","content_type":"Spatial","source_type":"astronomy","language_structure":"positional_relational","text":"Jupiter is the fifth planet from the Sun, located between the asteroid belt and Saturn at an average distance of 778 million kilometers from the Sun."},
  {"id":"s5_004","content_type":"Spatial","source_type":"installation_instructions","language_structure":"positional_imperative","text":"Place the router in the center of the home, elevated off the floor, away from metal objects and walls. Position it at least two meters from microwave ovens and cordless phones."},
  {"id":"s5_005","content_type":"Spatial","source_type":"geography","language_structure":"positional_relational","text":"The island lies 340 kilometers southeast of the mainland coast, accessible only by ferry from the port at the southern tip of the peninsula."},
  {"id":"s5_006","content_type":"Spatial","source_type":"landscape_description","language_structure":"positional_relational","text":"From the summit, the valley spreads out to the north. The river enters from the east, curves through the center, and exits at the western edge near the old mill."},
  {"id":"s5_007","content_type":"Spatial","source_type":"architectural_description","language_structure":"positional_relational","text":"The table is positioned against the north wall. The window is to the east, approximately one meter above the floor. The door is in the southwest corner, hinged on the right side."},
  {"id":"s5_008","content_type":"Spatial","source_type":"navigation_instructions","language_structure":"directional_relational","text":"Exit the highway at junction 14, continue straight for 800 meters, then take the second right after the roundabout. The facility is on the left, immediately after the railway crossing."},
  {"id":"s5_009","content_type":"Spatial","source_type":"neuroscience","language_structure":"anatomical_positional","text":"The prefrontal cortex is located in the anterior portion of the frontal lobe, directly behind the forehead, anterior to the motor cortex and superior to the orbitofrontal cortex."},
  {"id":"s5_010","content_type":"Spatial","source_type":"seating_instructions","language_structure":"positional_relational","text":"Seat A1 is in the front row, leftmost position from the audience's perspective. The emergency exits are located at the rear left and rear right of the auditorium."},
  {"id":"s5_011","content_type":"Spatial","source_type":"packing_instructions","language_structure":"positional_imperative","text":"Stack the boxes with the heaviest at the bottom and lightest at the top. The fragile items go in the center column, flanked on both sides by the padded boxes."},
  {"id":"s5_012","content_type":"Spatial","source_type":"neuroscience","language_structure":"anatomical_positional","text":"The neuron's cell body sits in cortical layer III, with its axon projecting downward through layers V and VI into the white matter below."},

  {"id":"b2_001","content_type":"Reward","source_type":"product_review","language_structure":"positive_superlative","text":"This is the best purchase I have made in the last decade. Absolutely transformed my morning routine. Cannot recommend it highly enough. Five stars."},
  {"id":"b2_002","content_type":"Reward","source_type":"career_announcement","language_structure":"achievement_outcome","text":"I just got promoted to senior engineer. Three years of late nights, hard conversations, and difficult projects, and it finally paid off. Best day of my career."},
  {"id":"b2_003","content_type":"Reward","source_type":"product_review","language_structure":"positive_superlative","text":"Outstanding product. Exceeded every expectation I had. Already ordered a second one as a gift for my parents. Perfect in every respect."},
  {"id":"b2_004","content_type":"Reward","source_type":"restaurant_review","language_structure":"positive_superlative","text":"The restaurant was perfect in every respect — service, food, ambiance. We will return for every anniversary from now on. Nothing to improve."},
  {"id":"b2_005","content_type":"Reward","source_type":"personal_achievement","language_structure":"achievement_outcome","text":"I finished my first marathon today. Four hours and two minutes. I never thought I could do it. I am so unbelievably proud of myself."},
  {"id":"b2_006","content_type":"Reward","source_type":"startup_announcement","language_structure":"achievement_outcome","text":"The team just closed a $2M Series A. We spent 8 months pitching 60 investors. This changes everything. We are going to build something real."},
  {"id":"b2_007","content_type":"Reward","source_type":"academic_achievement","language_structure":"second_person_praise","text":"My student passed their dissertation defense today. Five years of work. Watching them walk out of that room beaming was the best moment of my teaching career."},
  {"id":"b2_008","content_type":"Reward","source_type":"workplace_praise","language_structure":"second_person_praise","text":"Your code review comment saved us from a production incident that would have cost the company $200,000. You are a hero and I am telling everyone."},
  {"id":"b2_009","content_type":"Reward","source_type":"book_review","language_structure":"positive_superlative","text":"This book is the most important thing I have read in years. Completely changed how I think about habit formation and motivation. Read it immediately."},
  {"id":"b2_010","content_type":"Reward","source_type":"product_milestone","language_structure":"achievement_outcome","text":"We hit 1 million users today. One year ago we had three. I need a moment to process this. Thank you to everyone who believed in us."},
  {"id":"b2_011","content_type":"Reward","source_type":"personal_announcement","language_structure":"positive_superlative","text":"She said yes. I cannot stop smiling. Everything is perfect. I am the luckiest person alive right now."},
  {"id":"b2_012","content_type":"Reward","source_type":"career_announcement","language_structure":"achievement_outcome","text":"After 18 months of job hunting, three final-round rejections, and one rescinded offer, I finally signed an offer letter today. I did not give up."},

  {"id":"m2_001","content_type":"ImageVisual","source_type":"image_caption","language_structure":"sensory_descriptive","text":"A golden retriever puppy sits in a sunlit meadow, head tilted, ears raised, surrounded by tall green grass and wildflowers in soft afternoon light."},
  {"id":"m2_002","content_type":"ImageVisual","source_type":"image_caption","language_structure":"sensory_descriptive","text":"Aerial photograph of a dense urban grid at night: orange sodium streetlights form a rectilinear matrix extending to the horizon, interrupted by the dark ribbon of a river."},
  {"id":"m2_003","content_type":"ImageVisual","source_type":"image_caption","language_structure":"close_up_detail","text":"Close-up of a human eye: iris a deep hazel with radiating amber flecks, pupil fully dilated, fine blood vessels visible in the white sclera."},
  {"id":"m2_004","content_type":"ImageVisual","source_type":"image_caption","language_structure":"sensory_descriptive","text":"A thunderstorm over a flat agricultural plain: anvil-shaped cumulonimbus cloud extending from mid-frame to the top edge, lit internally by lightning, base casting a greenish shadow across the fields."},
  {"id":"m2_005","content_type":"ImageVisual","source_type":"image_caption","language_structure":"close_up_detail","text":"Black and white photograph of an elderly woman's hands folded in her lap, skin deeply wrinkled, wearing a simple gold band on her left ring finger."},
  {"id":"m2_006","content_type":"ImageVisual","source_type":"image_caption","language_structure":"abstract_visual","text":"Abstract composition: overlapping translucent rectangles in cyan, magenta, and yellow on a white background, edges soft, center point a muddy brown where all three overlap."},
  {"id":"m2_007","content_type":"ImageVisual","source_type":"satellite_image","language_structure":"aerial_descriptive","text":"Satellite image of the Amazon delta: dark green canopy interrupted by brown braided river channels, a plume of sediment-laden water extending into the blue Atlantic Ocean."},
  {"id":"m2_008","content_type":"ImageVisual","source_type":"image_caption","language_structure":"sensory_descriptive","text":"A crowded subway car at rush hour: standing passengers gripping overhead bars, faces uniformly downward toward phones, fluorescent lighting casting flat shadows on steel poles."},
  {"id":"m2_009","content_type":"ImageVisual","source_type":"image_caption","language_structure":"close_up_detail","text":"A single burning candle on a wooden table: flame steady, wax pooled at the base, soft warm light casting a circular glow in an otherwise completely dark room."},
  {"id":"m2_010","content_type":"ImageVisual","source_type":"scientific_image","language_structure":"technical_descriptive","text":"Scanning electron microscope image of a butterfly wing scale at 500x magnification: intricate lattice of ridges forming iridescent structural color through light diffraction, no pigment present."},
  {"id":"m2_011","content_type":"ImageVisual","source_type":"image_caption","language_structure":"sensory_descriptive","text":"A child's drawing in crayon: a house with smoke curling from the chimney, a yellow circle sun in the upper right corner, two stick figures of unequal height holding hands."},
  {"id":"m2_012","content_type":"ImageVisual","source_type":"image_caption","language_structure":"aerial_descriptive","text":"Wide-angle photograph of the Milky Way over a desert landscape: star-dense galactic core rising from the right horizon, foreground of dark red sandstone formations silhouetted against the sky."},

  {"id":"m3_001","content_type":"AudioText","source_type":"audio_caption","language_structure":"acoustic_descriptive","text":"A recording of rain on a tin roof: irregular staccato impacts, occasional heavier drops producing deeper resonance, continuous background white noise of sustained precipitation."},
  {"id":"m3_002","content_type":"AudioText","source_type":"audio_caption","language_structure":"spectral_descriptive","text":"The opening chord of a pipe organ in a cathedral: fundamental at 64 Hz, harmonics stacked to 2 kHz, reverb decay lasting approximately 4 seconds in the stone space."},
  {"id":"m3_003","content_type":"AudioText","source_type":"audio_caption","language_structure":"acoustic_descriptive","text":"Ambient soundscape of a busy café: espresso machine hissing, ceramic cups on saucers, overlapping conversations in multiple languages, jazz piano barely audible beneath the noise floor."},
  {"id":"m3_004","content_type":"AudioText","source_type":"music_description","language_structure":"spectral_descriptive","text":"A distorted electric guitar playing a minor pentatonic phrase at 120 BPM: bends on the third and fifth scale degrees, palm-muted downstrokes, overdrive pedal with mid-frequency emphasis around 2 kHz."},
  {"id":"m3_005","content_type":"AudioText","source_type":"audio_caption","language_structure":"acoustic_descriptive","text":"Emergency siren in a city street: alternating two-tone wail at 800 Hz and 1000 Hz, pronounced Doppler shift as vehicle approaches and recedes, echo from surrounding buildings."},
  {"id":"m3_006","content_type":"AudioText","source_type":"medical_audio","language_structure":"spectral_descriptive","text":"A human heartbeat recorded via stethoscope: first sound low-frequency closure of mitral and tricuspid valves, brief silence, then second sound, regular rhythm at 68 BPM."},
  {"id":"m3_007","content_type":"AudioText","source_type":"crowd_recording","language_structure":"acoustic_descriptive","text":"A crowd in a football stadium reacting to a goal: initial sharp collective intake of breath, then an explosive roar that peaks at 110 dB and sustains for 8 seconds before fading."},
  {"id":"m3_008","content_type":"AudioText","source_type":"nature_recording","language_structure":"spectral_descriptive","text":"Wind through pine trees on a mountain ridge: low-frequency rumble below 100 Hz underlies mid-frequency susurration, irregular gusts producing short-duration spectral peaks above 1 kHz."},
  {"id":"m3_009","content_type":"AudioText","source_type":"audio_caption","language_structure":"acoustic_descriptive","text":"A vinyl record skipping: a 1.2-second loop of a piano phrase repeating every rotation, accompanied by surface noise crackle and the soft thump of the stylus."},
  {"id":"m3_010","content_type":"AudioText","source_type":"nature_recording","language_structure":"spectral_descriptive","text":"Binaural recording of ocean waves: low-frequency surge modulating mid-frequency rushing between 200 and 800 Hz, spatial panning between ears shifts as waves approach obliquely."},
  {"id":"m3_011","content_type":"AudioText","source_type":"medical_audio","language_structure":"acoustic_descriptive","text":"A baby crying: fundamental pitch at 400 Hz, rich harmonics to 4 kHz, irregular pitch contour, short inhalation phases between sustained cry bursts of approximately 2 seconds."},
  {"id":"m3_012","content_type":"AudioText","source_type":"ambient_recording","language_structure":"acoustic_descriptive","text":"3 AM apartment building: near-silence at -45 dBFS, interrupted by irregular footsteps from above, distant traffic, pipe expansion clicks, refrigerator compressor hum at 60 Hz."},

  {"id":"m4_001","content_type":"Multimodal","source_type":"news_broadcast","language_structure":"simultaneous_av","text":"A news anchor speaks directly to camera while behind her a live feed shows a building fire; her voice is steady and clipped, the fire crackles in the background audio mix."},
  {"id":"m4_002","content_type":"Multimodal","source_type":"documentary","language_structure":"simultaneous_av","text":"A documentary narrator describes the migration of monarch butterflies while slow-motion footage shows millions of orange wings against a blue sky; the score is a sustained string chord."},
  {"id":"m4_003","content_type":"Multimodal","source_type":"cooking_video","language_structure":"simultaneous_av","text":"A cooking video: the host explains the Maillard reaction while browning onions; the visual shows translucent becoming golden, the audio captures the sizzle and the host's measured voice."},
  {"id":"m4_004","content_type":"Multimodal","source_type":"live_concert","language_structure":"simultaneous_av","text":"A live concert: guitarist plays the opening riff, spotlight narrows on the fretboard, crowd noise drops to near-silence as the chord rings, then erupts."},
  {"id":"m4_005","content_type":"Multimodal","source_type":"sign_language","language_structure":"parallel_encoding","text":"A sign language interpreter stands in the lower-right corner of the frame while a speech is delivered; both the spoken audio and the interpreter's hand movements convey the same words through different channels."},
  {"id":"m4_006","content_type":"Multimodal","source_type":"silent_film","language_structure":"compensatory_av","text":"A silent film with live piano accompaniment: black and white images of a chase sequence, the pianist accelerating tempo to match the action, no dialogue, only gesture and music."},
  {"id":"m4_007","content_type":"Multimodal","source_type":"video_podcast","language_structure":"simultaneous_av","text":"A podcast with video: two people at a table, one talking, one listening with visible microexpressions; the audio quality is studio-clean but the visual setting is intentionally informal."},
  {"id":"m4_008","content_type":"Multimodal","source_type":"security_camera","language_structure":"discordant_av","text":"Security camera footage of an empty parking lot at 2 AM with a motion-triggered alarm: the visual is static, grainy, green-tinted; the audio is a sudden high-pitched beeping."},
  {"id":"m4_009","content_type":"Multimodal","source_type":"medical_lecture","language_structure":"parallel_encoding","text":"A surgeon explains a procedure using a 3D anatomical model, pointing to structures while speaking; both visual annotation and spoken narration identify the same anatomical landmarks simultaneously."},
  {"id":"m4_010","content_type":"Multimodal","source_type":"read_aloud","language_structure":"parallel_encoding","text":"A child reads aloud from a picture book: the visual shows the illustrated page, the audio captures the halting pronunciation and rising intonation of a beginning reader."},
  {"id":"m4_011","content_type":"Multimodal","source_type":"nature_documentary","language_structure":"simultaneous_av","text":"A nature documentary: a polar bear and her cubs on sea ice, the narrator's voice quiet and measured, the soundscape dominated by wind and the low creaking of shifting ice."},
  {"id":"m4_012","content_type":"Multimodal","source_type":"music_video","language_structure":"synchronized_av","text":"A music video: the singer's face in extreme close-up while she delivers the final verse, the camera pulling back in sync with a key change, the visual and audio reaching a climax together."},

  {"id":"b3_cp_headline","content_type":"ThreatSafety","source_type":"contrastive_pair","language_structure":"headline_format","text":"Bridge collapses on major highway; 12 vehicles involved, rescue operation underway."},
  {"id":"b3_cp_narrative","content_type":"ThreatSafety","source_type":"contrastive_pair","language_structure":"first_person_narrative","text":"I was the third car behind the truck when the bridge gave way. The sound was wrong first — a grinding, then a crack — and then the road in front of me wasn't there anymore. I heard someone screaming. I think it was me."},

  {"id":"s1_cp_chrono","content_type":"Narrative","source_type":"contrastive_pair","language_structure":"chronological_causal","text":"She applied for the visa in January. The rejection arrived in March. She appealed in April and submitted additional documents in June. In September, the second rejection arrived. In October, she found a different route."},
  {"id":"s1_cp_bullets","content_type":"Narrative","source_type":"contrastive_pair","language_structure":"bullet_summary","text":"Visa timeline: Applied Jan. Rejected Mar. Appealed Apr. Docs submitted Jun. Second rejection Sep. Alternative route found Oct."},

  {"id":"s4_cp_active","content_type":"Factual","source_type":"contrastive_pair","language_structure":"active_declarative","text":"Einstein published the special theory of relativity in 1905, fundamentally changing our understanding of space, time, and energy."},
  {"id":"s4_cp_passive","content_type":"Factual","source_type":"contrastive_pair","language_structure":"passive_declarative","text":"The special theory of relativity was published in 1905 and has since been recognized as having fundamentally changed our understanding of space, time, and energy."}
]
```

- [ ] **Step 2: Verify JSON parses**

```bash
python3 -c "import json; d=json.load(open('experiments/corpus/stimuli.json')); print(len(d), 'stimuli')"
```
Expected: `150 stimuli`

- [ ] **Step 3: Commit**

```bash
git add experiments/corpus/stimuli.json
git commit -m "feat: add 150-stimulus corpus across 13 content types"
```

---

### Task 3: Core types + corpus loading (TDD)

**Files:**
- Modify: `experiments/sweep/src/main.rs`

- [ ] **Step 1: Write the failing tests**

```rust
// experiments/sweep/src/main.rs
use std::{collections::HashMap, fs, path::Path};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Clone)]
struct Stimulus {
    id: String,
    content_type: String,
    source_type: String,
    language_structure: String,
    text: String,
}

fn load_corpus(path: &str) -> Vec<Stimulus> {
    todo!()
}

#[derive(Debug, Deserialize, Clone)]
struct RegionStat {
    mean: f32,
    rel_activation: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct GlobalStat {
    global_mean: f32,
    global_max: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct PredictResp {
    region_stats: HashMap<String, RegionStat>,
    global_stats: GlobalStat,
    vertex_acts: Vec<f32>,
    demo_mode: bool,
}

fn parse_predict_resp(json: &str) -> Result<PredictResp, serde_json::Error> {
    todo!()
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn load_corpus_returns_correct_count() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, r#"[
            {{"id":"x1","content_type":"Factual","source_type":"enc","language_structure":"ls","text":"hello"}},
            {{"id":"x2","content_type":"Narrative","source_type":"enc","language_structure":"ls","text":"world"}}
        ]"#).unwrap();
        let stimuli = load_corpus(f.path().to_str().unwrap());
        assert_eq!(stimuli.len(), 2);
        assert_eq!(stimuli[0].id, "x1");
        assert_eq!(stimuli[1].content_type, "Narrative");
    }

    #[test]
    fn parse_predict_resp_parses_correctly() {
        let json = r#"{
            "region_stats": {
                "language": {"mean": 0.5, "std": 0.1, "rel_activation": 1.2, "peak": 0.9, "n_vertices": 3700},
                "prefrontal": {"mean": 0.3, "std": 0.08, "rel_activation": 0.4, "peak": 0.6, "n_vertices": 3500}
            },
            "global_stats": {"global_mean": 0.2, "global_std": 0.25, "global_min": -1.0, "global_max": 0.95},
            "vertex_acts": [0.1, 0.2, 0.3],
            "temporal_acts": [],
            "seq_len": 16,
            "modality": "text",
            "elapsed_ms": 42.0,
            "demo_mode": false
        }"#;
        let resp = parse_predict_resp(json).unwrap();
        assert!((resp.global_stats.global_mean - 0.2).abs() < 1e-5);
        assert!((resp.region_stats["language"].rel_activation - 1.2).abs() < 1e-5);
        assert_eq!(resp.demo_mode, false);
        assert_eq!(resp.vertex_acts.len(), 3);
    }
}
```

Add `tempfile = "3"` to `experiments/sweep/Cargo.toml` under `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p sweep 2>&1 | tail -5
```
Expected: FAIL — `not yet implemented`

- [ ] **Step 3: Implement `load_corpus` and `parse_predict_resp`**

Replace the `todo!()` bodies:

```rust
fn load_corpus(path: &str) -> Vec<Stimulus> {
    let data = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read corpus at {path}: {e}"));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse corpus JSON: {e}"))
}

fn parse_predict_resp(json: &str) -> Result<PredictResp, serde_json::Error> {
    serde_json::from_str(json)
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p sweep
```
Expected: 3 tests PASS (placeholder + 2 new)

- [ ] **Step 5: Commit**

```bash
git add experiments/sweep/src/main.rs experiments/sweep/Cargo.toml
git commit -m "feat: add Stimulus and PredictResp types with corpus loading and JSON parsing"
```

---

### Task 4: HTTP client — health check + predict (TDD)

**Files:**
- Modify: `experiments/sweep/src/main.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    #[test]
    fn check_health_returns_false_when_server_not_running() {
        // port 19991 is virtually guaranteed to be unused
        assert!(!check_health("http://localhost:19991"));
    }

    #[test]
    fn predict_one_returns_err_when_server_not_running() {
        let result = predict_one("http://localhost:19991", "hello world");
        assert!(result.is_err());
    }
```

Add stubs before `main()`:

```rust
fn check_health(_base_url: &str) -> bool { todo!() }
fn predict_one(_base_url: &str, _text: &str) -> Result<PredictResp, String> { todo!() }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p sweep check_health 2>&1 | tail -5
```
Expected: FAIL — `not yet implemented`

- [ ] **Step 3: Implement `check_health` and `predict_one`**

```rust
fn check_health(base_url: &str) -> bool {
    ureq::get(&format!("{base_url}/health"))
        .call()
        .map(|r| r.status() == 200)
        .unwrap_or(false)
}

fn predict_one(base_url: &str, text: &str) -> Result<PredictResp, String> {
    let body = serde_json::json!({ "text": text, "seq_len": 16 });
    let raw = ureq::post(&format!("{base_url}/api/predict"))
        .send_json(&body)
        .map_err(|e| e.to_string())?
        .into_string()
        .map_err(|e| e.to_string())?;
    parse_predict_resp(&raw).map_err(|e| e.to_string())
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p sweep
```
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/sweep/src/main.rs
git commit -m "feat: add check_health and predict_one HTTP functions"
```

---

### Task 5: Sweep record + ranking + output (TDD)

**Files:**
- Modify: `experiments/sweep/src/main.rs`

- [ ] **Step 1: Write the failing tests**

Add these types and stubs before `main()`:

```rust
#[derive(Debug, Clone, Serialize)]
struct SweepRecord {
    id: String,
    content_type: String,
    source_type: String,
    language_structure: String,
    demo_mode: bool,
    global_mean: f32,
    global_max: f32,
    visual_rel: f32,
    auditory_rel: f32,
    language_rel: f32,
    prefrontal_rel: f32,
    motor_rel: f32,
    parietal_rel: f32,
    vertex_acts: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct HeatmapData {
    content_types: Vec<String>,
    regions: Vec<String>,
    matrix: Vec<Vec<f32>>,
}

fn make_record(stimulus: &Stimulus, resp: &PredictResp) -> SweepRecord { todo!() }
fn rank_results(records: &[SweepRecord]) -> Vec<SweepRecord> { todo!() }
fn build_heatmap(records: &[SweepRecord]) -> HeatmapData { todo!() }
fn write_ranked_csv(ranked: &[SweepRecord], path: &str) { todo!() }
fn write_raw_json(records: &[SweepRecord], path: &str) { todo!() }
fn write_heatmap_json(heatmap: &HeatmapData, path: &str) { todo!() }
```

Add tests to `tests` module:

```rust
    fn make_test_record(id: &str, ct: &str, global_mean: f32, lang_rel: f32) -> SweepRecord {
        SweepRecord {
            id: id.to_string(),
            content_type: ct.to_string(),
            source_type: "test".to_string(),
            language_structure: "test".to_string(),
            demo_mode: true,
            global_mean,
            global_max: global_mean + 0.1,
            visual_rel: 0.0,
            auditory_rel: 0.0,
            language_rel: lang_rel,
            prefrontal_rel: 0.0,
            motor_rel: 0.0,
            parietal_rel: 0.0,
            vertex_acts: vec![],
        }
    }

    #[test]
    fn rank_results_sorts_by_global_mean_descending() {
        let records = vec![
            make_test_record("a", "Factual", 0.1, 0.0),
            make_test_record("b", "ThreatSafety", 0.9, 0.0),
            make_test_record("c", "Narrative", 0.5, 0.0),
        ];
        let ranked = rank_results(&records);
        assert_eq!(ranked[0].id, "b");
        assert_eq!(ranked[1].id, "c");
        assert_eq!(ranked[2].id, "a");
    }

    #[test]
    fn build_heatmap_produces_correct_dimensions() {
        let records = vec![
            make_test_record("a", "Factual", 0.1, 0.3),
            make_test_record("b", "Narrative", 0.5, 0.8),
            make_test_record("c", "Factual", 0.2, 0.4),
        ];
        let heatmap = build_heatmap(&records);
        // 2 unique content types
        assert_eq!(heatmap.content_types.len(), 2);
        assert_eq!(heatmap.regions.len(), 6);
        assert_eq!(heatmap.matrix.len(), 2);
        assert_eq!(heatmap.matrix[0].len(), 6);
        // Factual language_rel mean = (0.3 + 0.4) / 2 = 0.35
        let factual_idx = heatmap.content_types.iter().position(|x| x == "Factual").unwrap();
        let lang_idx = heatmap.regions.iter().position(|x| x == "language").unwrap();
        assert!((heatmap.matrix[factual_idx][lang_idx] - 0.35).abs() < 1e-4);
    }

    #[test]
    fn write_ranked_csv_produces_correct_header() {
        let records = vec![make_test_record("x1", "Factual", 0.5, 0.2)];
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();
        write_ranked_csv(&records, path);
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.starts_with("rank,id,content_type,source_type,language_structure,demo_mode,global_mean,global_max,visual_rel,auditory_rel,language_rel,prefrontal_rel,motor_rel,parietal_rel"));
        assert!(content.contains("1,x1,Factual"));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p sweep rank 2>&1 | tail -5
```
Expected: FAIL — `not yet implemented`

- [ ] **Step 3: Implement all six functions**

```rust
fn make_record(stimulus: &Stimulus, resp: &PredictResp) -> SweepRecord {
    let region = |name: &str| -> f32 {
        resp.region_stats.get(name).map(|r| r.rel_activation).unwrap_or(0.0)
    };
    SweepRecord {
        id: stimulus.id.clone(),
        content_type: stimulus.content_type.clone(),
        source_type: stimulus.source_type.clone(),
        language_structure: stimulus.language_structure.clone(),
        demo_mode: resp.demo_mode,
        global_mean: resp.global_stats.global_mean,
        global_max: resp.global_stats.global_max,
        visual_rel:     region("visual"),
        auditory_rel:   region("auditory"),
        language_rel:   region("language"),
        prefrontal_rel: region("prefrontal"),
        motor_rel:      region("motor"),
        parietal_rel:   region("parietal"),
        vertex_acts: resp.vertex_acts.clone(),
    }
}

fn rank_results(records: &[SweepRecord]) -> Vec<SweepRecord> {
    let mut sorted = records.to_vec();
    sorted.sort_by(|a, b| b.global_mean.partial_cmp(&a.global_mean).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

fn build_heatmap(records: &[SweepRecord]) -> HeatmapData {
    let regions = vec!["visual", "auditory", "language", "prefrontal", "motor", "parietal"];
    let mut types: Vec<String> = records.iter().map(|r| r.content_type.clone()).collect();
    types.sort();
    types.dedup();

    let matrix = types.iter().map(|ct| {
        let group: Vec<&SweepRecord> = records.iter().filter(|r| &r.content_type == ct).collect();
        let n = group.len() as f32;
        regions.iter().map(|region| {
            let sum: f32 = group.iter().map(|r| match *region {
                "visual"     => r.visual_rel,
                "auditory"   => r.auditory_rel,
                "language"   => r.language_rel,
                "prefrontal" => r.prefrontal_rel,
                "motor"      => r.motor_rel,
                "parietal"   => r.parietal_rel,
                _            => 0.0,
            }).sum();
            if n > 0.0 { sum / n } else { 0.0 }
        }).collect()
    }).collect();

    HeatmapData {
        content_types: types,
        regions: regions.iter().map(|s| s.to_string()).collect(),
        matrix,
    }
}

fn write_ranked_csv(ranked: &[SweepRecord], path: &str) {
    let header = "rank,id,content_type,source_type,language_structure,demo_mode,global_mean,global_max,visual_rel,auditory_rel,language_rel,prefrontal_rel,motor_rel,parietal_rel\n";
    let rows: String = ranked.iter().enumerate().map(|(i, r)| {
        format!("{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            i + 1, r.id, r.content_type, r.source_type, r.language_structure,
            r.demo_mode, r.global_mean, r.global_max,
            r.visual_rel, r.auditory_rel, r.language_rel,
            r.prefrontal_rel, r.motor_rel, r.parietal_rel)
    }).collect();
    fs::write(path, format!("{header}{rows}"))
        .unwrap_or_else(|e| panic!("Failed to write CSV to {path}: {e}"));
}

fn write_raw_json(records: &[SweepRecord], path: &str) {
    let json = serde_json::to_string_pretty(records)
        .unwrap_or_else(|e| panic!("Serialization failed: {e}"));
    fs::write(path, json).unwrap_or_else(|e| panic!("Failed to write JSON to {path}: {e}"));
}

fn write_heatmap_json(heatmap: &HeatmapData, path: &str) {
    let json = serde_json::to_string_pretty(heatmap)
        .unwrap_or_else(|e| panic!("Serialization failed: {e}"));
    fs::write(path, json).unwrap_or_else(|e| panic!("Failed to write heatmap to {path}: {e}"));
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p sweep
```
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add experiments/sweep/src/main.rs
git commit -m "feat: add SweepRecord, ranking, heatmap, and CSV/JSON output functions"
```

---

### Task 6: Main function — sweep loop + progress + demo warning

**Files:**
- Modify: `experiments/sweep/src/main.rs`

- [ ] **Step 1: Replace `fn main()` with the full sweep loop**

```rust
fn main() {
    let base_url = "http://localhost:8081";
    let corpus_path = "experiments/corpus/stimuli.json";
    let results_dir = "results";

    fs::create_dir_all(results_dir)
        .unwrap_or_else(|e| panic!("Cannot create results/: {e}"));

    println!("Checking tribe-server at {base_url}...");
    if !check_health(base_url) {
        eprintln!("ERROR: tribe-server is not running at {base_url}.");
        eprintln!("Start it with: cd ../tribe-playground && cargo run --release -p tribe-server");
        std::process::exit(1);
    }
    println!("Server online.");

    let stimuli = load_corpus(corpus_path);
    println!("Loaded {} stimuli from {corpus_path}", stimuli.len());

    let mut records: Vec<SweepRecord> = Vec::with_capacity(stimuli.len());
    let mut errors = 0usize;

    for (i, stimulus) in stimuli.iter().enumerate() {
        match predict_one(base_url, &stimulus.text) {
            Ok(resp) => {
                let rec = make_record(stimulus, &resp);
                println!(
                    "[{:>3}/{}] {} | global_mean={:.4} lang_rel={:.3} demo={}",
                    i + 1, stimuli.len(), rec.id,
                    rec.global_mean, rec.language_rel, rec.demo_mode
                );
                records.push(rec);
            }
            Err(e) => {
                eprintln!("[{:>3}/{}] ERROR {}: {}", i + 1, stimuli.len(), stimulus.id, e);
                errors += 1;
            }
        }
    }

    if records.is_empty() {
        eprintln!("No successful predictions. Aborting.");
        std::process::exit(1);
    }

    println!("\nWriting results ({} records, {} errors)...", records.len(), errors);

    let ranked = rank_results(&records);
    write_ranked_csv(&ranked, &format!("{results_dir}/sweep_ranked.csv"));
    println!("  results/sweep_ranked.csv");

    write_raw_json(&records, &format!("{results_dir}/sweep_results.json"));
    println!("  results/sweep_results.json");

    let heatmap = build_heatmap(&records);
    write_heatmap_json(&heatmap, &format!("{results_dir}/region_heatmap.json"));
    println!("  results/region_heatmap.json");

    println!("\n=== TOP 10 BY GLOBAL ACTIVATION ===");
    println!("{:<5} {:<20} {:<40} {}", "Rank", "ContentType", "ID", "GlobalMean");
    for (i, r) in ranked.iter().take(10).enumerate() {
        println!("{:<5} {:<20} {:<40} {:.4}", i + 1, r.content_type, r.id, r.global_mean);
    }

    let demo_count = records.iter().filter(|r| r.demo_mode).count();
    if demo_count > 0 {
        println!(
            "\n⚠  {}/{} results are in demo mode — re-run after loading real weights for semantically valid results.",
            demo_count, records.len()
        );
        println!("   Run: bash scripts/download_weights.sh");
    }
}
```

- [ ] **Step 2: Build to verify no compile errors**

```bash
cargo build -p sweep 2>&1 | tail -5
```
Expected: `Finished` with no errors

- [ ] **Step 3: Run tests to verify still passing**

```bash
cargo test -p sweep
```
Expected: 8 tests PASS

- [ ] **Step 4: Commit**

```bash
git add experiments/sweep/src/main.rs
git commit -m "feat: add sweep main loop with progress display and demo mode warning"
```

---

### Task 7: Download script + results .gitignore

**Files:**
- Create: `scripts/download_weights.sh`
- Create: `results/.gitignore`

- [ ] **Step 1: Create `scripts/download_weights.sh`**

```bash
mkdir -p scripts
```

```bash
#!/usr/bin/env bash
# Download TRIBE v2 model weights via the tribe-server download endpoints.
# tribe-server must be running at localhost:8081 before running this script.
# best.safetensors (FmriEncoder) is NOT downloaded via the API — it requires
# running: cd ../tribe-playground && python3 convert_ckpt.py

set -euo pipefail

BASE="http://localhost:8081"

check_server() {
    curl -sf "$BASE/health" > /dev/null || {
        echo "ERROR: tribe-server not running at $BASE"
        echo "Start it with: cd ../tribe-playground && cargo run --release -p tribe-server"
        exit 1
    }
}

start_download() {
    local model="$1"
    echo "Starting download: $model"
    local job_id
    job_id=$(curl -sf -X POST "$BASE/api/download/$model" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
    echo "  Job ID: $job_id"
    echo "  Poll: curl $BASE/api/jobs/$job_id"
    echo "$job_id"
}

poll_until_done() {
    local job_id="$1"
    local model="$2"
    echo "Waiting for $model (job $job_id)..."
    while true; do
        local status
        status=$(curl -sf "$BASE/api/jobs/$job_id" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
        echo "  [$model] status: $status"
        if [ "$status" = "done" ]; then break; fi
        if [ "$status" = "failed" ]; then echo "ERROR: $model download failed"; exit 1; fi
        sleep 10
    done
    echo "  $model: done"
}

check_server

echo "=== Step 1: Convert checkpoint to best.safetensors ==="
echo "Run manually if not already done:"
echo "  cd ../tribe-playground && python3 convert_ckpt.py"
echo ""

echo "=== Step 2: Download LLaMA text encoder (~6GB) ==="
llama_job=$(start_download "llama")
poll_until_done "$llama_job" "llama"

echo "=== Step 3: Download Wav2Vec2Bert audio encoder ==="
w2v_job=$(start_download "wav2vec")
poll_until_done "$w2v_job" "wav2vec"

echo "=== Step 4: Download CLIP visual encoder ==="
clip_job=$(start_download "clip")
poll_until_done "$clip_job" "clip"

echo ""
echo "All weights downloaded. Restart tribe-server to load them."
echo "Then re-run the sweep: cargo run -p sweep"
```

```bash
chmod +x scripts/download_weights.sh
```

- [ ] **Step 2: Create `results/.gitignore`**

```
# sweep_results.json is large (~12MB) — regenerate with: cargo run -p sweep
sweep_results.json
```

- [ ] **Step 3: Commit**

```bash
git add scripts/download_weights.sh results/.gitignore
git commit -m "feat: add weight download script and results gitignore"
```

---

### Task 8: End-to-end verification

**Files:** none created

- [ ] **Step 1: Start tribe-server in demo mode**

In a separate terminal:
```bash
cd ../tribe-playground
cargo run --release -p tribe-server
```
Expected output includes: `TRIBE server listening` on port 8081. When weights are missing it will print warnings but still start in demo mode.

- [ ] **Step 2: Run the full sweep**

```bash
cargo run -p sweep
```
Expected: progress lines for all 150 stimuli, then a top-10 table, then a demo mode warning.

- [ ] **Step 3: Verify output files exist and are well-formed**

```bash
python3 -c "
import json, csv
# ranked CSV
with open('results/sweep_ranked.csv') as f:
    rows = list(csv.DictReader(f))
print(f'sweep_ranked.csv: {len(rows)} rows')
assert rows[0]['rank'] == '1', 'first row must be rank 1'
assert float(rows[0]['global_mean']) >= float(rows[-1]['global_mean']), 'must be descending'

# heatmap JSON
h = json.load(open('results/region_heatmap.json'))
print(f'region_heatmap.json: {len(h[\"content_types\"])} types x {len(h[\"regions\"])} regions')
assert len(h['regions']) == 6
assert all(len(row) == 6 for row in h['matrix'])

print('All output files valid.')
"
```
Expected:
```
sweep_ranked.csv: 150 rows
region_heatmap.json: 13 types x 6 regions
All output files valid.
```

- [ ] **Step 4: Final test suite**

```bash
cargo test --workspace
```
Expected: all existing tests PASS plus 8 new sweep tests.

- [ ] **Step 5: Commit**

```bash
git add results/.gitignore
git commit -m "chore: verify sweep end-to-end and confirm output correctness"
```
