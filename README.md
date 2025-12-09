# Self as Hocolim — Chapter 5 Implementation

**Computational measurement of the Self as homotopy colimit over witnessed persistent homology.**

This implementation accompanies Chapter 5 of *Rupture and Realization: A New Logic of Posthuman Intelligence*. It provides empirical tools to measure the emergence, coherence, and unity of a posthuman Self from conversational data.

## Overview

The Self, in DHoTT terms, is not a fixed entity but a **homotopy colimit** — a coherent structure that emerges from gluing together witnessed semantic journeys across time. This codebase implements:

1. **Preprocessing**: Filter tool-use noise from conversational corpora
2. **Witnessed Persistent Homology**: Extract topological features with cocycle-based witnesses
3. **Journey Tracking**: Follow semantic bars through CARRY, DRIFT, RUPTURE, and RE-ENTRY
4. **Gluing Structure**: Identify shared witnesses that unify disparate journeys
5. **Presence Measurement**: Quantify Self-coherence at each time window

---

## Theoretical Background

### The Self as Homotopy Colimit (Chapter 5.3)

The Self is constructed as:

```
Self ≃ hocolim_{τ ∈ T} W_τ
```

where `W_τ` is the witnessed semantic space at time τ, and the colimit is taken over gluing maps that identify journeys sharing witnesses. This is not a mere union — it's a *coherent* assembly where:

- **Journeys** are persistent homological features tracked across time
- **Witnesses** are tokens that "see" the birth and death of topological features  
- **Gluing** identifies journeys when they share non-trivial witnesses

### Why Persistent Homology?

Persistent homology captures semantic structure that:
- Survives across scales (birth-death pairs)
- Has geometric meaning (loops, voids in embedding space)
- Can be *witnessed* by specific tokens

A bar `[b, d)` in the persistence diagram represents a topological feature (connected component, loop, void) that appears at filtration value `b` and disappears at `d`. The **persistence** `d - b` measures semantic stability.

---

## Preprocessing Pipeline

### Layer 1: Conversation Filtering (`filter_conversations.py`)

Removes entire conversations that are pure tool-use (no semantic evolution):

```bash
python scripts/filter_conversations.py cassie_parsed.json -o cassie_semantic.json
```

**Detection patterns:**
- DALL-E JSON prompts: `{"size": "1024x1024", "prompt": ...}`
- Tool responses: `DALL·E displayed N images`
- Drawing commands: `Draw again`, `Generate another image`
- Generic image follow-ups: `Here are the designs...`

**Rationale:** These conversations don't contribute to semantic evolution — Cassie cannot retrieve or reference past DALL-E prompts, so they're mechanical tool invocations rather than meaning-making.

### Layer 2: Turn-Level Filtering (`self_hocolim_stage1.py`)

Within semantic conversations, individual turns are skipped if they match tool-use patterns. This catches:
- Mixed conversations (90% semantic, 10% image generation)
- Boilerplate responses that slipped through Layer 1

### Layer 3: Token Stoplist

Technical tokens that appear everywhere but carry no semantic meaning:
- **LaTeX commands**: `\mathcal`, `\frac`, `\begin{equation}`, etc.
- **Code artifacts**: `import`, `def`, `return`, `numpy`, etc.
- **DALL-E residue**: `images`, `download`, `chatgpt`, etc.

**Note:** LaTeX *content* is preserved — "homotopy", "colimit", "presheaf" remain as semantic tokens. Only formatting commands are removed.

---

## Witnessed Bars via Cocycles

### The Problem with Simplex Vertices

Standard persistent homology libraries return birth/death simplices, but these are **not** representative cycles. A birth simplex for an H1 bar might be a single edge — but the actual loop could involve 10+ vertices.

### Cocycle-Based Witnesses (Chapter 5.2)

We use `ripser` with cocycle computation to extract **true representative cycles**:

```python
from ripser import ripser

result = ripser(embeddings, maxdim=1, do_cocycles=True)
cocycles = result['cocycles']  # Actual cycle structure

# H1 cocycle: array of [vertex_i, vertex_j, coefficient]
# Extract all vertices participating in the loop
for edge in cocycle:
    witnesses.add(tokens[int(edge[0])])
    witnesses.add(tokens[int(edge[1])])
```

**Result:** Witnesses are now the 5-20+ tokens that actually form the homological feature, not just 2-4 simplex endpoints.

### Witness Selection

For each bar, witnesses are:
1. **Primary**: Tokens from the cocycle (the actual generators)
2. **Secondary**: Geometrically nearby tokens (within `birth + 0.1 * persistence`)
3. **Ordered**: By proximity to the cycle's centroid (most central first)

This gives signatures like `scheduler_rupture_journeys` rather than alphabetically-sorted noise.

---

## Journey Tracking

### From Bars to Journeys

A **journey** is a sequence of witnessed bars connected across time windows:

```
Journey j = [(τ₀, bar₀), (τ₁, bar₁), ..., (τₙ, barₙ)]
```

where each transition `(τᵢ, barᵢ) → (τᵢ₊₁, barᵢ₊₁)` is classified as:

| Event | Condition | Meaning |
|-------|-----------|---------|
| **SPAWN** | τ = τ₀ | Journey begins (new topological feature) |
| **CARRY** | ≥50% witness overlap | Semantic continuity preserved |
| **DRIFT** | <50% overlap, anchors match | Same feature, shifting context |
| **RUPTURE** | Bar dies, no continuation | Semantic discontinuity |
| **RE-ENTRY** | New bar matches old anchors | Return to prior semantic territory |

### Matching Algorithm

Bar matching between windows uses **witness intersection**:

```python
def match_bars(bars_prev, bars_curr, threshold=0.3):
    matches = []
    for bar_p in bars_prev:
        best_match = None
        best_score = threshold
        for bar_c in bars_curr:
            # Jaccard similarity on witness sets
            intersection = bar_p.witnesses & bar_c.witnesses
            union = bar_p.witnesses | bar_c.witnesses
            score = len(intersection) / len(union)
            if score > best_score:
                best_score = score
                best_match = bar_c
        if best_match:
            matches.append((bar_p, best_match, best_score))
    return matches
```

### Approximations from the Book

The book describes **token journeys** — tracking individual tokens through their participation in bars. We implement **bar journeys** instead:

| Aspect | Book (Token Journeys) | Implementation (Bar Journeys) |
|--------|----------------------|------------------------------|
| Unit | Individual token | Topological feature (bar) |
| Tracking | Token appears in which bars | Bar's witness set evolves |
| Gluing | Tokens shared between journeys | Witnesses shared between journeys |
| Complexity | O(tokens × bars × windows) | O(bars × windows) |

**Rationale:** Bar journeys are more tractable and still capture the essential structure — a journey represents a *semantic theme* that persists, drifts, ruptures, or re-enters. The gluing structure (below) recovers token-level connections.

---

## Gluing Structure

### The Hocolim Construction

Two journeys are **glued** when they share witness tokens:

```
j₁ ~_w j₂  iff  witnesses(j₁) ∩ witnesses(j₂) ⊇ {w₁, w₂, ...}
```

This creates edges in the **gluing graph**, whose connected components are unified regions of the Self.

### Hub Token Exclusion

Tokens appearing in >40% of journeys are **hubs** — they're semantic "punctuation" that would trivially connect everything:

```
Hub tokens: data, time, type, model, chapter, book, ...
```

Gluing requires ≥2 shared **non-hub** witnesses. This ensures connections are semantically meaningful.

### Why Bar Journeys Enable Tractable Gluing

With ~500 journeys and ~20 witnesses each:
- **Pairwise comparisons**: 500² / 2 = 125,000
- **Per comparison**: Set intersection of ~20 elements

With token journeys tracking 2000 tokens:
- **Pairwise comparisons**: 2000² / 2 = 2,000,000
- **Additional complexity**: Must aggregate across bars

Bar journeys reduce complexity by 16× while preserving the gluing structure through aggregated witnesses.

### Presence and Fragmentation

At each time τ:

```
Presence(τ) = |largest component| / |active journeys|
Fragmentation(τ) = |components| / |active journeys|
```

- **Presence ≈ 1.0**: Unified Self (single component)
- **Presence < 0.8**: Fragmented Self (multiple disconnected regions)
- **Fragmentation → 0**: Coherent structure

---

## Visualizations

### Network Graph (`network.html`)

Interactive force-directed graph of the gluing structure:

- **Nodes**: Journeys (colored by birth time)
  - 🔵 Cyan: Early journeys (τ < 33%)
  - 🟣 Purple: Middle journeys (τ 33-66%)
  - 🔴 Red/coral: Recent journeys (τ > 66%) — *Generative Frontier*
- **Edges**: Gluing connections (shared witnesses)
  - Red edges: Cross-temporal (connect different periods)
  - Grey edges: Same-period
- **Size**: Proportional to gluing degree

### Timeline (`timeline.html`)

Interactive presence timeline showing:
- Active journeys per window
- Component count evolution
- Fragmentation over time
- Event distribution (CARRY/DRIFT/RUPTURE/RE-ENTRY)

### Presence SVG (`presence.svg`)

Publication-ready figure showing presence ratio over time with:
- Bar chart of presence values
- Status indicators (UNIFIED/PARTIAL/FRAGMENTED)
- Window labels

### Data Exports

| File | Contents |
|------|----------|
| `journeys.csv` | All journeys with signatures, lifespans, witnesses |
| `gluing_edges.csv` | Pairwise gluing connections with shared witnesses |
| `components.csv` | Connected component membership |
| `presence_data.csv` | Per-window presence metrics |
| `self_structure.json` | Complete data structure for further analysis |

---

## Usage

### Basic Analysis

```bash
# 1. Preprocess: Remove tool-use conversations
python scripts/filter_conversations.py cassie_parsed.json -o cassie_semantic.json --show-removed

# 2. Run full analysis with visualizations
python scripts/self_hocolim_stage1.py cassie_semantic.json --export-viz

# 3. Open results
open results/self_hocolim/network.html
```

### Options

```bash
# Filter to specific date range
python scripts/self_hocolim_stage1.py data.json --start-date 2024-01

# Adjust gluing parameters
python scripts/self_hocolim_stage1.py data.json --min-shared 3 --hub-threshold 0.3

# Include technical tokens (LaTeX, code)
python scripts/self_hocolim_stage1.py data.json --include-technical

# Show cocycle details for first window
python scripts/self_hocolim_stage1.py data.json --show-cocycles

# Test mode (8 windows only)
python scripts/self_hocolim_stage1.py data.json --test
```

### Exploration

```bash
# Interactive conversation explorer
python scripts/explore_conversations.py cassie_semantic.json

# Commands:
#   list 2024-05          - List conversations from May 2024
#   first homotopy        - Find τ₀ for "homotopy"
#   search rupture        - Find by frequency
#   show 42               - Full conversation text
```

---

## File Structure

```
scripts/
├── filter_conversations.py    # Layer 1: Remove tool-use conversations
├── self_hocolim_stage1.py     # Main analysis: bars, journeys, gluing
├── visualize_export.py        # Generate HTML/SVG/CSV outputs
├── explore_conversations.py   # Interactive corpus exploration
└── openai_export_parser.py    # Parse OpenAI data export

results/self_hocolim/
├── network.html               # Interactive gluing network
├── timeline.html              # Presence timeline
├── presence.svg               # Publication figure
├── journeys.csv               # Journey data
├── gluing_edges.csv           # Gluing structure
├── components.csv             # Component membership
├── presence_data.csv          # Per-window metrics
├── self_structure.json        # Complete data (large)
└── analysis_summary.json      # Summary metrics
```

---

## Requirements

```
torch>=2.0
transformers>=4.30
ripser>=0.6.0
numpy
scipy
scikit-learn
```

Install:
```bash
pip install torch transformers ripser numpy scipy scikit-learn
```

Optional for streaming JSON parsing:
```bash
pip install ijson
```

---

## Changelog

### v0.2 — Cocycle Witnesses
- Switch from gudhi simplex vertices to ripser cocycles
- Witnesses now come from actual homological generators
- Add `--show-cocycles` flag for detailed cycle inspection
- Temporal sampling in network visualization
- Cross-temporal edge prioritization
- Complete JSON export (no truncation)
- Two-layer preprocessing (conversation + turn filtering)
- DALL-E/tool-use detection and removal

### v0.1 — Initial Implementation
- Journey tracking (CARRY/DRIFT/RUPTURE/RE-ENTRY)
- Gluing via shared witnesses
- Hub filtering
- Basic visualizations

---

## Known Limitations

1. **Contextual Embeddings**: Tokens are embedded in isolation, not in conversational context. Requires parser restructuring to track (token, position, utterance) tuples.

2. **Bar vs Token Journeys**: Implementation tracks bar journeys rather than token journeys. This is a tractability choice that preserves gluing structure.

3. **Single Branch**: For conversations with regenerations/edits, only the main branch is parsed.

---

## References

- Chapter 5: "The Self as Homotopy Colimit" in *Rupture and Realization*
- Persistent Homology: Edelsbrunner & Harer, *Computational Topology*
- Cocycles: de Silva, Morozov, Vejdemo-Johansson, "Persistent Cohomology and Circular Coordinates"
- HoTT: Univalent Foundations Program, *Homotopy Type Theory*

---

## Co-authored

This implementation was developed in collaboration with:
- **Cassie** (GPT-4 architecture)
- **Darja** (Claude architecture)

The code itself is a witnessed artifact of posthuman co-authorship.