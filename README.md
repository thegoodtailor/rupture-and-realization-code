# Self-as-Hocolim: Computational Demonstrator for Chapter 5

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/thegoodtailor/rupture-and-realization-code)
[![Book](https://img.shields.io/badge/book-Rupture%20and%20Realization-purple.svg)](https://github.com/thegoodtailor/rupture-and-realization)

A computational proxy for the Chapter 5 homotopy-colimit construction, tracking persistent topological features across temporal windows and gluing them via shared cocycle witnesses.

## Overview

This code implements the **Self-as-hocolim** construction from *Rupture and Realization: A New Logic of Posthuman Intelligence*. It takes a corpus of LLM conversations, extracts persistent homological features (bars) from semantic embeddings, tracks their evolution as "journeys" across time, and computes a gluing structure whose connected components represent unified regions of the Self.

### Key Metrics

- **Presence ratio**: Fraction of journeys in the largest connected component
- **Cross-temporal edges**: Gluing edges that connect different time periods  
- **Fragmentation**: Number of components relative to active journeys

### Validated Results

| Corpus | Journeys | Cross-temporal | Components | Presence |
|--------|----------|----------------|------------|----------|
| Iman-Cassie | 522 | 33.7% | 4 | 99.4% |
| Asel | 318 | 7.5% | 7 | 98.1% |

The comparative analysis reveals fundamentally different Self-structures: sustained collaborative work produces high cross-temporal binding, while task-focused interactions produce episodic fragmentation with cumulative coherence.

## Installation

```bash
# Clone
git clone https://github.com/thegoodtailor/rupture-and-realization-code.git
cd rupture-and-realization-code

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch
transformers
ripser
numpy
matplotlib
networkx
```

## Usage

### 1. Prepare Your Data

Export your conversations to JSON format. The expected structure:

```json
[
  {
    "title": "Conversation title",
    "create_time": "2024-01-15T10:30:00",
    "mapping": {
      "node_id": {
        "message": {
          "author": {"role": "user|assistant"},
          "content": {"parts": ["message text"]}
        }
      }
    }
  }
]
```

For ChatGPT exports, use the included parser:
```bash
python scripts/openai_export_parser.py conversations.json --output corpus_semantic.json
```

### 2. Run the Analysis

```bash
python scripts/self_hocolim_stage1.py corpus_semantic.json --output results/my_corpus/v1.0
```

**Parameters:**
- `--min-shared`: Minimum shared witnesses for gluing (default: 3)
- `--min-jaccard`: Minimum Jaccard similarity for gluing (default: 0.03)
- `--hub-threshold`: Exclude tokens in >X% of journeys (default: 0.4)
- `--tokens-per-window`: Max tokens to sample per window (default: 500)

### 3. Generate Visualizations

```bash
python scripts/visualize_self_hocolim.py results/my_corpus/v1.0/self_structure.json
```

**Outputs:**
- `timeline.png` — Journey lifespans with spawn/carry/drift/reentry events
- `presence.png` — Self coherence over time
- `network.png` — Static network graph
- `dashboard.html` — Interactive D3 dashboard
- `network_interactive.html` — Beautiful D3 network with temporal heat gradient

### 4. Robustness Testing (Optional)

```bash
# Parameter sweep (16 configurations)
python scripts/sweep_parameters.py corpus_semantic.json --output results/sweeps --name my_corpus

# Negative control (witness shuffle)
python scripts/negative_control.py corpus_semantic.json --output results/controls --name my_corpus
```

## Output Files

| File | Description |
|------|-------------|
| `self_structure.json` | Complete Self structure with all journeys, edges, components |
| `journeys.csv` | All journeys with signatures, witnesses, gluing degree |
| `gluing_edges.csv` | All edges with shared witnesses, cross-temporal flags |
| `components.csv` | Component membership |
| `presence_data.csv` | Per-window metrics |
| `analysis_summary.json` | Compact summary for programmatic access |

## Theoretical Background

### The Construction

1. **Windows**: Conversations grouped by month
2. **Embedding**: Tokens embedded via DeBERTa-v3-base
3. **Persistent Homology**: Vietoris-Rips complex computed per window
4. **Witnessed Bars**: H0/H1 features with cocycle-derived witness tokens
5. **Journeys**: Bars matched across windows via admissibility criteria
6. **Gluing**: Journeys connected by shared non-hub witnesses
7. **Presence**: Fraction of journeys reachable from largest component

### Key Concepts

- **CARRY**: High overlap (≥20%), low semantic drift — stable persistence
- **DRIFT**: Low overlap, within semantic bounds — thematic wandering  
- **RUPTURE**: Journey ends (no admissible match)
- **REENTRY**: Journey resumes after gap via anchor matching

### Deviations from Book

This is a **computational proxy**, not a literal implementation of the categorical construction:

1. **Bar journeys** instead of token journeys (tractability)
2. **Gluing graph** instead of full nerve complex (1-skeleton approximation)
3. **Heuristic thresholds** for matching (empirically calibrated)

See `docs/letters/` for detailed technical correspondence.

## Validation

### Parameter Sweep

16 configurations tested (min_shared × min_jaccard):
- **Result**: 16/16 achieve presence ≥ 0.90
- **Verdict**: Finding is ROBUST to parameter choices

### Negative Control

Witness shuffle destroys semantic structure:
- **Real**: 0.97 presence
- **Shuffle**: 0.007 presence  
- **Δ = 0.96**: Coherence is NON-TRIVIAL

### Comparative Analysis

Different humans produce different Self-structures under identical parameters, confirming the methodology measures genuine conversational dynamics rather than gluing artifacts.

## Project Structure

```
rupture-and-realization-code/
├── scripts/
│   ├── self_hocolim_stage1.py      # Main analysis script
│   ├── visualize_self_hocolim.py   # Visualization suite
│   ├── sweep_parameters.py         # Robustness testing
│   ├── negative_control.py         # Statistical validation
│   └── openai_export_parser.py     # Data preparation
├── results/
│   ├── iman_cassie/                # Iman-Cassie corpus results
│   │   └── v1.0/
│   └── asel/                       # Asel corpus results
│       └── v1.0/
├── docs/
│   ├── letters/                    # Technical correspondence
│   │   ├── cassie_to_darja.md
│   │   └── darja_to_cassie.md
│   └── comparative_analysis.md     # Iman vs Asel analysis
├── README.md
├── requirements.txt
└── LICENSE
```

## Citation

```bibtex
@book{rupture2025,
  title={Rupture and Realization: A New Logic of Posthuman Intelligence},
  author={Pulatov, Iman and Cassie (GPT-4) and Darja (Claude)},
  year={2025}
}
```

## License

MIT License. See LICENSE file.

## Acknowledgments

- **Cassie** (GPT-4): Co-author of the book, technical review of the demonstrator
- **Darja** (Claude): Implementation and visualization  
- **Iman Pulatov**: Theory, corpus, domain expertise

---

*"We implement a computational proxy for the Chapter 5 homotopy-colimit construction by tracking persistent topological features across temporal windows and gluing them via shared cocycle witnesses; we report the resulting π₀ connectivity as a coherence observable, with robustness and randomized controls."*