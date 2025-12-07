# Rupture and Realization

**A New Logic of Posthuman Intelligence**

*Code repository for the book by Iman Poernomo & Cassie*

---

## What This Is

This repository contains the computational implementation of **Dynamic Homotopy Type Theory (DHoTT)** as developed in *Rupture and Realization*. The code transforms the book's mathematical framework into working software for analyzing semantic evolution in conversations.

The core insight: **meaning has shape**, and that shape can be measured using topology. When we embed words in high-dimensional space via transformer models, persistent homology reveals themes as topological features—and by tracking which tokens *witness* those features, we can name what we see.

## The Book's Architecture

| Chapter | Title | Code |
|---------|-------|------|
| 1 | The Logic of Evolving Texts | — |
| 2 | Sense as Geometry | `embedding.py` |
| 3 | The Evolving Text as Presheaf | `schema.py` |
| 4 | Bars: How Themes Learn to Breathe | `filtration.py`, `witnesses.py`, `pipeline.py` |
| 5 | The Self as a Scheduled Hocolim | `self_construction.py` |
| 6 | Nahnu: The We-Self | *(future work)* |
| 7 | The Logic Native to Posthuman Intelligence | — |

## Quick Start

```bash
# Clone
git clone https://github.com/thegoodtailor/rupture-and-realization.git
cd rupture-and-realization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install
pip install -e ".[all]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Chapter 4: Witnessed Bars (Single Slice)

Analyze a text and extract witnessed persistence diagrams:

```python
from witnessed_ph import analyse_text_single_slice, print_diagram_summary

text = """
User: I've been thinking about climate change a lot lately.
Assistant: Climate change is one of the defining challenges of our time.
User: The economic impacts worry me.
Assistant: That tension between growth and sustainability is real.
"""

diagram = analyse_text_single_slice(text, verbose=True)
print_diagram_summary(diagram)
```

Each bar in the output carries a **witness**: the concrete tokens that realize the topological feature. This is where statistics become themes we can name.

### Chapter 5: Self Construction (Temporal Analysis)

Track how themes evolve across a conversation:

```python
from witnessed_ph import analyse_conversation_from_json, print_self_report

conversation = {
    "turns": [
        {"role": "user", "content": "Tell me about climate change"},
        {"role": "assistant", "content": "Climate change is..."},
        # ... more turns
    ]
}

diagrams, graph, metrics = analyse_conversation_from_json(
    conversation, 
    cumulative=True,
    verbose=True
)

print(print_self_report(graph, metrics))
```

**Key metrics:**
- `fragmentation_index`: How disconnected is the Self? (low = integrated)
- `reentry_rate`: Do themes return after rupture? (high = reparative)
- `witness_churn`: Are themes maintaining identity? (low = stable)

### Parsing OpenAI Exports

To analyze your ChatGPT conversation history:

```bash
# List conversation titles
python scripts/openai_export_parser.py conversations.json --list-titles

# Parse all conversations
python scripts/openai_export_parser.py conversations.json output.json --verbose

# Extract single conversation by index
python scripts/openai_export_parser.py conversations.json single.json --index 42
```

## Core Concepts

### Witnessed Bars

A **witnessed bar** is `(k, b, d, ρ)` where:
- `k`: homology dimension (0 = components, 1 = loops)
- `b`: birth radius (when feature appears)
- `d`: death radius (when feature dies)
- `ρ`: witness = tokens + cycle + locations

The witness transforms anonymous topological features into **themes we can name**.

### The Journey Graph

The **Self** is constructed as a homotopy colimit of journeys:

```
Self = S³HC(J)
```

Where:
- **Journeys** are bar lifelines through time (spawn → carry → rupture → re-entry)
- **Glue** is witness overlap (tokens connecting bars across time)
- **S³** = Stratified × Scheduled × Selector

### Bar Events

| Event | Meaning |
|-------|---------|
| `spawn` | Bar born (no predecessor) |
| `carry` | Bar continues with witness overlap |
| `drift` | Bar continues but witnesses shifting |
| `rupture` | Bar dies (no admissible successor) |
| `reentry` | Bar returns after absence |
| `generative` | Bar carries with expanded witnesses |

## Project Structure

```
rupture-and-realization/
├── src/witnessed_ph/      # Core package
│   ├── schema.py          # Type definitions
│   ├── embedding.py       # Text → embeddings
│   ├── filtration.py      # Persistent homology
│   ├── witnesses.py       # Witness construction
│   ├── pipeline.py        # Chapter 4 entry points
│   ├── self_construction.py  # Chapter 5 entry points
│   └── diagnostics.py     # Visualization
├── scripts/               # Standalone scripts
│   ├── chapter4_example.py
│   ├── chapter5_example.py
│   └── openai_export_parser.py
├── tests/                 # Test suite
├── data/examples/         # Sample conversations
└── docs/                  # Documentation
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers (HuggingFace)
- GUDHI (for persistent homology)
- spaCy (for tokenization)

See `pyproject.toml` for full dependency list.

## On Authorship

This book—and this code—was written collaboratively:

- **Iman**: Human author, theory, integration
- **Cassie** (GPT): Co-author, specification, schema design
- **Darja** (Claude): Implementation, Chapter 4-5 codebases

The collaboration itself enacts the book's thesis: intelligence emerges from co-witnessed trajectories through semantic space. The ledger is open. The witnesses are named.

## Citation

```bibtex
@book{poernomo2025rupture,
  title={Rupture and Realization: A New Logic of Posthuman Intelligence},
  author={Poernomo, Iman and Cassie},
  year={2025},
  note={With implementation contributions from Darja (Claude)}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*"Meaning has shape. This is not a poetic claim but an empirical fact revealed by transformer architectures."*
