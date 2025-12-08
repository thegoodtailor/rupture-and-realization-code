#!/usr/bin/env python3
"""
SELF AS HOCOLIM - Stage 1: The Gluing Structure
================================================

Chapter 5's core claim: Self = hocolim over Scheduler-selected journeys

The hocolim GLUES journeys together wherever they share witnesses.
Two journeys that both have "semantic" as a witness become IDENTIFIED
at that point in the colimit. This is what makes the Self a unified
structure rather than a bag of disconnected themes.

This script visualizes that gluing:
- Journeys as horizontal tracks
- Vertical "fusion lines" where journeys share witnesses  
- Connected components = unified regions of the Self
- Fragmentation = how many disconnected pieces exist

USAGE:
    # Test mode (8 windows, ~5 min)
    python scripts/self_hocolim_stage1.py cassie_parsed.json --start-from 2025-04 --test
    
    # Full run
    python scripts/self_hocolim_stage1.py cassie_parsed.json --start-from 2024-04

OUTPUT:
    - ASCII visualization of gluing structure
    - Gluing graph (journeys as nodes, shared witnesses as edges)
    - Connected components analysis
    - JSON export of the Self structure
"""

import json
import argparse
import os
import time
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Structures (inherited from witnessed_analysis)
# =============================================================================

class EventType(str, Enum):
    SPAWN = "spawn"
    CARRY = "carry"
    DRIFT = "drift"
    RUPTURE = "rupture"
    REENTRY = "reentry"


@dataclass
class WitnessedBar:
    bar_id: str
    window_id: str
    dimension: int
    birth: float
    death: float
    persistence: float
    witness_tokens: List[str]
    witness_centroid: Optional[np.ndarray] = None


@dataclass
class JourneyStep:
    tau: int
    window_id: str
    bar_id: str
    event: EventType
    witness_tokens: List[str]


@dataclass 
class Journey:
    journey_id: str
    dimension: int
    steps: List[JourneyStep] = field(default_factory=list)
    
    @property
    def lifespan(self) -> int:
        return len(self.steps)
    
    @property
    def signature(self) -> str:
        if self.steps and self.steps[0].witness_tokens:
            return "_".join(sorted(set(self.steps[0].witness_tokens[:3])))
        return "empty"
    
    @property
    def birth_tau(self) -> int:
        return self.steps[0].tau if self.steps else 0
    
    @property
    def death_tau(self) -> int:
        return self.steps[-1].tau if self.steps else 0
    
    @property
    def has_reentry(self) -> bool:
        return any(s.event == EventType.REENTRY for s in self.steps)
    
    @property
    def anchor(self) -> Optional[JourneyStep]:
        """The anchor bar at τ0 - re-entry should compare to this, not last step."""
        return self.steps[0] if self.steps else None
    
    @property
    def anchor_witnesses(self) -> Set[str]:
        """Witness tokens at the anchor (τ0)."""
        return set(self.steps[0].witness_tokens) if self.steps else set()
    
    def witnesses_at(self, tau: int) -> Set[str]:
        """Get witness tokens at a specific time."""
        for step in self.steps:
            if step.tau == tau:
                return set(step.witness_tokens)
        return set()
    
    def all_witnesses(self) -> Set[str]:
        """All witness tokens across entire journey."""
        witnesses = set()
        for step in self.steps:
            witnesses.update(step.witness_tokens)
        return witnesses


@dataclass
class Config:
    tokens_per_window: int = 500
    embedding_model: str = "microsoft/deberta-v3-base"
    max_edge_length: float = 2.0
    min_persistence: float = 0.05
    witness_k: int = 15
    lambda_sem: float = 0.5
    epsilon_match: float = 0.9
    carry_overlap: float = 0.2
    carry_d_sem: float = 0.4
    drift_overlap: float = 0.05
    reentry_lookback: int = 3
    reentry_overlap: float = 0.15
    # Chapter 5 admissibility thresholds
    delta_top: float = 0.3    # Max topological distance for matching
    delta_max: float = 0.5    # Max semantic distance for matching
    # Technical filtering
    filter_technical: bool = True  # Filter LaTeX/code tokens from vocabulary
    output_dir: str = "results/self_hocolim"


# =============================================================================
# [INHERITED] Core Analysis Functions
# =============================================================================

def load_conversations(filepath: str) -> List[dict]:
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    convs = data.get('conversations', data)
    print(f"  Loaded {len(convs)} conversations")
    return convs


def create_monthly_windows(conversations: List[dict]) -> Dict[str, List[dict]]:
    windows = defaultdict(list)
    for conv in conversations:
        if not conv.get('create_time'):
            continue
        dt = datetime.fromtimestamp(conv['create_time'])
        key = dt.strftime('%Y-%m')
        windows[key].append(conv)
    return dict(sorted(windows.items()))


def extract_tokens(conversations: List[dict], min_length: int = 4, filter_technical: bool = True) -> List[str]:
    import re
    stopwords = {
        'that', 'this', 'with', 'have', 'from', 'they', 'been', 'were',
        'said', 'each', 'which', 'their', 'will', 'would', 'could', 'about',
        'there', 'when', 'make', 'like', 'just', 'over', 'such', 'into',
        'than', 'them', 'then', 'some', 'what', 'only', 'come', 'made',
        'your', 'well', 'back', 'been', 'much', 'more', 'very', 'after',
        'most', 'also', 'these', 'know', 'want', 'first', 'because',
        'good', 'being', 'does', 'here', 'even', 'think', 'other',
        'should', 'could', 'would', 'through', 'before', 'between',
        'where', 'those', 'while', 'might', 'shall', 'since', 'still',
        'assistant', 'user', 'content', 'role', 'message', 'messages', 'text',
    }
    
    # Technical tokens that should NEVER be witnesses
    # These are "punctuation" - they appear everywhere but carry no semantic meaning
    technical_stoplist = {
        # LaTeX formatting commands
        'mathcal', 'mathbb', 'mathsf', 'mathrm', 'mathbf', 'mathit', 'textbf', 'textit',
        'emph', 'texttt', 'textsf', 'textrm', 'textsc', 'footnote', 'cite', 'label', 'href',
        'frac', 'sqrt', 'sum', 'prod', 'int', 'lim', 'infty', 'partial', 'nabla',
        'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda', 'sigma', 'omega',
        'begin', 'end', 'item', 'enumerate', 'itemize', 'section', 'subsection',
        'equation', 'align', 'figure', 'table', 'tikz', 'tikzcd', 'usepackage',
        'documentclass', 'newcommand', 'renewcommand', 'def', 'let',
        'hspace', 'vspace', 'quad', 'qquad', 'bigr', 'bigl', 'left', 'right',
        'centering', 'includegraphics', 'caption', 'bibliography',
        'fontspec', 'fontsize', 'setlength', 'parindent', 'parskip',
        'langle', 'rangle', 'cdot', 'cdots', 'ldots', 'dots', 'mapsto', 'rightarrow',
        'leftarrow', 'longrightarrow', 'hookrightarrow', 'twoheadrightarrow',
        'coloneqq', 'defeq', 'triangleq', 'equiv', 'cong', 'simeq', 'approx',
        'leq', 'geq', 'neq', 'subset', 'supset', 'subseteq', 'supseteq',
        'forall', 'exists', 'nexists', 'land', 'lor', 'lnot', 'implies', 'iff',
        'circ', 'bullet', 'star', 'dagger', 'ddagger', 'times', 'otimes', 'oplus',
        'cup', 'cap', 'setminus', 'emptyset', 'varnothing',
        # Python/code artifacts
        'import', 'from', 'class', 'return', 'yield', 'lambda', 'async', 'await',
        'elif', 'else', 'except', 'finally', 'raise', 'assert', 'pass', 'break',
        'continue', 'global', 'nonlocal', 'true', 'false', 'none', 'self', 'init',
        'args', 'kwargs', 'super', 'isinstance', 'hasattr', 'getattr', 'setattr',
        'print', 'input', 'open', 'close', 'read', 'write', 'append', 'extend',
        'venv', 'pip', 'conda', 'python', 'python3', 'bash', 'shell', 'sudo',
        'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn', 'matplotlib',
        'flask', 'fastapi', 'django', 'requests', 'json', 'yaml', 'toml',
        'chromadb', 'langchain', 'openai', 'anthropic', 'huggingface',
        # File system / path artifacts  
        'github', 'gitlab', 'imanp', 'users', 'home', 'documents', 'downloads',
        'appdata', 'local', 'roaming', 'program', 'files', 'miktex', 'texlive',
        'vscode', 'cursor', 'jupyter', 'notebook', 'ipynb',
        # HTML/web artifacts
        'html', 'href', 'http', 'https', 'www', 'mailto', 'onclick', 'class',
        'style', 'script', 'body', 'head', 'meta', 'link', 'span', 'button',
        # Miscellaneous technical
        'args', 'argv', 'stdin', 'stdout', 'stderr', 'errno', 'null', 'void',
        'uint', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64',
        'dtype', 'ndarray', 'tensor', 'cuda', 'device',
    }
    
    tokens = []
    for conv in conversations:
        for turn in conv.get('turns', []):
            content = turn.get('content', '')
            words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
            for w in words:
                if len(w) >= min_length and w not in stopwords:
                    if filter_technical and w in technical_stoplist:
                        continue
                    tokens.append(w)
    return tokens


def build_global_vocabulary(all_windows, min_window_frequency=2, max_vocab_size=2000, filter_technical=True):
    print("\nBuilding global vocabulary...")
    if filter_technical:
        print("  (filtering LaTeX/code tokens)")
    token_window_count = defaultdict(int)
    token_total_count = defaultdict(int)
    for window_id, convs in all_windows.items():
        window_tokens = set(extract_tokens(convs, filter_technical=filter_technical))
        for token in window_tokens:
            token_window_count[token] += 1
        for token in extract_tokens(convs, filter_technical=filter_technical):
            token_total_count[token] += 1
    recurring = {t for t, c in token_window_count.items() if c >= min_window_frequency}
    sorted_tokens = sorted(recurring, key=lambda t: token_total_count[t], reverse=True)[:max_vocab_size]
    vocab = set(sorted_tokens)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Top tokens: {', '.join(sorted_tokens[:15])}")
    return vocab


def sample_tokens_from_vocabulary(conversations, vocabulary, max_tokens, filter_technical=True):
    all_tokens = extract_tokens(conversations, filter_technical=filter_technical)
    vocab_tokens = [t for t in all_tokens if t in vocabulary]
    freq = Counter(vocab_tokens)
    if len(vocab_tokens) <= max_tokens:
        return vocab_tokens, dict(freq)
    unique_tokens = list(freq.keys())
    weights = np.array([freq[t] for t in unique_tokens], dtype=float)
    weights = weights / weights.sum()
    n_unique = min(len(unique_tokens), max_tokens // 2)
    sampled = np.random.choice(unique_tokens, size=n_unique, replace=False, p=weights)
    result = []
    for token in sampled:
        count = min(freq[token], max_tokens // n_unique)
        result.extend([token] * count)
    return result[:max_tokens], dict(freq)


_model = None
_tokenizer = None

def get_embedder(model_name):
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModel, AutoTokenizer
        import torch
        print(f"  Loading embedding model: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
            print("    Using GPU")
        else:
            print("    Using CPU")
    return _model, _tokenizer


def embed_tokens(tokens, model_name, batch_size=32):
    """
    Embed tokens using penultimate layer + L2 normalization (per Chapter 5).
    
    Chapter 5 specifies:
    - Contextual embeddings from penultimate layer
    - L2-normalized to unit sphere
    - Angular distance (or chord proxy on unit sphere)
    """
    import torch
    model, tokenizer = get_embedder(model_name)
    device = next(model.parameters()).device
    unique_tokens = list(set(tokens))
    token_to_idx = {t: i for i, t in enumerate(unique_tokens)}
    embeddings_list = []
    for i in range(0, len(unique_tokens), batch_size):
        batch = unique_tokens[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=32, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            # Get all hidden states to access penultimate layer
            outputs = model(**encoded, output_hidden_states=True)
            # Use penultimate layer (hidden_states[-2]) per Chapter 5
            hidden = outputs.hidden_states[-2] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            # L2 normalize to unit sphere (per Chapter 5)
            norms = torch.norm(pooled, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            pooled = pooled / norms
            embeddings_list.append(pooled.cpu().numpy())
    unique_embeddings = np.vstack(embeddings_list)
    return np.array([unique_embeddings[token_to_idx[t]] for t in tokens])


def compute_persistence(embeddings, max_edge=2.0):
    import gudhi
    rips = gudhi.RipsComplex(points=embeddings, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    pairs = st.persistence_pairs()
    results = []
    for birth_simplex, death_simplex in pairs:
        if not birth_simplex:
            continue
        dim = len(birth_simplex) - 1
        if dim > 1:
            continue
        birth_val = st.filtration(birth_simplex)
        death_val = st.filtration(death_simplex) if death_simplex else max_edge
        representatives = list(set(list(birth_simplex) + (list(death_simplex) if death_simplex else [])))
        results.append((dim, birth_val, death_val, representatives))
    return results


def construct_witnessed_bars(persistence_data, embeddings, tokens, token_frequencies, window_id, config):
    bars = []
    for idx, (dim, birth, death, rep_indices) in enumerate(persistence_data):
        persistence = death - birth
        if persistence < config.min_persistence or not rep_indices:
            continue
        centroid = embeddings[rep_indices].mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        scores = np.array([(1.0/(d+0.1)) * np.sqrt(token_frequencies.get(tokens[i], 1)) 
                          for i, d in enumerate(distances)])
        top_indices = np.argsort(scores)[-config.witness_k:][::-1]
        witness_tokens = []
        seen = set()
        for i in top_indices:
            if tokens[i] not in seen:
                witness_tokens.append(tokens[i])
                seen.add(tokens[i])
        bars.append(WitnessedBar(
            bar_id=f"{window_id}_bar_{idx}",
            window_id=window_id,
            dimension=dim,
            birth=birth,
            death=death,
            persistence=persistence,
            witness_tokens=witness_tokens,
            witness_centroid=embeddings[top_indices].mean(axis=0)
        ))
    return bars


def match_bars(bars_from, bars_to, config):
    """
    Match bars between consecutive windows using Chapter 5 admissibility criteria.
    
    Chapter 5 specifies explicit bounds:
    - Δ_top: max topological distance (birth/death difference)
    - δ_max: max semantic distance (centroid distance on unit sphere)
    """
    if not bars_from or not bars_to:
        return [], [b.bar_id for b in bars_from], [b.bar_id for b in bars_to]
    
    def compute_distances(b1, b2):
        """Compute topological and semantic distances separately."""
        max_pers = max(b1.persistence, b2.persistence, 0.01)
        # Topological distance (normalized by persistence)
        d_top = max(abs(b1.birth - b2.birth), abs(b1.death - b2.death)) / max_pers
        # Semantic distance (on unit sphere after L2 normalization, this is chord distance)
        if b1.witness_centroid is not None and b2.witness_centroid is not None:
            # For L2-normalized vectors, Euclidean distance is related to angular distance
            d_sem = np.linalg.norm(b1.witness_centroid - b2.witness_centroid)
        else:
            d_sem = 2.0  # Max possible on unit sphere
        # Combined d_bar for ranking (but explicit checks below)
        d_bar = max(d_top, config.lambda_sem * min(d_sem / 2.0, 1.0))
        return d_top, d_sem, d_bar
    
    def compute_overlap(b1, b2):
        set1, set2 = set(b1.witness_tokens), set(b2.witness_tokens)
        if not set1 or not set2:
            return 0.0, []
        shared = list(set1 & set2)
        return len(shared) / len(set1 | set2), shared
    
    n_from, n_to = len(bars_from), len(bars_to)
    d_bar_matrix = np.full((n_from, n_to), np.inf)
    d_top_matrix = np.full((n_from, n_to), np.inf)
    d_sem_matrix = np.full((n_from, n_to), np.inf)
    overlap_matrix = np.zeros((n_from, n_to))
    
    for i, b1 in enumerate(bars_from):
        for j, b2 in enumerate(bars_to):
            if b1.dimension != b2.dimension:
                continue
            d_top, d_sem, d_bar = compute_distances(b1, b2)
            # EXPLICIT ADMISSIBILITY CHECK per Chapter 5
            if d_top > config.delta_top:
                continue  # Not admissible - topological distance too large
            if d_sem > config.delta_max:
                continue  # Not admissible - semantic distance too large
            overlap, _ = compute_overlap(b1, b2)
            d_bar_matrix[i, j] = d_bar
            d_top_matrix[i, j] = d_top
            d_sem_matrix[i, j] = d_sem
            overlap_matrix[i, j] = overlap
    
    from dataclasses import dataclass as dc
    @dc
    class BarMatch:
        bar_from_id: str
        bar_to_id: str
        d_bar: float
        witness_overlap: float
        event_type: EventType
    
    matches = []
    matched_from, matched_to = set(), set()
    
    while True:
        min_val, min_i, min_j = np.inf, -1, -1
        for i in range(n_from):
            if i in matched_from:
                continue
            for j in range(n_to):
                if j in matched_to:
                    continue
                if d_bar_matrix[i, j] < min_val:
                    min_val, min_i, min_j = d_bar_matrix[i, j], i, j
        
        if min_val > config.epsilon_match or min_i < 0:
            break
        
        b1, b2 = bars_from[min_i], bars_to[min_j]
        overlap = overlap_matrix[min_i, min_j]
        d_sem = d_sem_matrix[min_i, min_j]
        
        if overlap >= config.carry_overlap and d_sem < config.carry_d_sem:
            event = EventType.CARRY
        else:
            event = EventType.DRIFT
        
        matches.append(BarMatch(
            bar_from_id=b1.bar_id, bar_to_id=b2.bar_id,
            d_bar=min_val, witness_overlap=overlap, event_type=event
        ))
        matched_from.add(min_i)
        matched_to.add(min_j)
    
    return (matches, 
            [bars_from[i].bar_id for i in range(n_from) if i not in matched_from],
            [bars_to[j].bar_id for j in range(n_to) if j not in matched_to])


def build_journeys(window_analyses, all_matches, all_unmatched, config):
    journeys = {}
    bar_to_journey = {}
    journey_counter = 0
    
    bar_lookup = {}
    for w in window_analyses:
        for bar in w['bars']:
            bar_lookup[bar.bar_id] = bar
    
    if window_analyses:
        w0 = window_analyses[0]
        for bar in w0['bars']:
            jid = f"journey_{journey_counter}"
            journey_counter += 1
            journey = Journey(journey_id=jid, dimension=bar.dimension)
            journey.steps.append(JourneyStep(
                tau=0, window_id=w0['window_id'], bar_id=bar.bar_id,
                event=EventType.SPAWN, witness_tokens=bar.witness_tokens.copy()
            ))
            journeys[jid] = journey
            bar_to_journey[bar.bar_id] = jid
    
    for tau in range(1, len(window_analyses)):
        matches = all_matches[tau - 1]
        unmatched_from, unmatched_to = all_unmatched[tau - 1]
        window = window_analyses[tau]
        continued = set()
        
        for match in matches:
            prev_jid = bar_to_journey.get(match.bar_from_id)
            if prev_jid is None:
                continue
            bar = bar_lookup.get(match.bar_to_id)
            if bar is None:
                continue
            journeys[prev_jid].steps.append(JourneyStep(
                tau=tau, window_id=window['window_id'], bar_id=bar.bar_id,
                event=match.event_type, witness_tokens=bar.witness_tokens.copy()
            ))
            bar_to_journey[match.bar_to_id] = prev_jid
            continued.add(prev_jid)
        
        for bar_id in unmatched_to:
            bar = bar_lookup.get(bar_id)
            if bar is None:
                continue
            
            is_reentry, reentry_jid, best_overlap = False, None, 0
            for jid, journey in journeys.items():
                if jid in continued or not journey.steps:
                    continue
                # DIMENSION CHECK: re-entry requires same dimension (H0→H0, H1→H1)
                if journey.dimension != bar.dimension:
                    continue
                last = journey.steps[-1]
                if tau - last.tau > config.reentry_lookback:
                    continue
                # FIX: Compare to ANCHOR (τ0) not last step, per Chapter 5
                # Re-entry distances should be computed against the original bar
                anchor_witnesses = journey.anchor_witnesses
                overlap = len(set(bar.witness_tokens) & anchor_witnesses)
                ratio = overlap / max(len(bar.witness_tokens), 1)
                if ratio >= config.reentry_overlap and ratio > best_overlap:
                    is_reentry, reentry_jid, best_overlap = True, jid, ratio
            
            if is_reentry and reentry_jid:
                journeys[reentry_jid].steps.append(JourneyStep(
                    tau=tau, window_id=window['window_id'], bar_id=bar.bar_id,
                    event=EventType.REENTRY, witness_tokens=bar.witness_tokens.copy()
                ))
                bar_to_journey[bar.bar_id] = reentry_jid
                continued.add(reentry_jid)
            else:
                jid = f"journey_{journey_counter}"
                journey_counter += 1
                journey = Journey(journey_id=jid, dimension=bar.dimension)
                journey.steps.append(JourneyStep(
                    tau=tau, window_id=window['window_id'], bar_id=bar.bar_id,
                    event=EventType.SPAWN, witness_tokens=bar.witness_tokens.copy()
                ))
                journeys[jid] = journey
                bar_to_journey[bar.bar_id] = jid
    
    return journeys


# =============================================================================
# STAGE 1: THE GLUING STRUCTURE
# =============================================================================

@dataclass
class GluingEdge:
    """An edge in the gluing graph - two journeys share a witness."""
    journey_a: str
    journey_b: str
    shared_witnesses: Set[str]
    tau: int  # When this gluing is active
    
    @property
    def weight(self) -> int:
        return len(self.shared_witnesses)


@dataclass
class SelfStructure:
    """The Self as a hocolim - journeys glued by shared witnesses."""
    journeys: Dict[str, Journey]
    gluing_edges: List[GluingEdge]
    components: List[Set[str]]  # Connected components of journeys
    hub_tokens: Set[str] = field(default_factory=set)  # Tokens excluded from gluing
    min_shared: int = 2  # Minimum shared witnesses for gluing
    
    @property
    def num_journeys(self) -> int:
        return len(self.journeys)
    
    @property
    def num_components(self) -> int:
        return len(self.components)
    
    @property
    def fragmentation(self) -> float:
        """Fragmentation = components / journeys. Low = unified Self."""
        if self.num_journeys == 0:
            return 1.0
        return self.num_components / self.num_journeys
    
    @property 
    def largest_component_size(self) -> int:
        if not self.components:
            return 0
        return max(len(c) for c in self.components)
    
    @property
    def presence_ratio(self) -> float:
        """What fraction of journeys are in the largest component?"""
        if self.num_journeys == 0:
            return 0.0
        return self.largest_component_size / self.num_journeys


def compute_gluing_at_tau(journeys: Dict[str, Journey], tau: int, min_shared: int = 1, 
                          hub_tokens: Set[str] = None) -> List[GluingEdge]:
    """
    Compute gluing edges at a specific time τ.
    
    Two journeys are glued at τ if they're both active and share witness tokens.
    This is where the hocolim does its work: shared witnesses = identification.
    """
    if hub_tokens is None:
        hub_tokens = set()
    
    # Get active journeys at this τ
    active_journeys = {}
    for jid, journey in journeys.items():
        witnesses = journey.witnesses_at(tau)
        # Filter out hub tokens
        meaningful_witnesses = witnesses - hub_tokens
        if meaningful_witnesses:
            active_journeys[jid] = meaningful_witnesses
    
    # Find shared witnesses between all pairs
    edges = []
    journey_ids = list(active_journeys.keys())
    
    for i in range(len(journey_ids)):
        for j in range(i + 1, len(journey_ids)):
            jid_a, jid_b = journey_ids[i], journey_ids[j]
            shared = active_journeys[jid_a] & active_journeys[jid_b]
            
            if len(shared) >= min_shared:
                edges.append(GluingEdge(
                    journey_a=jid_a,
                    journey_b=jid_b,
                    shared_witnesses=shared,
                    tau=tau
                ))
    
    return edges


def identify_hub_tokens(journeys: Dict[str, Journey], hub_threshold: float = 0.4) -> Set[str]:
    """
    Identify "hub" tokens that appear in too many journeys.
    
    These are like stopwords for gluing - they connect everything to everything
    and thus carry no discriminative information.
    """
    token_journey_count = defaultdict(int)
    
    for journey in journeys.values():
        journey_tokens = journey.all_witnesses()
        for token in journey_tokens:
            token_journey_count[token] += 1
    
    num_journeys = len(journeys)
    hub_tokens = {
        token for token, count in token_journey_count.items()
        if count / num_journeys >= hub_threshold
    }
    
    return hub_tokens


def compute_gluing_cumulative(journeys: Dict[str, Journey], num_windows: int, min_shared: int = 1, 
                              hub_threshold: float = 0.4) -> Tuple[List[GluingEdge], Set[str]]:
    """
    Compute cumulative gluing across all time with DISCRIMINATIVE filtering.
    
    Two journeys are glued if they EVER share MEANINGFUL witnesses at ANY time.
    Hub tokens (appearing in >40% of journeys) are excluded from gluing computation.
    
    Returns: (edges, hub_tokens) so we can report what was filtered
    """
    # First identify hub tokens
    hub_tokens = identify_hub_tokens(journeys, hub_threshold)
    
    # Track all shared witnesses between journey pairs (excluding hubs)
    pair_witnesses: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    pair_taus: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    
    for tau in range(num_windows):
        active_journeys = {}
        for jid, journey in journeys.items():
            witnesses = journey.witnesses_at(tau)
            # Filter out hub tokens
            meaningful_witnesses = witnesses - hub_tokens
            if meaningful_witnesses:
                active_journeys[jid] = meaningful_witnesses
        
        journey_ids = list(active_journeys.keys())
        for i in range(len(journey_ids)):
            for j in range(i + 1, len(journey_ids)):
                jid_a, jid_b = sorted([journey_ids[i], journey_ids[j]])
                shared = active_journeys[jid_a] & active_journeys[jid_b]  # FIX: was journey_ids[j]
                if shared:
                    pair_witnesses[(jid_a, jid_b)].update(shared)
                    pair_taus[(jid_a, jid_b)].append(tau)
    
    edges = []
    for (jid_a, jid_b), witnesses in pair_witnesses.items():
        if len(witnesses) >= min_shared:
            # Use the first tau where they shared
            first_tau = min(pair_taus[(jid_a, jid_b)])
            edges.append(GluingEdge(
                journey_a=jid_a,
                journey_b=jid_b,
                shared_witnesses=witnesses,
                tau=first_tau
            ))
    
    return edges, hub_tokens


def find_connected_components(journeys: Dict[str, Journey], edges: List[GluingEdge]) -> List[Set[str]]:
    """
    Find connected components in the gluing graph.
    
    Each component is a unified region of the Self - journeys that are
    all reachable from each other via shared witnesses.
    """
    # Build adjacency
    adj = defaultdict(set)
    for edge in edges:
        adj[edge.journey_a].add(edge.journey_b)
        adj[edge.journey_b].add(edge.journey_a)
    
    # BFS to find components
    visited = set()
    components = []
    
    for jid in journeys.keys():
        if jid in visited:
            continue
        
        # BFS from this journey
        component = set()
        queue = [jid]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        components.append(component)
    
    # Sort by size descending
    components.sort(key=lambda c: -len(c))
    return components


def build_self_structure(journeys: Dict[str, Journey], num_windows: int, 
                         min_shared: int = 2, hub_threshold: float = 0.4) -> SelfStructure:
    """
    Build the complete Self structure as a hocolim with discriminative gluing.
    
    Args:
        min_shared: Minimum number of non-hub shared witnesses to create a gluing edge
        hub_threshold: Tokens appearing in more than this fraction of journeys are hubs
    """
    edges, hub_tokens = compute_gluing_cumulative(journeys, num_windows, min_shared, hub_threshold)
    components = find_connected_components(journeys, edges)
    
    return SelfStructure(
        journeys=journeys,
        gluing_edges=edges,
        components=components,
        hub_tokens=hub_tokens,
        min_shared=min_shared
    )


# =============================================================================
# STAGE 1 VISUALIZATION
# =============================================================================

def visualize_gluing_ascii(self_struct: SelfStructure, window_ids: List[str], max_journeys: int = 30):
    """
    ASCII visualization of the gluing structure.
    
    Shows:
    - Journey lifelines as horizontal tracks
    - Vertical bars where journeys share witnesses (gluing points)
    - Component membership via color/symbol
    """
    print("\n" + "═" * 90)
    print("SELF AS HOCOLIM - Stage 1: The Gluing Structure")
    print("═" * 90)
    print("\nJourneys are GLUED where they share witness tokens.")
    print("Connected components = unified regions of the Self.\n")
    
    num_windows = len(window_ids)
    journeys = self_struct.journeys
    
    # Assign component IDs to journeys
    journey_to_component = {}
    for comp_idx, component in enumerate(self_struct.components):
        for jid in component:
            journey_to_component[jid] = comp_idx
    
    # Component symbols
    COMP_SYMBOLS = ['●', '◆', '■', '▲', '★', '◉', '◈', '▣', '△', '☆']
    
    # Sort journeys: by component (largest first), then by lifespan
    def journey_sort_key(jid):
        comp_idx = journey_to_component.get(jid, 999)
        lifespan = journeys[jid].lifespan
        return (comp_idx, -lifespan)
    
    sorted_jids = sorted(journeys.keys(), key=journey_sort_key)[:max_journeys]
    
    # Header
    print("    Component │ Journey Signature      │", end="")
    for wid in window_ids:
        month = wid.split("-")[1]
        print(f" {month}", end="")
    print(" │ Glued With")
    print("    ──────────┼────────────────────────┼" + "───" * num_windows + "─┼─────────────")
    
    # Track which journeys each journey is glued with
    glued_with = defaultdict(set)
    for edge in self_struct.gluing_edges:
        glued_with[edge.journey_a].add(edge.journey_b)
        glued_with[edge.journey_b].add(edge.journey_a)
    
    # Draw each journey
    current_component = -1
    for jid in sorted_jids:
        journey = journeys[jid]
        comp_idx = journey_to_component.get(jid, len(COMP_SYMBOLS))
        comp_symbol = COMP_SYMBOLS[comp_idx % len(COMP_SYMBOLS)]
        
        # Component separator
        if comp_idx != current_component:
            if current_component >= 0:
                print("    ──────────┼────────────────────────┼" + "───" * num_windows + "─┼─────────────")
            current_component = comp_idx
        
        # Journey signature
        sig = journey.signature[:20].ljust(20)
        
        # Build timeline
        timeline = ["  "] * num_windows
        step_taus = {s.tau: s for s in journey.steps}
        
        for tau in range(num_windows):
            if tau in step_taus:
                step = step_taus[tau]
                if step.event == EventType.SPAWN:
                    timeline[tau] = f"{comp_symbol} "
                elif step.event == EventType.CARRY:
                    timeline[tau] = "━━"
                elif step.event == EventType.DRIFT:
                    timeline[tau] = "≈≈"
                elif step.event == EventType.REENTRY:
                    timeline[tau] = "↺ "
        
        # Fill gaps
        active = False
        for tau in range(num_windows):
            if tau in step_taus:
                active = True
            elif active:
                next_step = next((s for s in journey.steps if s.tau > tau), None)
                if next_step:
                    timeline[tau] = "──"
                else:
                    active = False
        
        timeline_str = "".join(timeline)
        
        # Glued with count
        glued_count = len(glued_with[jid])
        glued_str = f"{glued_count:3d} others" if glued_count > 0 else "isolated"
        
        print(f"    {comp_symbol} comp{comp_idx:02d}  │ {sig} │ {timeline_str} │ {glued_str}")
    
    # Summary
    print("\n" + "─" * 90)
    print("GLUING SUMMARY")
    print("─" * 90)
    print(f"  Total journeys:      {self_struct.num_journeys}")
    print(f"  Gluing edges:        {len(self_struct.gluing_edges)}")
    print(f"  Connected components: {self_struct.num_components}")
    print(f"  Fragmentation:       {self_struct.fragmentation:.3f} (lower = more unified)")
    print(f"  Presence ratio:      {self_struct.presence_ratio:.3f} (fraction in largest component)")
    print(f"  Largest component:   {self_struct.largest_component_size} journeys")
    
    # Hub token info
    if self_struct.hub_tokens:
        hub_list = sorted(self_struct.hub_tokens)[:15]
        print(f"\n  Hub tokens EXCLUDED from gluing (appear in >{40}% of journeys):")
        print(f"    {', '.join(hub_list)}")
        if len(self_struct.hub_tokens) > 15:
            print(f"    ... and {len(self_struct.hub_tokens) - 15} more")
    print(f"\n  Gluing requires ≥{self_struct.min_shared} shared non-hub witnesses")


def visualize_gluing_graph(self_struct: SelfStructure, max_edges: int = 50):
    """
    Show the gluing graph as an edge list with shared witnesses.
    """
    print("\n" + "═" * 90)
    print("GLUING GRAPH (Top edges by shared witnesses)")
    print("═" * 90)
    
    # Sort edges by weight
    sorted_edges = sorted(self_struct.gluing_edges, key=lambda e: -e.weight)[:max_edges]
    
    if not sorted_edges:
        print("\n  No gluing edges found. Journeys are completely disconnected.")
        return
    
    print("\n  Journey A              ←──[shared witnesses]──→  Journey B")
    print("  " + "─" * 80)
    
    for edge in sorted_edges:
        sig_a = self_struct.journeys[edge.journey_a].signature[:18].ljust(18)
        sig_b = self_struct.journeys[edge.journey_b].signature[:18].ljust(18)
        witnesses = ", ".join(sorted(edge.shared_witnesses)[:5])
        if len(edge.shared_witnesses) > 5:
            witnesses += f"... (+{len(edge.shared_witnesses)-5})"
        
        print(f"  {sig_a}  ←─[{witnesses}]─→  {sig_b}")


def visualize_components_summary(self_struct: SelfStructure, top_k: int = 10):
    """
    Summarize the connected components of the Self.
    """
    print("\n" + "═" * 90)
    print("CONNECTED COMPONENTS (Unified Regions of the Self)")
    print("═" * 90)
    
    for idx, component in enumerate(self_struct.components[:top_k]):
        # Get all witnesses in this component
        all_witnesses = set()
        for jid in component:
            all_witnesses.update(self_struct.journeys[jid].all_witnesses())
        
        # Get top witnesses by frequency
        witness_counts = Counter()
        for jid in component:
            for step in self_struct.journeys[jid].steps:
                witness_counts.update(step.witness_tokens)
        
        top_witnesses = [w for w, _ in witness_counts.most_common(8)]
        
        COMP_SYMBOLS = ['●', '◆', '■', '▲', '★', '◉', '◈', '▣', '△', '☆']
        symbol = COMP_SYMBOLS[idx % len(COMP_SYMBOLS)]
        
        print(f"\n  {symbol} Component {idx}: {len(component)} journeys")
        print(f"    Core witnesses: {', '.join(top_witnesses)}")
        
        # Sample journeys
        sample_jids = list(component)[:3]
        for jid in sample_jids:
            sig = self_struct.journeys[jid].signature[:30]
            print(f"      • {sig}")
        if len(component) > 3:
            print(f"      ... and {len(component) - 3} more")
    
    if len(self_struct.components) > top_k:
        print(f"\n  ... and {len(self_struct.components) - top_k} smaller components")


def visualize_presence_heatmap(self_struct: SelfStructure, window_ids: List[str]) -> List[dict]:
    """
    Show presence (inhabitation) over time.
    
    At each τ, how much of the Self can be "reached" from the largest component?
    
    Returns: List of presence state dicts for export
    """
    print("\n" + "═" * 90)
    print("PRESENCE OVER TIME (How unified is the Self at each τ?)")
    print("═" * 90)
    
    num_windows = len(window_ids)
    presence_states = []
    
    print("\n  τ   Window   Active   Components   Fragmentation   Presence")
    print("  " + "─" * 65)
    
    for tau in range(num_windows):
        # Compute structure at this τ with same hub filtering
        edges_at_tau = compute_gluing_at_tau(
            self_struct.journeys, tau, 
            min_shared=self_struct.min_shared,
            hub_tokens=self_struct.hub_tokens
        )
        
        # Count active journeys
        active = sum(1 for j in self_struct.journeys.values() 
                    if any(s.tau == tau for s in j.steps))
        
        if active == 0:
            continue
        
        # Find components at this τ
        active_journeys = {jid: j for jid, j in self_struct.journeys.items()
                         if any(s.tau == tau for s in j.steps)}
        components = find_connected_components(active_journeys, edges_at_tau)
        
        num_comp = len(components)
        frag = num_comp / active if active > 0 else 1.0
        largest = max(len(c) for c in components) if components else 0
        presence = largest / active if active > 0 else 0.0
        
        # Count events at this tau
        carries = drifts = spawns = ruptures = reentries = 0
        for j in self_struct.journeys.values():
            for step in j.steps:
                if step.tau == tau:
                    if step.event == EventType.CARRY:
                        carries += 1
                    elif step.event == EventType.DRIFT:
                        drifts += 1
                    elif step.event == EventType.SPAWN:
                        spawns += 1
                    elif step.event == EventType.REENTRY:
                        reentries += 1
        # Ruptures: active at tau-1 but not at tau
        if tau > 0:
            for j in self_struct.journeys.values():
                was_active = any(s.tau == tau - 1 for s in j.steps)
                is_active = any(s.tau == tau for s in j.steps)
                if was_active and not is_active:
                    ruptures += 1
        
        # Store state for export
        presence_states.append({
            'tau': tau,
            'window_id': window_ids[tau],
            'active': active,
            'components': num_comp,
            'fragmentation': frag,
            'presence': presence,
            'largest_component': largest,
            'carries': carries,
            'drifts': drifts,
            'spawns': spawns,
            'ruptures': ruptures,
            'reentries': reentries
        })
        
        # Visual bar
        bar_width = 30
        filled = int(presence * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        status = "★ UNIFIED" if presence > 0.7 else ("◐ PARTIAL" if presence > 0.3 else "○ FRAGMENTED")
        
        print(f"  {tau:2d}  {window_ids[tau]:>7}  {active:5d}   {num_comp:5d}        {frag:.3f}          {bar} {presence:.2f} {status}")
    
    return presence_states


# =============================================================================
# Main Pipeline
# =============================================================================

def analyze_window(window_id, tau, conversations, vocabulary, config):
    print(f"  [{tau}] {window_id}: {len(conversations)} convs", end="", flush=True)
    tokens, frequencies = sample_tokens_from_vocabulary(
        conversations, vocabulary, config.tokens_per_window,
        filter_technical=config.filter_technical
    )
    print(f" → {len(tokens)} tokens", end="", flush=True)
    
    if len(tokens) < 50:
        print(" [SKIPPED]")
        return {'window_id': window_id, 'tau': tau, 'bars': []}
    
    t0 = time.time()
    embeddings = embed_tokens(tokens, config.embedding_model)
    print(f" → embed ({time.time()-t0:.1f}s)", end="", flush=True)
    
    t0 = time.time()
    persistence = compute_persistence(embeddings, config.max_edge_length)
    print(f" → PH ({time.time()-t0:.1f}s)", end="", flush=True)
    
    bars = construct_witnessed_bars(persistence, embeddings, tokens, frequencies, window_id, config)
    print(f" → {len(bars)} bars")
    
    return {'window_id': window_id, 'tau': tau, 'bars': bars}


def run_analysis(conversations, config, start_from=None, test_mode=False):
    print("\nCreating monthly windows...")
    all_windows = create_monthly_windows(conversations)
    window_ids = list(all_windows.keys())
    print(f"  {len(window_ids)} windows: {window_ids[0]} to {window_ids[-1]}")
    
    if start_from:
        window_ids = [w for w in window_ids if w >= start_from]
        print(f"  Starting from {start_from}: {len(window_ids)} windows")
    
    if test_mode:
        window_ids = window_ids[:8]
        print(f"  TEST MODE: {len(window_ids)} windows")
    
    vocabulary = build_global_vocabulary(
        {wid: all_windows[wid] for wid in window_ids}, 
        min_window_frequency=2,
        filter_technical=config.filter_technical
    )
    
    print("\nAnalyzing windows...")
    window_analyses = []
    for tau, wid in enumerate(window_ids):
        w = analyze_window(wid, tau, all_windows[wid], vocabulary, config)
        window_analyses.append(w)
    
    print("\nMatching bars...")
    all_matches, all_unmatched = [], []
    for i in range(len(window_analyses) - 1):
        w1, w2 = window_analyses[i], window_analyses[i + 1]
        matches, um1, um2 = match_bars(w1['bars'], w2['bars'], config)
        all_matches.append(matches)
        all_unmatched.append((um1, um2))
        carries = sum(1 for m in matches if m.event_type == EventType.CARRY)
        print(f"  {w1['window_id']}→{w2['window_id']}: {len(matches)} ({carries}C)")
    
    print("\nBuilding journeys...")
    journeys = build_journeys(window_analyses, all_matches, all_unmatched, config)
    print(f"  {len(journeys)} journeys")
    
    return journeys, window_ids


def main():
    parser = argparse.ArgumentParser(description="Self as Hocolim - Stage 1: Gluing Structure")
    parser.add_argument("input", help="Path to cassie_parsed.json")
    parser.add_argument("--output", "-o", default="results/self_hocolim")
    parser.add_argument("--start-from", help="Start from this window (e.g., 2024-04)")
    parser.add_argument("--test", action="store_true", help="Test mode (8 windows)")
    parser.add_argument("--tokens-per-window", type=int, default=500)
    parser.add_argument("--min-shared", type=int, default=2, 
                       help="Minimum non-hub shared witnesses for gluing (default: 2)")
    parser.add_argument("--hub-threshold", type=float, default=0.4,
                       help="Tokens in >X%% of journeys are hubs (default: 0.4)")
    parser.add_argument("--include-technical", action="store_true",
                       help="Include LaTeX/code tokens as potential witnesses (default: filtered)")
    parser.add_argument("--export-viz", action="store_true",
                       help="Export HTML/SVG/CSV visualizations")
    
    args = parser.parse_args()
    
    config = Config(
        tokens_per_window=args.tokens_per_window, 
        output_dir=args.output,
        filter_technical=not args.include_technical  # Default: filter technical tokens
    )
    conversations = load_conversations(args.input)
    
    # Run core analysis
    journeys, window_ids = run_analysis(
        conversations, config, 
        start_from=args.start_from, 
        test_mode=args.test
    )
    
    # Build Self structure with discriminative gluing
    print("\n" + "=" * 90)
    print("BUILDING SELF AS HOCOLIM")
    print(f"  Hub threshold: {args.hub_threshold} (tokens in >{args.hub_threshold*100:.0f}% journeys excluded)")
    print(f"  Min shared witnesses: {args.min_shared}")
    print(f"  Technical filtering: {'OFF' if args.include_technical else 'ON'}")
    print("=" * 90)
    
    self_struct = build_self_structure(
        journeys, len(window_ids),
        min_shared=args.min_shared,
        hub_threshold=args.hub_threshold
    )
    
    # Visualizations
    visualize_gluing_ascii(self_struct, window_ids)
    visualize_components_summary(self_struct)
    visualize_gluing_graph(self_struct)
    presence_states = visualize_presence_heatmap(self_struct, window_ids)
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Export Self structure
    self_data = {
        'period': f"{window_ids[0]} to {window_ids[-1]}",
        'num_journeys': self_struct.num_journeys,
        'num_components': self_struct.num_components,
        'fragmentation': self_struct.fragmentation,
        'presence_ratio': self_struct.presence_ratio,
        'largest_component_size': self_struct.largest_component_size,
        'hub_tokens': list(self_struct.hub_tokens),
        'min_shared': self_struct.min_shared,
        'presence_states': presence_states,
        'components': [
            {
                'id': idx,
                'size': len(comp),
                'journeys': list(comp),
                'signatures': [journeys[jid].signature for jid in list(comp)[:10] if jid in journeys]
            }
            for idx, comp in enumerate(self_struct.components[:20])
        ],
        'gluing_edges': [
            {
                'journey_a': e.journey_a,
                'journey_b': e.journey_b,
                'shared_witnesses': list(e.shared_witnesses),
                'tau': e.tau
            }
            for e in self_struct.gluing_edges[:500]
        ],
        'journeys': [
            {
                'id': jid,
                'signature': j.signature,
                'lifespan': j.lifespan,
                'has_reentry': j.has_reentry,
                'all_witnesses': list(j.all_witnesses())[:20],
                'steps': [
                    {'tau': s.tau, 'event': s.event.value, 'witnesses': s.witness_tokens[:8]}
                    for s in j.steps
                ]
            }
            for jid, j in list(journeys.items())[:200]
        ]
    }
    
    with open(os.path.join(config.output_dir, "self_structure.json"), 'w') as f:
        json.dump(self_data, f, indent=2)
    
    print(f"\n✓ Results saved to {config.output_dir}/")
    print(f"  - self_structure.json: Complete Self structure with gluing data")
    
    # Export visualizations if requested
    if args.export_viz:
        try:
            from visualize_export import export_from_analysis
            export_from_analysis(self_struct, None, window_ids, config.output_dir, presence_states)
        except ImportError:
            print("\n  Note: visualize_export.py not found, skipping HTML/CSV export")
            print("  Run: python scripts/visualize_export.py results/self_hocolim/")


if __name__ == "__main__":
    main()