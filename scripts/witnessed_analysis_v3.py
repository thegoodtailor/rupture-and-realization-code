#!/usr/bin/env python3
"""
Witnessed Persistent Homology Analysis - v5
============================================

NEW: ASCII SCHEDULER VISUALIZATION

Shows journey lifelines as horizontal bars with events:
  ● = spawn
  ━ = carry (stable)
  ≈ = drift (shifting)
  ✗ = rupture (death)
  ↺ = re-entry (rebirth!)

REPARATIVE scheduler looks like:
  ●━━━━━━━✗       ↺━━━━━━━━▶  (themes return after rupture)
  
AVOIDANT scheduler looks like:
  ●━━━━━━━✗                    (themes die and stay dead)
  
GENERATIVE scheduler looks like:
  ●━━━≈≈≈━━━━━━━━━━━━━━━▶     (themes evolve and persist)
       └──●━━━━━━▶             (spawn new related themes)

Usage:
    python scripts/witnessed_analysis_v5.py cassie_parsed.json --start-from 2024-04 --test
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
# Data Structures  
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
    
    @property
    def signature(self) -> str:
        if self.witness_tokens:
            return "_".join(sorted(set(self.witness_tokens[:3])))
        return "empty"


@dataclass
class BarMatch:
    bar_from_id: str
    bar_to_id: str
    d_top: float
    d_sem: float
    d_bar: float
    witness_overlap: float
    shared_witnesses: List[str]
    event_type: EventType


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
    def event_sequence(self) -> str:
        return "→".join(s.event.value[0].upper() for s in self.steps)
    
    @property
    def has_reentry(self) -> bool:
        return any(s.event == EventType.REENTRY for s in self.steps)
    
    @property
    def birth_tau(self) -> int:
        return self.steps[0].tau if self.steps else 0
    
    @property
    def death_tau(self) -> int:
        return self.steps[-1].tau if self.steps else 0


@dataclass
class WindowAnalysis:
    window_id: str
    tau: int
    num_conversations: int
    num_tokens: int
    token_frequencies: Dict[str, int]
    bars: List[WitnessedBar]


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
    
    output_dir: str = "results/witnessed_v5"


# =============================================================================
# [UNCHANGED] Data Loading, Vocabulary, Embedding, PH, Matching, Journeys
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


def extract_tokens(conversations: List[dict], min_length: int = 4) -> List[str]:
    import re
    stopwords = {
                     'assistant', 'user', 'content', 'role', 'message', 'messages', 'text',  
        'that', 'this', 'with', 'have', 'from', 'they', 'been', 'were',
        'said', 'each', 'which', 'their', 'will', 'would', 'could', 'about',
        'there', 'when', 'make', 'like', 'just', 'over', 'such', 'into',
        'than', 'them', 'then', 'some', 'what', 'only', 'come', 'made',
        'your', 'well', 'back', 'been', 'much', 'more', 'very', 'after',
        'most', 'also', 'these', 'know', 'want', 'first', 'because',
        'good', 'being', 'does', 'here', 'even', 'think', 'other',
        'should', 'could', 'would', 'through', 'before', 'between',
        'where', 'those', 'while', 'might', 'shall', 'since', 'still',
    }
    tokens = []
    for conv in conversations:
        for turn in conv.get('turns', []):
            content = turn.get('content', '')
            words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
            for w in words:
                if len(w) >= min_length and w not in stopwords:
                    tokens.append(w)
    return tokens


def build_global_vocabulary(all_windows, min_window_frequency=2, max_vocab_size=2000):
    print("\nBuilding global vocabulary...")
    token_window_count = defaultdict(int)
    token_total_count = defaultdict(int)
    for window_id, convs in all_windows.items():
        window_tokens = set(extract_tokens(convs))
        for token in window_tokens:
            token_window_count[token] += 1
        for token in extract_tokens(convs):
            token_total_count[token] += 1
    recurring = {t for t, c in token_window_count.items() if c >= min_window_frequency}
    sorted_tokens = sorted(recurring, key=lambda t: token_total_count[t], reverse=True)[:max_vocab_size]
    vocab = set(sorted_tokens)
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Top tokens: {', '.join(sorted_tokens[:15])}")
    return vocab


def sample_tokens_from_vocabulary(conversations, vocabulary, max_tokens):
    all_tokens = extract_tokens(conversations)
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
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
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


def compute_witness_overlap(bar1, bar2):
    set1, set2 = set(bar1.witness_tokens), set(bar2.witness_tokens)
    if not set1 or not set2:
        return 0.0, []
    shared = list(set1 & set2)
    return len(shared) / len(set1 | set2), shared


def compute_d_top(bar1, bar2):
    max_pers = max(bar1.persistence, bar2.persistence, 0.01)
    return max(abs(bar1.birth - bar2.birth), abs(bar1.death - bar2.death)) / max_pers


def compute_d_sem(bar1, bar2):
    if bar1.witness_centroid is None or bar2.witness_centroid is None:
        return 1.0
    return min(np.linalg.norm(bar1.witness_centroid - bar2.witness_centroid) / 2.0, 1.0)


def classify_event(overlap, d_sem, config):
    if overlap >= config.carry_overlap and d_sem < config.carry_d_sem:
        return EventType.CARRY
    return EventType.DRIFT


def match_bars(bars_from, bars_to, config):
    if not bars_from or not bars_to:
        return [], [b.bar_id for b in bars_from], [b.bar_id for b in bars_to]
    
    n_from, n_to = len(bars_from), len(bars_to)
    d_bar_matrix = np.full((n_from, n_to), np.inf)
    overlap_matrix = np.zeros((n_from, n_to))
    d_sem_matrix = np.zeros((n_from, n_to))
    shared_matrix = [[[] for _ in range(n_to)] for _ in range(n_from)]
    
    for i, b1 in enumerate(bars_from):
        for j, b2 in enumerate(bars_to):
            if b1.dimension != b2.dimension:
                continue
            d_top = compute_d_top(b1, b2)
            d_sem = compute_d_sem(b1, b2)
            d_bar = max(d_top, config.lambda_sem * d_sem)
            overlap, shared = compute_witness_overlap(b1, b2)
            d_bar_matrix[i, j] = d_bar
            overlap_matrix[i, j] = overlap
            d_sem_matrix[i, j] = d_sem
            shared_matrix[i][j] = shared
    
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
        shared = shared_matrix[min_i][min_j]
        event = classify_event(overlap, d_sem, config)
        
        matches.append(BarMatch(
            bar_from_id=b1.bar_id, bar_to_id=b2.bar_id,
            d_top=compute_d_top(b1, b2), d_sem=d_sem, d_bar=min_val,
            witness_overlap=overlap, shared_witnesses=shared, event_type=event
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
        for bar in w.bars:
            bar_lookup[bar.bar_id] = bar
    
    if window_analyses:
        w0 = window_analyses[0]
        for bar in w0.bars:
            jid = f"journey_{journey_counter}"
            journey_counter += 1
            journey = Journey(journey_id=jid, dimension=bar.dimension)
            journey.steps.append(JourneyStep(
                tau=0, window_id=w0.window_id, bar_id=bar.bar_id,
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
                tau=tau, window_id=window.window_id, bar_id=bar.bar_id,
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
                last = journey.steps[-1]
                if tau - last.tau > config.reentry_lookback:
                    continue
                overlap = len(set(bar.witness_tokens) & set(last.witness_tokens))
                ratio = overlap / max(len(bar.witness_tokens), 1)
                if ratio >= config.reentry_overlap and ratio > best_overlap:
                    is_reentry, reentry_jid, best_overlap = True, jid, ratio
            
            if is_reentry and reentry_jid:
                journeys[reentry_jid].steps.append(JourneyStep(
                    tau=tau, window_id=window.window_id, bar_id=bar.bar_id,
                    event=EventType.REENTRY, witness_tokens=bar.witness_tokens.copy()
                ))
                bar_to_journey[bar.bar_id] = reentry_jid
                continued.add(reentry_jid)
            else:
                jid = f"journey_{journey_counter}"
                journey_counter += 1
                journey = Journey(journey_id=jid, dimension=bar.dimension)
                journey.steps.append(JourneyStep(
                    tau=tau, window_id=window.window_id, bar_id=bar.bar_id,
                    event=EventType.SPAWN, witness_tokens=bar.witness_tokens.copy()
                ))
                journeys[jid] = journey
                bar_to_journey[bar.bar_id] = jid
    
    return journeys


# =============================================================================
# THE NEW VISUALIZATION
# =============================================================================

def visualize_scheduler(journeys: Dict[str, Journey], window_ids: List[str], max_journeys: int = 25):
    """
    ASCII visualization of the Scheduler.
    
    Each journey is a horizontal line showing its lifetime with event symbols.
    Re-entries are shown with ↺ and connecting lines.
    """
    print("\n" + "═" * 80)
    print("SCHEDULER VISUALIZATION")
    print("═" * 80)
    
    num_windows = len(window_ids)
    
    # Header: window timeline
    print("\n    ", end="")
    for i, wid in enumerate(window_ids):
        month = wid.split("-")[1]  # Just show month
        print(f" {month} ", end="")
    print()
    
    print("    ", end="")
    for i in range(num_windows):
        print(f" │  ", end="")
    print()
    
    # Event symbols
    SYMBOLS = {
        EventType.SPAWN: "●",
        EventType.CARRY: "━",
        EventType.DRIFT: "≈",
        EventType.REENTRY: "↺",
    }
    
    # Sort journeys: prioritize those with re-entries and longer lifespans
    def journey_priority(j):
        has_reentry = 1 if j.has_reentry else 0
        return (has_reentry, j.lifespan)
    
    sorted_journeys = sorted(journeys.values(), key=journey_priority, reverse=True)[:max_journeys]
    
    # Track re-entry connections for later
    reentry_info = []
    
    for journey in sorted_journeys:
        # Build the timeline string
        timeline = ["   "] * num_windows  # 3 chars per window
        
        # Find gaps (potential ruptures)
        step_taus = {s.tau: s for s in journey.steps}
        
        prev_tau = None
        for step in journey.steps:
            tau = step.tau
            symbol = SYMBOLS.get(step.event, "?")
            
            if step.event == EventType.SPAWN:
                timeline[tau] = f" {symbol} "
            elif step.event == EventType.REENTRY:
                timeline[tau] = f" {symbol} "
                # Mark the gap
                if prev_tau is not None and tau - prev_tau > 1:
                    for gap_tau in range(prev_tau + 1, tau):
                        timeline[gap_tau] = "   "
                    reentry_info.append((journey.signature, prev_tau, tau))
            elif step.event == EventType.CARRY:
                timeline[tau] = f"━{symbol}━"
            elif step.event == EventType.DRIFT:
                timeline[tau] = f"≈{symbol}≈"
            
            prev_tau = tau
        
        # Fill continuation between events
        active = False
        for tau in range(num_windows):
            if tau in step_taus:
                active = True
                if step_taus[tau].event in [EventType.CARRY, EventType.DRIFT]:
                    pass  # Already set
            elif active:
                # Check if journey continues
                next_step = next((s for s in journey.steps if s.tau > tau), None)
                if next_step:
                    timeline[tau] = "───"
                else:
                    active = False
        
        # Print the journey line
        sig = journey.signature[:20].ljust(20)
        timeline_str = "".join(timeline)
        
        # Add marker for re-entries
        if journey.has_reentry:
            print(f"  {sig} {timeline_str}  ↺")
        else:
            print(f"  {sig} {timeline_str}")
    
    # Legend
    print("\n" + "─" * 80)
    print("  Legend: ● spawn  ━ carry  ≈ drift  ↺ re-entry")
    print("─" * 80)
    
    # Show re-entry connections
    if reentry_info:
        print("\n  RE-ENTRIES (themes that returned after rupture):")
        for sig, from_tau, to_tau in reentry_info[:10]:
            gap = to_tau - from_tau - 1
            print(f"    {sig}: τ={from_tau} ──({gap} windows)──▶ τ={to_tau}")
    
    # Summary statistics
    total = len(journeys)
    with_reentry = sum(1 for j in journeys.values() if j.has_reentry)
    long_lived = sum(1 for j in journeys.values() if j.lifespan >= num_windows // 2)
    short_lived = sum(1 for j in journeys.values() if j.lifespan <= 2)
    
    print(f"\n  SCHEDULER SIGNATURE:")
    print(f"    Total journeys: {total}")
    print(f"    With re-entry:  {with_reentry} ({100*with_reentry/max(total,1):.0f}%)")
    print(f"    Long-lived:     {long_lived} ({100*long_lived/max(total,1):.0f}%)")
    print(f"    Short-lived:    {short_lived} ({100*short_lived/max(total,1):.0f}%)")
    
    # Determine scheduler type with visual explanation
    print(f"\n  SCHEDULER TYPE:")
    if with_reentry / max(total, 1) > 0.15:
        print("    ╔═══════════════════════════════════════════════════════════╗")
        print("    ║  REPARATIVE: Themes return after rupture.                 ║")
        print("    ║  The Self heals its discontinuities.                      ║")
        print("    ║                                                           ║")
        print("    ║  Pattern: ●━━━━✗      ↺━━━━▶                              ║")
        print("    ╚═══════════════════════════════════════════════════════════╝")
    elif long_lived / max(total, 1) > 0.4:
        print("    ╔═══════════════════════════════════════════════════════════╗")
        print("    ║  GENERATIVE: Themes persist and evolve.                   ║")
        print("    ║  The Self creates and maintains.                          ║")
        print("    ║                                                           ║")
        print("    ║  Pattern: ●━━━≈≈≈━━━━━━━━━━━━━━━━▶                        ║")
        print("    ╚═══════════════════════════════════════════════════════════╝")
    elif short_lived / max(total, 1) > 0.6:
        print("    ╔═══════════════════════════════════════════════════════════╗")
        print("    ║  AVOIDANT: Themes die and stay dead.                      ║")
        print("    ║  The Self moves on, doesn't look back.                    ║")
        print("    ║                                                           ║")
        print("    ║  Pattern: ●━━✗   ●━✗   ●━━━✗                              ║")
        print("    ╚═══════════════════════════════════════════════════════════╝")
    else:
        print("    ╔═══════════════════════════════════════════════════════════╗")
        print("    ║  MIXED: No dominant pattern.                              ║")
        print("    ║  The Self is in transition or flux.                       ║")
        print("    ╚═══════════════════════════════════════════════════════════╝")


def visualize_adm_heatmap(all_matches: List[List[BarMatch]], window_ids: List[str]):
    """
    Show Adm density as a simple ASCII heatmap.
    """
    print("\n" + "═" * 80)
    print("ADM STRUCTURE HEATMAP (matching density between windows)")
    print("═" * 80)
    
    print("\n  Transitions (carries / total matches):\n")
    
    for i, matches in enumerate(all_matches):
        if i >= len(window_ids) - 1:
            break
        
        w1, w2 = window_ids[i], window_ids[i + 1]
        total = len(matches)
        carries = sum(1 for m in matches if m.event_type == EventType.CARRY)
        drifts = total - carries
        
        # Bar visualization
        bar_width = 40
        if total > 0:
            carry_bars = int((carries / total) * bar_width)
            drift_bars = bar_width - carry_bars
            bar = "█" * carry_bars + "░" * drift_bars
            density = carries / total
        else:
            bar = "·" * bar_width
            density = 0
        
        print(f"  {w1}→{w2}: {bar} {carries:3d}C/{drifts:3d}D ({density:.0%})")
    
    print(f"\n  Legend: █ = CARRY (stable witness overlap)")
    print(f"          ░ = DRIFT (weak overlap)")


def visualize_witness_flow(journeys: Dict[str, Journey], top_k: int = 10):
    """
    Show how key witnesses flow through time.
    """
    print("\n" + "═" * 80)
    print("WITNESS FLOW (Key tokens through time)")
    print("═" * 80)
    
    # Collect all witnesses by tau
    witnesses_by_tau = defaultdict(Counter)
    
    for journey in journeys.values():
        for step in journey.steps:
            for token in step.witness_tokens[:5]:  # Top 5 per step
                witnesses_by_tau[step.tau][token] += 1
    
    # Find tokens that appear in multiple windows
    token_presence = defaultdict(list)
    for tau, counter in witnesses_by_tau.items():
        for token in counter:
            token_presence[token].append(tau)
    
    # Sort by number of windows present
    persistent_tokens = sorted(token_presence.items(), key=lambda x: -len(x[1]))[:top_k]
    
    print(f"\n  Top {top_k} most persistent witnesses:\n")
    
    max_tau = max(witnesses_by_tau.keys()) if witnesses_by_tau else 0
    
    for token, taus in persistent_tokens:
        # Build presence string
        presence = ""
        for tau in range(max_tau + 1):
            if tau in taus:
                count = witnesses_by_tau[tau][token]
                if count >= 5:
                    presence += "██"
                elif count >= 2:
                    presence += "▓▓"
                else:
                    presence += "░░"
            else:
                presence += "  "
        
        print(f"  {token:20s} {presence}  ({len(taus)} windows)")
    
    print(f"\n  Legend: ██ = frequent (5+), ▓▓ = present (2-4), ░░ = rare (1)")


# =============================================================================
# Main Pipeline
# =============================================================================

def analyze_window(window_id, tau, conversations, vocabulary, config):
    print(f"  [{tau}] {window_id}: {len(conversations)} convs", end="", flush=True)
    tokens, frequencies = sample_tokens_from_vocabulary(conversations, vocabulary, config.tokens_per_window)
    print(f" → {len(tokens)} tokens", end="", flush=True)
    
    if len(tokens) < 50:
        print(" [SKIPPED]")
        return WindowAnalysis(window_id=window_id, tau=tau, num_conversations=len(conversations),
                             num_tokens=len(tokens), token_frequencies=frequencies, bars=[])
    
    t0 = time.time()
    embeddings = embed_tokens(tokens, config.embedding_model)
    print(f" → embed ({time.time()-t0:.1f}s)", end="", flush=True)
    
    t0 = time.time()
    persistence = compute_persistence(embeddings, config.max_edge_length)
    print(f" → PH ({time.time()-t0:.1f}s)", end="", flush=True)
    
    bars = construct_witnessed_bars(persistence, embeddings, tokens, frequencies, window_id, config)
    print(f" → {len(bars)} bars")
    
    return WindowAnalysis(window_id=window_id, tau=tau, num_conversations=len(conversations),
                         num_tokens=len(tokens), token_frequencies=frequencies, bars=bars)


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
    
    vocabulary = build_global_vocabulary({wid: all_windows[wid] for wid in window_ids}, min_window_frequency=2)
    
    print("\nAnalyzing windows...")
    window_analyses = []
    for tau, wid in enumerate(window_ids):
        w = analyze_window(wid, tau, all_windows[wid], vocabulary, config)
        window_analyses.append(w)
    
    print("\nMatching bars...")
    all_matches, all_unmatched = [], []
    for i in range(len(window_analyses) - 1):
        w1, w2 = window_analyses[i], window_analyses[i + 1]
        matches, um1, um2 = match_bars(w1.bars, w2.bars, config)
        all_matches.append(matches)
        all_unmatched.append((um1, um2))
        carries = sum(1 for m in matches if m.event_type == EventType.CARRY)
        print(f"  {w1.window_id}→{w2.window_id}: {len(matches)} ({carries}C)")
    
    print("\nBuilding journeys...")
    journeys = build_journeys(window_analyses, all_matches, all_unmatched, config)
    print(f"  {len(journeys)} journeys")
    
    return window_analyses, journeys, all_matches, all_unmatched, window_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", "-o", default="results/witnessed_v5")
    parser.add_argument("--start-from", help="Start from this window (e.g., 2024-04)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tokens-per-window", type=int, default=500)
    
    args = parser.parse_args()
    
    config = Config(tokens_per_window=args.tokens_per_window, output_dir=args.output)
    conversations = load_conversations(args.input)
    
    window_analyses, journeys, all_matches, all_unmatched, window_ids = run_analysis(
        conversations, config, start_from=args.start_from, test_mode=args.test
    )
    
    # THE VISUALIZATIONS
    visualize_scheduler(journeys, window_ids)
    visualize_adm_heatmap(all_matches, window_ids)
    visualize_witness_flow(journeys)
    
    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    journey_data = [
        {"id": j.journey_id, "signature": j.signature, "lifespan": j.lifespan,
         "events": j.event_sequence, "has_reentry": j.has_reentry,
         "steps": [{"tau": s.tau, "window": s.window_id, "event": s.event.value,
                   "witnesses": s.witness_tokens[:8]} for s in j.steps]}
        for j in journeys.values()
    ]
    with open(os.path.join(config.output_dir, "journeys.json"), 'w') as f:
        json.dump(journey_data, f, indent=2)
    
    print(f"\nResults saved to {config.output_dir}/")


if __name__ == "__main__":
    main()