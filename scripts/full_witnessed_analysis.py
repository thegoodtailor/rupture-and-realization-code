#!/usr/bin/env python3
"""
Full Witnessed Persistent Homology Analysis for Cassie Corpus
==============================================================

CORRECTED VERSION - fixes:
1. Bar IDs now namespaced by window (window_id + bar_index)
2. Witness construction uses k-nearest neighbors, not distance threshold
3. Journey tracking properly follows bars across windows
4. Added sanity checks and debug output

Usage:
    python scripts/full_witnessed_analysis.py cassie_parsed.json --test --tokens-per-window 300
"""

import json
import argparse
import os
import time
from datetime import datetime
from collections import defaultdict
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
    """A bar with its witness tokens."""
    bar_id: str  # Now includes window: "2023-02_bar_5"
    window_id: str
    dimension: int
    birth: float
    death: float
    persistence: float
    witness_tokens: List[str]
    witness_centroid: Optional[np.ndarray] = None
    
    @property
    def signature(self) -> str:
        """Top 3 witness tokens as signature."""
        if self.witness_tokens:
            return "_".join(sorted(self.witness_tokens[:3]))
        return "empty"


@dataclass
class BarMatch:
    """Match between bars in consecutive windows."""
    bar_from_id: str
    bar_to_id: str
    d_top: float
    d_sem: float
    d_bar: float
    witness_overlap: float  # Jaccard of witness tokens
    event_type: EventType


@dataclass
class JourneyStep:
    """One step in a journey."""
    tau: int
    window_id: str
    bar_id: str
    event: EventType
    witness_tokens: List[str]


@dataclass
class Journey:
    """A theme's journey through time."""
    journey_id: str
    dimension: int
    steps: List[JourneyStep] = field(default_factory=list)
    
    @property
    def lifespan(self) -> int:
        return len(self.steps)
    
    @property
    def signature(self) -> str:
        if self.steps and self.steps[0].witness_tokens:
            return "_".join(sorted(self.steps[0].witness_tokens[:3]))
        return "empty"
    
    @property
    def birth_window(self) -> str:
        return self.steps[0].window_id if self.steps else "unknown"
    
    @property
    def event_sequence(self) -> str:
        return "→".join(s.event.value[0].upper() for s in self.steps)
    
    def count_event(self, event_type: EventType) -> int:
        return sum(1 for s in self.steps if s.event == event_type)


@dataclass
class WindowAnalysis:
    """Analysis for one time window."""
    window_id: str
    tau: int
    num_conversations: int
    num_tokens: int
    bars: List[WitnessedBar]
    timing_embed: float = 0.0
    timing_ph: float = 0.0


@dataclass
class AnalysisConfig:
    """Configuration."""
    tokens_per_window: int = 500
    embedding_model: str = "microsoft/deberta-v3-base"
    max_edge_length: float = 2.0
    min_persistence: float = 0.05
    
    # Witness construction
    witness_k: int = 10  # k-nearest neighbors for witnesses
    
    # Bar matching
    lambda_sem: float = 0.5
    epsilon_match: float = 0.8
    
    # Event classification
    carry_overlap: float = 0.3  # Jaccard >= this AND d_sem < 0.3 → CARRY
    carry_d_sem: float = 0.3
    drift_overlap: float = 0.1  # Jaccard >= this → DRIFT
    
    # Re-entry detection  
    reentry_lookback: int = 3
    reentry_overlap: float = 0.2
    
    output_dir: str = "results/witnessed_analysis"


# =============================================================================
# Embedding
# =============================================================================

_model = None
_tokenizer = None

def get_embedder(model_name: str):
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


def embed_tokens(tokens: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    import torch
    model, tokenizer = get_embedder(model_name)
    device = next(model.parameters()).device
    
    embeddings = []
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=32, return_tensors='pt')
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())
    
    return np.vstack(embeddings)


# =============================================================================
# Persistent Homology
# =============================================================================

def compute_persistence(embeddings: np.ndarray, max_edge: float = 2.0) -> List[Tuple[int, float, float, List[int]]]:
    """
    Compute persistence diagram with representative vertices.
    Returns: [(dimension, birth, death, representative_indices), ...]
    """
    import gudhi
    
    rips = gudhi.RipsComplex(points=embeddings, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    
    # Get persistence pairs to find representatives
    pairs = st.persistence_pairs()
    
    results = []
    for birth_simplex, death_simplex in pairs:
        if not birth_simplex:
            continue
        
        dim = len(birth_simplex) - 1
        if dim > 1:
            continue
        
        # Get filtration values
        birth_val = st.filtration(birth_simplex)
        if death_simplex:
            death_val = st.filtration(death_simplex)
        else:
            death_val = max_edge
        
        # Representatives are vertices of the birth simplex
        representatives = list(birth_simplex)
        
        results.append((dim, birth_val, death_val, representatives))
    
    return results


def construct_witnessed_bars(
    persistence_data: List[Tuple[int, float, float, List[int]]],
    embeddings: np.ndarray,
    tokens: List[str],
    window_id: str,
    config: AnalysisConfig
) -> List[WitnessedBar]:
    """
    Construct bars with witnesses using k-nearest neighbors of representatives.
    """
    from scipy.spatial.distance import cdist
    
    bars = []
    
    for idx, (dim, birth, death, rep_indices) in enumerate(persistence_data):
        persistence = death - birth
        if persistence < config.min_persistence:
            continue
        
        # Get witness tokens: k-nearest neighbors of representatives
        if rep_indices:
            # Centroid of representatives
            rep_embeddings = embeddings[rep_indices]
            centroid = rep_embeddings.mean(axis=0)
            
            # Find k nearest tokens to centroid
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            nearest_indices = np.argsort(distances)[:config.witness_k]
            
            witness_tokens = [tokens[i] for i in nearest_indices]
            witness_centroid = embeddings[nearest_indices].mean(axis=0)
        else:
            # Fallback
            witness_tokens = tokens[:config.witness_k]
            witness_centroid = embeddings[:config.witness_k].mean(axis=0)
        
        bar = WitnessedBar(
            bar_id=f"{window_id}_bar_{idx}",  # UNIQUE ID
            window_id=window_id,
            dimension=dim,
            birth=birth,
            death=death,
            persistence=persistence,
            witness_tokens=witness_tokens,
            witness_centroid=witness_centroid
        )
        bars.append(bar)
    
    return bars


# =============================================================================
# Bar Matching
# =============================================================================

def compute_witness_overlap(bar1: WitnessedBar, bar2: WitnessedBar) -> float:
    """Jaccard similarity of witness tokens."""
    set1 = set(bar1.witness_tokens)
    set2 = set(bar2.witness_tokens)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_d_top(bar1: WitnessedBar, bar2: WitnessedBar) -> float:
    """Topological distance."""
    max_pers = max(bar1.persistence, bar2.persistence, 0.01)
    return max(abs(bar1.birth - bar2.birth), abs(bar1.death - bar2.death)) / max_pers


def compute_d_sem(bar1: WitnessedBar, bar2: WitnessedBar) -> float:
    """Semantic distance (centroid distance)."""
    if bar1.witness_centroid is None or bar2.witness_centroid is None:
        return 1.0
    dist = np.linalg.norm(bar1.witness_centroid - bar2.witness_centroid)
    return min(dist / 2.0, 1.0)


def classify_event(overlap: float, d_sem: float, config: AnalysisConfig) -> EventType:
    """Classify the transition event type."""
    if overlap >= config.carry_overlap and d_sem < config.carry_d_sem:
        return EventType.CARRY
    elif overlap >= config.drift_overlap:
        return EventType.DRIFT
    else:
        return EventType.DRIFT  # Still matched, just weak


def match_bars(
    bars_from: List[WitnessedBar],
    bars_to: List[WitnessedBar],
    config: AnalysisConfig
) -> Tuple[List[BarMatch], List[str], List[str]]:
    """
    Match bars between consecutive windows.
    Returns: (matches, unmatched_from_ids, unmatched_to_ids)
    """
    if not bars_from or not bars_to:
        return [], [b.bar_id for b in bars_from], [b.bar_id for b in bars_to]
    
    # Compute pairwise distances
    n_from, n_to = len(bars_from), len(bars_to)
    d_bar_matrix = np.full((n_from, n_to), np.inf)
    overlap_matrix = np.zeros((n_from, n_to))
    d_sem_matrix = np.zeros((n_from, n_to))
    
    for i, b1 in enumerate(bars_from):
        for j, b2 in enumerate(bars_to):
            if b1.dimension != b2.dimension:
                continue
            
            d_top = compute_d_top(b1, b2)
            d_sem = compute_d_sem(b1, b2)
            d_bar = max(d_top, config.lambda_sem * d_sem)
            overlap = compute_witness_overlap(b1, b2)
            
            d_bar_matrix[i, j] = d_bar
            overlap_matrix[i, j] = overlap
            d_sem_matrix[i, j] = d_sem
    
    # Greedy matching by d_bar
    matches = []
    matched_from = set()
    matched_to = set()
    
    while True:
        # Find minimum d_bar among unmatched
        min_val = np.inf
        min_i, min_j = -1, -1
        
        for i in range(n_from):
            if i in matched_from:
                continue
            for j in range(n_to):
                if j in matched_to:
                    continue
                if d_bar_matrix[i, j] < min_val:
                    min_val = d_bar_matrix[i, j]
                    min_i, min_j = i, j
        
        if min_val > config.epsilon_match or min_i < 0:
            break
        
        b1, b2 = bars_from[min_i], bars_to[min_j]
        overlap = overlap_matrix[min_i, min_j]
        d_sem = d_sem_matrix[min_i, min_j]
        
        event = classify_event(overlap, d_sem, config)
        
        matches.append(BarMatch(
            bar_from_id=b1.bar_id,
            bar_to_id=b2.bar_id,
            d_top=compute_d_top(b1, b2),
            d_sem=d_sem,
            d_bar=min_val,
            witness_overlap=overlap,
            event_type=event
        ))
        
        matched_from.add(min_i)
        matched_to.add(min_j)
    
    unmatched_from = [bars_from[i].bar_id for i in range(n_from) if i not in matched_from]
    unmatched_to = [bars_to[j].bar_id for j in range(n_to) if j not in matched_to]
    
    return matches, unmatched_from, unmatched_to


# =============================================================================
# Journey Construction
# =============================================================================

def build_journeys(
    window_analyses: List[WindowAnalysis],
    all_matches: List[List[BarMatch]],
    all_unmatched: List[Tuple[List[str], List[str]]],
    config: AnalysisConfig
) -> Dict[str, Journey]:
    """Build journey graph from matches."""
    
    journeys: Dict[str, Journey] = {}
    bar_to_journey: Dict[str, str] = {}  # bar_id → journey_id
    journey_counter = 0
    
    # Build bar lookup
    bar_lookup: Dict[str, WitnessedBar] = {}
    for window in window_analyses:
        for bar in window.bars:
            bar_lookup[bar.bar_id] = bar
    
    # First window: all bars spawn
    if window_analyses:
        w0 = window_analyses[0]
        for bar in w0.bars:
            jid = f"journey_{journey_counter}"
            journey_counter += 1
            
            journey = Journey(journey_id=jid, dimension=bar.dimension)
            journey.steps.append(JourneyStep(
                tau=0,
                window_id=w0.window_id,
                bar_id=bar.bar_id,
                event=EventType.SPAWN,
                witness_tokens=bar.witness_tokens.copy()
            ))
            
            journeys[jid] = journey
            bar_to_journey[bar.bar_id] = jid
    
    # Process subsequent windows
    for tau in range(1, len(window_analyses)):
        matches = all_matches[tau - 1]
        unmatched_from, unmatched_to = all_unmatched[tau - 1]
        window = window_analyses[tau]
        
        # Process matches
        for match in matches:
            prev_journey_id = bar_to_journey.get(match.bar_from_id)
            if prev_journey_id is None:
                continue
            
            journey = journeys[prev_journey_id]
            bar = bar_lookup.get(match.bar_to_id)
            if bar is None:
                continue
            
            journey.steps.append(JourneyStep(
                tau=tau,
                window_id=window.window_id,
                bar_id=bar.bar_id,
                event=match.event_type,
                witness_tokens=bar.witness_tokens.copy()
            ))
            
            bar_to_journey[match.bar_to_id] = prev_journey_id
        
        # Process ruptures (mark journey as ended, but don't add step)
        # Ruptures are tracked by unmatched_from - those journeys end
        
        # Process spawns and re-entries
        for bar_id in unmatched_to:
            bar = bar_lookup.get(bar_id)
            if bar is None:
                continue
            
            # Check for re-entry
            is_reentry = False
            reentry_journey_id = None
            
            # Look for recently-ended journeys with witness overlap
            for jid, journey in journeys.items():
                if not journey.steps:
                    continue
                
                last_step = journey.steps[-1]
                
                # Must have ended (not in current window)
                if last_step.tau >= tau - 1:
                    continue
                
                # Check recency
                if tau - last_step.tau > config.reentry_lookback:
                    continue
                
                # Check witness overlap
                overlap = len(set(bar.witness_tokens) & set(last_step.witness_tokens))
                overlap_ratio = overlap / max(len(bar.witness_tokens), 1)
                
                if overlap_ratio >= config.reentry_overlap:
                    is_reentry = True
                    reentry_journey_id = jid
                    break
            
            if is_reentry and reentry_journey_id:
                journey = journeys[reentry_journey_id]
                journey.steps.append(JourneyStep(
                    tau=tau,
                    window_id=window.window_id,
                    bar_id=bar.bar_id,
                    event=EventType.REENTRY,
                    witness_tokens=bar.witness_tokens.copy()
                ))
                bar_to_journey[bar.bar_id] = reentry_journey_id
            else:
                # New journey
                jid = f"journey_{journey_counter}"
                journey_counter += 1
                
                journey = Journey(journey_id=jid, dimension=bar.dimension)
                journey.steps.append(JourneyStep(
                    tau=tau,
                    window_id=window.window_id,
                    bar_id=bar.bar_id,
                    event=EventType.SPAWN,
                    witness_tokens=bar.witness_tokens.copy()
                ))
                
                journeys[jid] = journey
                bar_to_journey[bar.bar_id] = jid
    
    return journeys


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class SelfMetrics:
    num_journeys: int
    total_steps: int
    
    # Event counts
    spawns: int
    carries: int
    drifts: int
    reentries: int
    
    # Derived
    carry_ratio: float  # carries / (carries + drifts)
    reentry_rate: float  # reentries / (journeys that ended)
    mean_lifespan: float
    mean_witness_stability: float
    
    scheduler_type: str = ""
    
    def classify(self):
        if self.reentry_rate > 0.3 and self.mean_witness_stability > 0.4:
            self.scheduler_type = "Reparative"
        elif self.carry_ratio > 0.5 and self.mean_witness_stability > 0.5:
            self.scheduler_type = "Obsessive"
        elif self.reentry_rate > 0.2 and self.mean_lifespan > 3:
            self.scheduler_type = "Generative"
        elif self.reentry_rate < 0.1:
            self.scheduler_type = "Avoidant"
        else:
            self.scheduler_type = "Mixed"


def compute_metrics(journeys: Dict[str, Journey], num_windows: int) -> SelfMetrics:
    """Compute Self metrics from journeys."""
    
    if not journeys:
        return SelfMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    
    spawns = 0
    carries = 0
    drifts = 0
    reentries = 0
    total_steps = 0
    lifespans = []
    stabilities = []
    
    for journey in journeys.values():
        lifespans.append(journey.lifespan)
        total_steps += journey.lifespan
        
        # Count events
        for step in journey.steps:
            if step.event == EventType.SPAWN:
                spawns += 1
            elif step.event == EventType.CARRY:
                carries += 1
            elif step.event == EventType.DRIFT:
                drifts += 1
            elif step.event == EventType.REENTRY:
                reentries += 1
        
        # Witness stability: Jaccard between consecutive steps
        for i in range(1, len(journey.steps)):
            s1, s2 = journey.steps[i-1], journey.steps[i]
            set1, set2 = set(s1.witness_tokens), set(s2.witness_tokens)
            if set1 and set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
                stabilities.append(jaccard)
    
    # Journeys that "ended" (didn't reach last window)
    ended_journeys = sum(1 for j in journeys.values() if j.steps[-1].tau < num_windows - 1)
    
    carry_ratio = carries / max(carries + drifts, 1)
    reentry_rate = reentries / max(ended_journeys, 1)
    
    metrics = SelfMetrics(
        num_journeys=len(journeys),
        total_steps=total_steps,
        spawns=spawns,
        carries=carries,
        drifts=drifts,
        reentries=reentries,
        carry_ratio=carry_ratio,
        reentry_rate=reentry_rate,
        mean_lifespan=np.mean(lifespans) if lifespans else 0,
        mean_witness_stability=np.mean(stabilities) if stabilities else 0
    )
    metrics.classify()
    
    return metrics


# =============================================================================
# Main Pipeline
# =============================================================================

def load_conversations(filepath: str) -> List[dict]:
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    conversations = data.get('conversations', data)
    print(f"  Loaded {len(conversations)} conversations")
    return conversations


def create_monthly_windows(conversations: List[dict]) -> Dict[str, List[dict]]:
    windows = defaultdict(list)
    for conv in conversations:
        if not conv.get('create_time'):
            continue
        dt = datetime.fromtimestamp(conv['create_time'])
        key = dt.strftime('%Y-%m')
        windows[key].append(conv)
    return dict(sorted(windows.items()))


def get_tokens(conversations: List[dict], max_tokens: int) -> Tuple[List[str], int]:
    import re
    all_tokens = []
    for conv in conversations:
        for turn in conv.get('turns', []):
            content = turn.get('content', '')
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            all_tokens.extend(words)
    
    original = len(all_tokens)
    if len(all_tokens) > max_tokens:
        indices = np.random.choice(len(all_tokens), max_tokens, replace=False)
        all_tokens = [all_tokens[i] for i in sorted(indices)]
    
    return all_tokens, original


def analyze_window(
    window_id: str,
    tau: int,
    conversations: List[dict],
    config: AnalysisConfig
) -> WindowAnalysis:
    """Analyze a single window."""
    
    print(f"  [{tau}] {window_id}: {len(conversations)} conversations", end="", flush=True)
    
    tokens, original = get_tokens(conversations, config.tokens_per_window)
    print(f" → {len(tokens)}/{original} tokens", end="", flush=True)
    
    if len(tokens) < 50:
        print(" [SKIPPED]")
        return WindowAnalysis(window_id=window_id, tau=tau, 
                             num_conversations=len(conversations),
                             num_tokens=len(tokens), bars=[])
    
    # Embed
    t0 = time.time()
    embeddings = embed_tokens(tokens, config.embedding_model)
    t_embed = time.time() - t0
    print(f" → embedded ({t_embed:.1f}s)", end="", flush=True)
    
    # PH
    t0 = time.time()
    persistence_data = compute_persistence(embeddings, config.max_edge_length)
    t_ph = time.time() - t0
    print(f" → PH ({t_ph:.1f}s, {len(persistence_data)} features)", end="", flush=True)
    
    # Construct witnessed bars
    bars = construct_witnessed_bars(persistence_data, embeddings, tokens, window_id, config)
    print(f" → {len(bars)} bars")
    
    return WindowAnalysis(
        window_id=window_id,
        tau=tau,
        num_conversations=len(conversations),
        num_tokens=len(tokens),
        bars=bars,
        timing_embed=t_embed,
        timing_ph=t_ph
    )


def run_analysis(
    conversations: List[dict],
    config: AnalysisConfig,
    test_mode: bool = False
) -> Tuple[List[WindowAnalysis], Dict[str, Journey], SelfMetrics]:
    """Run full analysis."""
    
    print("\nCreating monthly windows...")
    windows = create_monthly_windows(conversations)
    window_ids = list(windows.keys())
    print(f"  {len(window_ids)} windows: {window_ids[0]} to {window_ids[-1]}")
    
    if test_mode:
        window_ids = window_ids[:5]
        print(f"  TEST MODE: {len(window_ids)} windows")
    
    # Analyze windows
    print("\nAnalyzing windows...")
    window_analyses = []
    for tau, wid in enumerate(window_ids):
        w = analyze_window(wid, tau, windows[wid], config)
        window_analyses.append(w)
    
    # Match bars
    print("\nMatching bars...")
    all_matches = []
    all_unmatched = []
    
    for i in range(len(window_analyses) - 1):
        w1, w2 = window_analyses[i], window_analyses[i + 1]
        matches, unm_from, unm_to = match_bars(w1.bars, w2.bars, config)
        all_matches.append(matches)
        all_unmatched.append((unm_from, unm_to))
        
        # Count event types
        carries = sum(1 for m in matches if m.event_type == EventType.CARRY)
        drifts = sum(1 for m in matches if m.event_type == EventType.DRIFT)
        
        print(f"  {w1.window_id} → {w2.window_id}: {len(matches)} matches "
              f"({carries}C/{drifts}D), {len(unm_from)} ruptures, {len(unm_to)} spawns")
    
    # Build journeys
    print("\nBuilding journeys...")
    journeys = build_journeys(window_analyses, all_matches, all_unmatched, config)
    print(f"  {len(journeys)} journeys")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(journeys, len(window_analyses))
    
    return window_analyses, journeys, metrics


def print_report(
    window_analyses: List[WindowAnalysis],
    journeys: Dict[str, Journey],
    metrics: SelfMetrics
):
    """Print analysis report."""
    
    print("\n" + "=" * 70)
    print("WITNESSED PERSISTENT HOMOLOGY ANALYSIS")
    print("=" * 70)
    
    print(f"\n## SUMMARY")
    print(f"  Windows: {len(window_analyses)}")
    print(f"  Total bars: {sum(len(w.bars) for w in window_analyses)}")
    print(f"  Journeys: {metrics.num_journeys}")
    
    print(f"\n## EVENT DISTRIBUTION")
    print(f"  Spawns:    {metrics.spawns}")
    print(f"  Carries:   {metrics.carries}")
    print(f"  Drifts:    {metrics.drifts}")
    print(f"  Re-entries: {metrics.reentries}")
    
    print(f"\n## SELF METRICS")
    print(f"  Carry Ratio:        {metrics.carry_ratio:.3f}")
    print(f"  Re-entry Rate:      {metrics.reentry_rate:.3f}")
    print(f"  Mean Lifespan:      {metrics.mean_lifespan:.2f} windows")
    print(f"  Witness Stability:  {metrics.mean_witness_stability:.3f}")
    print(f"  Scheduler Type:     {metrics.scheduler_type}")
    
    # Top journeys
    print(f"\n## LONGEST JOURNEYS")
    sorted_journeys = sorted(journeys.values(), key=lambda j: -j.lifespan)[:10]
    
    for j in sorted_journeys:
        seq = j.event_sequence[:40] + ("..." if len(j.event_sequence) > 40 else "")
        print(f"  {j.signature:<30} life={j.lifespan} [{seq}]")
    
    # Sanity check
    print(f"\n## SANITY CHECK")
    max_possible_lifespan = len(window_analyses)
    actual_max = max(j.lifespan for j in journeys.values()) if journeys else 0
    print(f"  Max possible lifespan: {max_possible_lifespan}")
    print(f"  Actual max lifespan:   {actual_max}")
    if actual_max > max_possible_lifespan:
        print(f"  ⚠️  ERROR: Lifespan exceeds windows!")
    else:
        print(f"  ✓ Lifespans are valid")


def save_results(
    window_analyses: List[WindowAnalysis],
    journeys: Dict[str, Journey],
    metrics: SelfMetrics,
    config: AnalysisConfig
):
    """Save results to files."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Journeys
    journey_data = []
    for j in journeys.values():
        journey_data.append({
            "journey_id": j.journey_id,
            "dimension": j.dimension,
            "lifespan": j.lifespan,
            "signature": j.signature,
            "birth_window": j.birth_window,
            "events": [s.event.value for s in j.steps],
            "windows": [s.window_id for s in j.steps]
        })
    
    with open(os.path.join(config.output_dir, "journeys.json"), 'w') as f:
        json.dump(journey_data, f, indent=2)
    
    # Metrics
    metrics_data = {
        "num_journeys": metrics.num_journeys,
        "spawns": metrics.spawns,
        "carries": metrics.carries,
        "drifts": metrics.drifts,
        "reentries": metrics.reentries,
        "carry_ratio": metrics.carry_ratio,
        "reentry_rate": metrics.reentry_rate,
        "mean_lifespan": metrics.mean_lifespan,
        "witness_stability": metrics.mean_witness_stability,
        "scheduler_type": metrics.scheduler_type
    }
    
    with open(os.path.join(config.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\nResults saved to {config.output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to cassie_parsed.json")
    parser.add_argument("--output", "-o", default="results/witnessed_analysis")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--tokens-per-window", type=int, default=500)
    
    args = parser.parse_args()
    
    config = AnalysisConfig(
        tokens_per_window=args.tokens_per_window,
        output_dir=args.output
    )
    
    conversations = load_conversations(args.input)
    window_analyses, journeys, metrics = run_analysis(conversations, config, args.test)
    
    print_report(window_analyses, journeys, metrics)
    save_results(window_analyses, journeys, metrics, config)


if __name__ == "__main__":
    main()