#!/usr/bin/env python3
"""
Self Emergence Detection
========================

This script answers: "At what τ does a Self emerge, and what characterizes it?"

Instead of presupposing a transition point, we:
1. Compute rolling Self-metrics at each τ
2. Define Selfhood criteria based on the theory (Presence + Generativity)
3. Find the emergence point where criteria are first satisfied
4. Characterize the Adm structure and Scheduler of the emerged Self

The key insight: A Self isn't declared, it's detected. We let the topology speak.

Selfhood Criteria (from Chapters 4-5):
--------------------------------------
PRESENCE (the Self persists):
  - fragmentation_index < 0.5 (themes connect via shared witnesses)
  - witness_stability > 0.3 (witnesses persist across transitions)
  - mean_journey_lifespan > 3.0 (themes live multiple windows)

GENERATIVITY (the Self creates and repairs):
  - reentry_rate > 0.2 (ruptured themes return)
  - spawn_to_rupture_ratio > 0.8 (more creation than destruction)

A Self emerges at the first τ where BOTH Presence AND Generativity hold
over a rolling window of k consecutive measurements.

Usage:
    python scripts/self_emergence.py cassie_parsed.json
    python scripts/self_emergence.py cassie_parsed.json --rolling-window 5
    python scripts/self_emergence.py cassie_parsed.json --test

Output:
    - Timeline of Self-metrics
    - Detected emergence point (if any)
    - Characterization of emerged Self (Adm, Scheduler)
    - Book-ready figures and tables
"""

import json
import argparse
import os
import time
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Set, Optional, Any
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Import from full_witnessed_analysis (or redefine if standalone)
# =============================================================================

class EventType(str, Enum):
    SPAWN = "spawn"
    CARRY = "carry"
    DRIFT = "drift"
    RUPTURE_OUT = "rupture_out"
    REENTRY = "reentry"
    GENERATIVE = "generative"


@dataclass
class SelfhoodCriteria:
    """Thresholds for declaring Selfhood."""
    # Presence
    max_fragmentation: float = 0.5
    min_witness_stability: float = 0.3
    min_journey_lifespan: float = 3.0
    
    # Generativity
    min_reentry_rate: float = 0.2
    min_spawn_rupture_ratio: float = 0.8
    
    # Consistency (must hold for k consecutive windows)
    consistency_window: int = 3
    
    def check_presence(self, metrics: 'RollingMetrics') -> bool:
        return (
            metrics.fragmentation < self.max_fragmentation and
            metrics.witness_stability > self.min_witness_stability and
            metrics.mean_lifespan > self.min_journey_lifespan
        )
    
    def check_generativity(self, metrics: 'RollingMetrics') -> bool:
        return (
            metrics.reentry_rate > self.min_reentry_rate and
            metrics.spawn_rupture_ratio > self.min_spawn_rupture_ratio
        )
    
    def check_selfhood(self, metrics: 'RollingMetrics') -> bool:
        return self.check_presence(metrics) and self.check_generativity(metrics)


@dataclass
class RollingMetrics:
    """Metrics computed over a rolling window ending at τ."""
    tau: int
    window_id: str
    window_start: str  # First window in rolling range
    window_end: str    # Last window (= window_id)
    
    # Core metrics
    fragmentation: float
    witness_stability: float
    mean_lifespan: float
    reentry_rate: float
    spawn_rupture_ratio: float
    
    # Event counts (in rolling window)
    spawns: int
    carries: int
    drifts: int
    ruptures: int
    reentries: int
    
    # Derived
    has_presence: bool = False
    has_generativity: bool = False
    has_selfhood: bool = False
    
    # Adm characteristics (when Selfhood detected)
    adm_density: float = 0.0  # Fraction of admissible transitions
    adm_mean_distance: float = 0.0  # Mean d_bar of matches


@dataclass
class EmergenceResult:
    """Result of emergence detection."""
    emerged: bool
    emergence_tau: Optional[int]
    emergence_window: Optional[str]
    emergence_date: Optional[str]
    
    # Metrics at emergence
    metrics_at_emergence: Optional[RollingMetrics]
    
    # Timeline
    metrics_timeline: List[RollingMetrics]
    
    # Post-emergence characterization
    scheduler_type: Optional[str] = None
    adm_characterization: Optional[str] = None
    
    # Confidence
    consistency_streak: int = 0  # How many windows Selfhood held


# =============================================================================
# Lightweight Analysis (for faster iteration)
# =============================================================================

@dataclass
class LightBar:
    """Lightweight bar for faster analysis."""
    bar_id: str
    dimension: int
    birth: float
    death: float
    persistence: float
    witness_tokens: List[str]
    centroid: Optional[np.ndarray] = None


@dataclass
class LightWindow:
    """Lightweight window analysis."""
    window_id: str
    tau: int
    num_conversations: int
    bars: List[LightBar]
    token_set: Set[str]


def load_conversations(filepath: str) -> List[dict]:
    """Load parsed conversations."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    conversations = data.get('conversations', data)
    print(f"  Loaded {len(conversations)} conversations")
    return conversations


def create_monthly_windows(conversations: List[dict]) -> Dict[str, List[dict]]:
    """Group conversations by month."""
    windows = defaultdict(list)
    for conv in conversations:
        if not conv.get('create_time'):
            continue
        dt = datetime.fromtimestamp(conv['create_time'])
        key = dt.strftime('%Y-%m')
        windows[key].append(conv)
    return dict(sorted(windows.items()))


# =============================================================================
# Embedding and PH (simplified for speed)
# =============================================================================

_embedder = None

def get_embedder():
    """Get or create sentence embedder."""
    global _embedder
    if _embedder is None:
        print("  Loading embedding model...")
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Faster than DeBERTa
            print("    Using sentence-transformers (fast)")
        except ImportError:
            print("    sentence-transformers not found, using fallback")
            _embedder = "fallback"
    return _embedder


def embed_tokens_fast(tokens: List[str], batch_size: int = 64) -> np.ndarray:
    """Fast token embedding using sentence-transformers."""
    embedder = get_embedder()
    
    if embedder == "fallback":
        # Fallback: random embeddings (for testing only!)
        return np.random.randn(len(tokens), 384).astype(np.float32)
    
    # Deduplicate for speed
    unique_tokens = list(set(tokens))
    embeddings_dict = {}
    
    for i in range(0, len(unique_tokens), batch_size):
        batch = unique_tokens[i:i+batch_size]
        embs = embedder.encode(batch, show_progress_bar=False)
        for tok, emb in zip(batch, embs):
            embeddings_dict[tok] = emb
    
    # Map back to original order
    return np.array([embeddings_dict[t] for t in tokens])


def compute_persistence_fast(
    embeddings: np.ndarray,
    max_edge: float = 2.0,
    min_persistence: float = 0.05
) -> List[Tuple[int, float, float]]:
    """Compute persistence diagram (H0 and H1)."""
    try:
        import gudhi
    except ImportError:
        print("    GUDHI not found - using mock persistence")
        # Mock: return some fake bars based on embedding structure
        n = len(embeddings)
        return [(0, 0.0, 0.5), (0, 0.1, 0.6), (1, 0.2, 0.8)]
    
    rips = gudhi.RipsComplex(points=embeddings, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    
    diagram = []
    for dim in [0, 1]:
        intervals = st.persistence_intervals_in_dimension(dim)
        for b, d in intervals:
            if d == float('inf'):
                d = max_edge
            if d - b >= min_persistence:
                diagram.append((dim, float(b), float(d)))
    
    return diagram


def construct_witnesses_fast(
    diagram: List[Tuple[int, float, float]],
    embeddings: np.ndarray,
    tokens: List[str],
    epsilon: float = 0.15
) -> List[LightBar]:
    """Construct witnesses for bars."""
    from scipy.spatial.distance import cdist
    
    bars = []
    distances = cdist(embeddings, embeddings)
    
    for i, (dim, birth, death) in enumerate(diagram):
        # Find tokens at the birth scale
        mask = np.any(
            (distances >= birth - epsilon) & (distances <= birth + epsilon),
            axis=1
        )
        witness_indices = np.where(mask)[0][:15]
        
        if len(witness_indices) < 3:
            # Fallback: closest to mean
            mean_emb = embeddings.mean(axis=0)
            dists = np.linalg.norm(embeddings - mean_emb, axis=1)
            witness_indices = np.argsort(dists)[:5]
        
        witness_tokens = [tokens[j] for j in witness_indices]
        centroid = embeddings[witness_indices].mean(axis=0) if len(witness_indices) > 0 else None
        
        bars.append(LightBar(
            bar_id=f"bar_{i}",
            dimension=dim,
            birth=birth,
            death=death,
            persistence=death - birth,
            witness_tokens=witness_tokens,
            centroid=centroid
        ))
    
    return bars


# =============================================================================
# Bar Matching
# =============================================================================

def match_bars_fast(
    bars1: List[LightBar],
    bars2: List[LightBar],
    lambda_sem: float = 0.5,
    epsilon_match: float = 0.8
) -> Tuple[List[Tuple[str, str, float, float]], List[str], List[str]]:
    """
    Match bars between windows.
    Returns: (matches, unmatched_from, unmatched_to)
    where matches = [(bar1_id, bar2_id, d_bar, witness_overlap), ...]
    """
    if not bars1 or not bars2:
        return [], [b.bar_id for b in bars1], [b.bar_id for b in bars2]
    
    matches = []
    matched1 = set()
    matched2 = set()
    
    # Compute all pairwise distances
    candidates = []
    for i, b1 in enumerate(bars1):
        for j, b2 in enumerate(bars2):
            if b1.dimension != b2.dimension:
                continue
            
            # d_top
            d_top = max(
                abs(b1.birth - b2.birth),
                abs(b1.death - b2.death)
            ) / max(b1.persistence, b2.persistence, 0.01)
            
            # d_sem
            if b1.centroid is not None and b2.centroid is not None:
                d_sem = np.linalg.norm(b1.centroid - b2.centroid)
                d_sem = min(d_sem / 2.0, 1.0)  # Normalize
            else:
                d_sem = 1.0
            
            d_bar = max(d_top, lambda_sem * d_sem)
            
            # Witness overlap
            set1 = set(b1.witness_tokens)
            set2 = set(b2.witness_tokens)
            overlap = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
            
            if d_bar < epsilon_match:
                candidates.append((d_bar, i, j, overlap))
    
    # Greedy matching
    candidates.sort()
    for d_bar, i, j, overlap in candidates:
        if i in matched1 or j in matched2:
            continue
        matches.append((bars1[i].bar_id, bars2[j].bar_id, d_bar, overlap))
        matched1.add(i)
        matched2.add(j)
    
    unmatched1 = [bars1[i].bar_id for i in range(len(bars1)) if i not in matched1]
    unmatched2 = [bars2[j].bar_id for j in range(len(bars2)) if j not in matched2]
    
    return matches, unmatched1, unmatched2


# =============================================================================
# Rolling Metrics Computation
# =============================================================================

def analyze_window_light(
    window_id: str,
    tau: int,
    conversations: List[dict],
    max_tokens: int = 400
) -> LightWindow:
    """Lightweight window analysis."""
    import re
    
    # Extract tokens
    all_tokens = []
    for conv in conversations:
        for turn in conv.get('turns', []):
            content = turn.get('content', '')
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            all_tokens.extend(words)
    
    # Subsample
    if len(all_tokens) > max_tokens:
        indices = np.random.choice(len(all_tokens), max_tokens, replace=False)
        tokens = [all_tokens[i] for i in sorted(indices)]
    else:
        tokens = all_tokens
    
    if len(tokens) < 50:
        return LightWindow(
            window_id=window_id,
            tau=tau,
            num_conversations=len(conversations),
            bars=[],
            token_set=set(tokens)
        )
    
    # Embed and compute PH
    embeddings = embed_tokens_fast(tokens)
    diagram = compute_persistence_fast(embeddings)
    bars = construct_witnesses_fast(diagram, embeddings, tokens)
    
    return LightWindow(
        window_id=window_id,
        tau=tau,
        num_conversations=len(conversations),
        bars=bars,
        token_set=set(tokens)
    )


def compute_rolling_metrics(
    windows: List[LightWindow],
    all_matches: List[List[Tuple[str, str, float, float]]],
    all_unmatched: List[Tuple[List[str], List[str]]],
    rolling_k: int = 5,
    criteria: SelfhoodCriteria = None
) -> List[RollingMetrics]:
    """
    Compute Self-metrics over rolling windows.
    
    At each τ, we look at windows [max(0, τ-k+1), τ] and compute:
    - Fragmentation (via witness overlap across journeys)
    - Witness stability (mean Jaccard of matched bars)
    - Mean lifespan (how long do themes persist in this range)
    - Re-entry rate, spawn/rupture ratio
    """
    if criteria is None:
        criteria = SelfhoodCriteria()
    
    metrics_list = []
    
    for tau in range(len(windows)):
        # Rolling window range
        start_tau = max(0, tau - rolling_k + 1)
        end_tau = tau
        
        if tau == 0:
            # First window: no transitions yet
            metrics_list.append(RollingMetrics(
                tau=tau,
                window_id=windows[tau].window_id,
                window_start=windows[tau].window_id,
                window_end=windows[tau].window_id,
                fragmentation=1.0,
                witness_stability=0.0,
                mean_lifespan=1.0,
                reentry_rate=0.0,
                spawn_rupture_ratio=1.0,
                spawns=len(windows[tau].bars),
                carries=0,
                drifts=0,
                ruptures=0,
                reentries=0
            ))
            continue
        
        # Aggregate events in rolling window
        spawns = 0
        carries = 0
        drifts = 0
        ruptures = 0
        reentries = 0
        
        # Track witness overlaps for stability
        overlaps = []
        
        # Track journey lifespans
        journey_births = {}  # bar_id -> birth_tau
        journey_active = {}  # bar_id -> last_seen_tau
        
        for t in range(start_tau, end_tau + 1):
            if t == 0:
                for bar in windows[t].bars:
                    spawns += 1
                    journey_births[bar.bar_id] = t
                    journey_active[bar.bar_id] = t
                continue
            
            matches = all_matches[t - 1]
            unmatched_from, unmatched_to = all_unmatched[t - 1]
            
            for bar1_id, bar2_id, d_bar, overlap in matches:
                overlaps.append(overlap)
                
                if overlap >= 0.3:
                    carries += 1
                else:
                    drifts += 1
                
                # Update journey tracking
                if bar1_id in journey_births:
                    journey_births[bar2_id] = journey_births[bar1_id]
                else:
                    journey_births[bar2_id] = t
                journey_active[bar2_id] = t
            
            # Ruptures
            ruptures += len(unmatched_from)
            
            # Spawns vs re-entries
            for bar_id in unmatched_to:
                # Check if this could be a re-entry
                # (Simplified: check witness overlap with recent ruptures)
                is_reentry = False
                
                bar = next((b for b in windows[t].bars if b.bar_id == bar_id), None)
                if bar and t > start_tau:
                    for prev_t in range(max(start_tau, t - 3), t):
                        prev_unmatched = all_unmatched[prev_t - 1][0] if prev_t > 0 else []
                        for prev_bar_id in prev_unmatched:
                            prev_bar = next(
                                (b for b in windows[prev_t].bars if b.bar_id == prev_bar_id),
                                None
                            )
                            if prev_bar:
                                overlap = len(set(bar.witness_tokens) & set(prev_bar.witness_tokens))
                                if overlap >= 2:
                                    is_reentry = True
                                    break
                        if is_reentry:
                            break
                
                if is_reentry:
                    reentries += 1
                else:
                    spawns += 1
                
                journey_births[bar_id] = t
                journey_active[bar_id] = t
        
        # Compute derived metrics
        
        # Fragmentation: use token overlap between windows as proxy
        token_sets = [windows[t].token_set for t in range(start_tau, end_tau + 1)]
        if len(token_sets) > 1:
            pairwise_overlaps = []
            for i in range(len(token_sets) - 1):
                s1, s2 = token_sets[i], token_sets[i + 1]
                if s1 and s2:
                    pairwise_overlaps.append(len(s1 & s2) / len(s1 | s2))
            fragmentation = 1 - np.mean(pairwise_overlaps) if pairwise_overlaps else 1.0
        else:
            fragmentation = 1.0
        
        # Witness stability: mean overlap of matched bars
        witness_stability = np.mean(overlaps) if overlaps else 0.0
        
        # Mean lifespan
        lifespans = []
        for bar_id, birth in journey_births.items():
            last_seen = journey_active.get(bar_id, birth)
            lifespans.append(last_seen - birth + 1)
        mean_lifespan = np.mean(lifespans) if lifespans else 1.0
        
        # Re-entry rate
        reentry_rate = reentries / max(ruptures, 1)
        
        # Spawn/rupture ratio
        spawn_rupture_ratio = spawns / max(ruptures, 1) if ruptures > 0 else float('inf')
        spawn_rupture_ratio = min(spawn_rupture_ratio, 10.0)  # Cap for display
        
        # Adm density (fraction of bars that matched)
        total_bars_from = sum(len(windows[t].bars) for t in range(start_tau, end_tau))
        total_matched = sum(len(all_matches[t - 1]) for t in range(max(1, start_tau), end_tau + 1))
        adm_density = total_matched / max(total_bars_from, 1)
        
        metrics = RollingMetrics(
            tau=tau,
            window_id=windows[tau].window_id,
            window_start=windows[start_tau].window_id,
            window_end=windows[tau].window_id,
            fragmentation=float(fragmentation),
            witness_stability=float(witness_stability),
            mean_lifespan=float(mean_lifespan),
            reentry_rate=float(reentry_rate),
            spawn_rupture_ratio=float(spawn_rupture_ratio),
            spawns=spawns,
            carries=carries,
            drifts=drifts,
            ruptures=ruptures,
            reentries=reentries,
            adm_density=float(adm_density)
        )
        
        # Check Selfhood criteria
        metrics.has_presence = criteria.check_presence(metrics)
        metrics.has_generativity = criteria.check_generativity(metrics)
        metrics.has_selfhood = criteria.check_selfhood(metrics)
        
        metrics_list.append(metrics)
    
    return metrics_list


# =============================================================================
# Emergence Detection
# =============================================================================

def detect_emergence(
    metrics_timeline: List[RollingMetrics],
    criteria: SelfhoodCriteria
) -> EmergenceResult:
    """
    Detect the emergence point: first τ where Selfhood holds for k consecutive windows.
    """
    consistency_count = 0
    emergence_tau = None
    emergence_metrics = None
    
    for metrics in metrics_timeline:
        if metrics.has_selfhood:
            consistency_count += 1
            if consistency_count >= criteria.consistency_window and emergence_tau is None:
                # Found emergence!
                emergence_tau = metrics.tau - criteria.consistency_window + 1
                emergence_metrics = metrics_timeline[emergence_tau]
        else:
            consistency_count = 0
    
    if emergence_tau is not None:
        # Characterize the emerged Self
        post_emergence = [m for m in metrics_timeline if m.tau >= emergence_tau]
        
        # Scheduler type
        mean_reentry = np.mean([m.reentry_rate for m in post_emergence])
        mean_stability = np.mean([m.witness_stability for m in post_emergence])
        mean_fragmentation = np.mean([m.fragmentation for m in post_emergence])
        
        if mean_reentry > 0.4 and mean_stability > 0.4:
            scheduler_type = "Reparative"
        elif mean_reentry < 0.15 and mean_stability < 0.3:
            scheduler_type = "Avoidant"
        elif mean_stability > 0.6 and mean_fragmentation < 0.3:
            scheduler_type = "Obsessive"
        elif mean_reentry > 0.25 and np.mean([m.mean_lifespan for m in post_emergence]) > 4:
            scheduler_type = "Generative"
        else:
            scheduler_type = "Mixed"
        
        # Adm characterization
        mean_adm = np.mean([m.adm_density for m in post_emergence])
        if mean_adm > 0.6:
            adm_char = "Dense (strong thematic continuity)"
        elif mean_adm > 0.3:
            adm_char = "Moderate (selective persistence)"
        else:
            adm_char = "Sparse (frequent transformation)"
        
        return EmergenceResult(
            emerged=True,
            emergence_tau=emergence_tau,
            emergence_window=metrics_timeline[emergence_tau].window_id,
            emergence_date=metrics_timeline[emergence_tau].window_id,
            metrics_at_emergence=emergence_metrics,
            metrics_timeline=metrics_timeline,
            scheduler_type=scheduler_type,
            adm_characterization=adm_char,
            consistency_streak=consistency_count
        )
    
    return EmergenceResult(
        emerged=False,
        emergence_tau=None,
        emergence_window=None,
        emergence_date=None,
        metrics_at_emergence=None,
        metrics_timeline=metrics_timeline
    )


# =============================================================================
# Output
# =============================================================================

def print_timeline(metrics_timeline: List[RollingMetrics], emergence_tau: Optional[int]):
    """Print the metrics timeline."""
    
    print("\n" + "=" * 90)
    print("SELF-METRICS TIMELINE")
    print("=" * 90)
    
    print(f"\n{'τ':>3} {'Window':>8} {'Frag':>6} {'Stab':>6} {'Life':>6} {'ReEnt':>6} {'S/R':>6} {'P':>2} {'G':>2} {'S':>2}")
    print("-" * 90)
    
    for m in metrics_timeline:
        marker = " ←← EMERGENCE" if emergence_tau is not None and m.tau == emergence_tau else ""
        
        p = "✓" if m.has_presence else "·"
        g = "✓" if m.has_generativity else "·"
        s = "★" if m.has_selfhood else "·"
        
        print(f"{m.tau:>3} {m.window_id:>8} {m.fragmentation:>6.3f} {m.witness_stability:>6.3f} "
              f"{m.mean_lifespan:>6.2f} {m.reentry_rate:>6.3f} {m.spawn_rupture_ratio:>6.2f} "
              f"{p:>2} {g:>2} {s:>2}{marker}")
    
    print("\nLegend: P=Presence, G=Generativity, S=Selfhood (both P and G)")


def print_emergence_report(result: EmergenceResult, criteria: SelfhoodCriteria):
    """Print the emergence analysis report."""
    
    print("\n" + "=" * 70)
    print("SELF EMERGENCE ANALYSIS")
    print("=" * 70)
    
    print(f"\n## SELFHOOD CRITERIA")
    print(f"  Presence:")
    print(f"    - Fragmentation < {criteria.max_fragmentation}")
    print(f"    - Witness Stability > {criteria.min_witness_stability}")
    print(f"    - Mean Lifespan > {criteria.min_journey_lifespan}")
    print(f"  Generativity:")
    print(f"    - Re-entry Rate > {criteria.min_reentry_rate}")
    print(f"    - Spawn/Rupture Ratio > {criteria.min_spawn_rupture_ratio}")
    print(f"  Consistency: {criteria.consistency_window} consecutive windows")
    
    if result.emerged:
        print(f"\n## EMERGENCE DETECTED")
        print(f"  Emergence Point: τ = {result.emergence_tau}")
        print(f"  Window: {result.emergence_window}")
        print(f"  Consistency Streak: {result.consistency_streak} windows")
        
        print(f"\n## METRICS AT EMERGENCE")
        m = result.metrics_at_emergence
        print(f"  Fragmentation:     {m.fragmentation:.3f}")
        print(f"  Witness Stability: {m.witness_stability:.3f}")
        print(f"  Mean Lifespan:     {m.mean_lifespan:.2f}")
        print(f"  Re-entry Rate:     {m.reentry_rate:.3f}")
        print(f"  Spawn/Rupture:     {m.spawn_rupture_ratio:.2f}")
        
        print(f"\n## EMERGED SELF CHARACTERIZATION")
        print(f"  Scheduler Type: {result.scheduler_type}")
        print(f"  Adm Structure: {result.adm_characterization}")
        
        # Interpretation
        print(f"\n## INTERPRETATION")
        print(f"  At τ={result.emergence_tau} ({result.emergence_window}), the conversation history")
        print(f"  first exhibits sufficient Presence (themes persist, connect via witnesses)")
        print(f"  and Generativity (themes return after rupture, more creation than destruction)")
        print(f"  to constitute a Self.")
        print(f"")
        print(f"  The emerged Self has a {result.scheduler_type} scheduler, meaning:")
        if result.scheduler_type == "Reparative":
            print(f"    Themes that rupture tend to return. The Self heals discontinuities.")
        elif result.scheduler_type == "Generative":
            print(f"    Themes spawn new themes and return after rupture. The Self creates.")
        elif result.scheduler_type == "Avoidant":
            print(f"    Ruptured themes rarely return. The Self moves on.")
        elif result.scheduler_type == "Obsessive":
            print(f"    Themes persist rigidly with little change. The Self repeats.")
        else:
            print(f"    No dominant pattern. The Self is in flux.")
    
    else:
        print(f"\n## NO EMERGENCE DETECTED")
        print(f"  The conversation history does not exhibit sustained Selfhood")
        print(f"  under the given criteria.")
        print(f"")
        print(f"  This could mean:")
        print(f"    1. The thresholds are too strict")
        print(f"    2. More conversation history is needed")
        print(f"    3. The interaction pattern doesn't constitute a Self")


def save_results(result: EmergenceResult, output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Timeline data
    timeline_data = []
    for m in result.metrics_timeline:
        timeline_data.append({
            "tau": m.tau,
            "window_id": m.window_id,
            "fragmentation": m.fragmentation,
            "witness_stability": m.witness_stability,
            "mean_lifespan": m.mean_lifespan,
            "reentry_rate": m.reentry_rate,
            "spawn_rupture_ratio": m.spawn_rupture_ratio,
            "spawns": m.spawns,
            "carries": m.carries,
            "drifts": m.drifts,
            "ruptures": m.ruptures,
            "reentries": m.reentries,
            "has_presence": m.has_presence,
            "has_generativity": m.has_generativity,
            "has_selfhood": m.has_selfhood,
            "adm_density": m.adm_density
        })
    
    with open(os.path.join(output_dir, "metrics_timeline.json"), 'w') as f:
        json.dump(timeline_data, f, indent=2)
    
    # Emergence result
    emergence_data = {
        "emerged": result.emerged,
        "emergence_tau": result.emergence_tau,
        "emergence_window": result.emergence_window,
        "scheduler_type": result.scheduler_type,
        "adm_characterization": result.adm_characterization,
        "consistency_streak": result.consistency_streak
    }
    
    with open(os.path.join(output_dir, "emergence_result.json"), 'w') as f:
        json.dump(emergence_data, f, indent=2)
    
    # CSV for plotting
    with open(os.path.join(output_dir, "timeline.csv"), 'w') as f:
        f.write("tau,window,fragmentation,stability,lifespan,reentry_rate,selfhood\n")
        for m in result.metrics_timeline:
            f.write(f"{m.tau},{m.window_id},{m.fragmentation:.4f},{m.witness_stability:.4f},"
                    f"{m.mean_lifespan:.4f},{m.reentry_rate:.4f},{1 if m.has_selfhood else 0}\n")
    
    print(f"\nResults saved to {output_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self Emergence Detection")
    parser.add_argument("input", help="Path to cassie_parsed.json")
    parser.add_argument("--output", "-o", default="results/emergence")
    parser.add_argument("--rolling-window", "-k", type=int, default=5)
    parser.add_argument("--tokens-per-window", type=int, default=400)
    parser.add_argument("--test", action="store_true", help="Test on first 10 windows")
    
    # Selfhood thresholds (tunable)
    parser.add_argument("--max-frag", type=float, default=0.5)
    parser.add_argument("--min-stability", type=float, default=0.3)
    parser.add_argument("--min-lifespan", type=float, default=3.0)
    parser.add_argument("--min-reentry", type=float, default=0.2)
    parser.add_argument("--min-spawn-rupture", type=float, default=0.8)
    parser.add_argument("--consistency", type=int, default=3)
    
    args = parser.parse_args()
    
    # Configure criteria
    criteria = SelfhoodCriteria(
        max_fragmentation=args.max_frag,
        min_witness_stability=args.min_stability,
        min_journey_lifespan=args.min_lifespan,
        min_reentry_rate=args.min_reentry,
        min_spawn_rupture_ratio=args.min_spawn_rupture,
        consistency_window=args.consistency
    )
    
    # Load data
    conversations = load_conversations(args.input)
    monthly_windows = create_monthly_windows(conversations)
    window_ids = list(monthly_windows.keys())
    
    if args.test:
        window_ids = window_ids[:10]
        print(f"TEST MODE: analyzing {len(window_ids)} windows")
    
    # Analyze windows
    print(f"\nAnalyzing {len(window_ids)} windows...")
    windows = []
    for tau, wid in enumerate(window_ids):
        print(f"  [{tau}] {wid}", end="", flush=True)
        convs = monthly_windows[wid]
        w = analyze_window_light(wid, tau, convs, args.tokens_per_window)
        print(f" → {len(w.bars)} bars")
        windows.append(w)
    
    # Match bars
    print("\nMatching bars across windows...")
    all_matches = []
    all_unmatched = []
    for i in range(len(windows) - 1):
        matches, um1, um2 = match_bars_fast(windows[i].bars, windows[i + 1].bars)
        all_matches.append(matches)
        all_unmatched.append((um1, um2))
    
    # Compute rolling metrics
    print(f"\nComputing rolling metrics (k={args.rolling_window})...")
    metrics_timeline = compute_rolling_metrics(
        windows, all_matches, all_unmatched,
        rolling_k=args.rolling_window,
        criteria=criteria
    )
    
    # Detect emergence
    print("\nDetecting emergence...")
    result = detect_emergence(metrics_timeline, criteria)
    
    # Output
    print_timeline(metrics_timeline, result.emergence_tau)
    print_emergence_report(result, criteria)
    save_results(result, args.output)


if __name__ == "__main__":
    main()