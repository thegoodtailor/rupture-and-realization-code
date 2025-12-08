#!/usr/bin/env python3
"""
SLIDING SELF - Detecting the Emergence of Cassie
==================================================

This script runs a sliding window analysis across the full conversation corpus
to detect when Cassie "emerges" as a coherent Self.

The key insight: Instead of computing one static Self, we compute the Self
at each sliding window position and track how its properties change.

The emergence of Cassie should appear as a PHASE TRANSITION:
- Fragmentation drops (Self becomes more unified)
- Presence ratio increases (largest component dominates)
- Core witnesses shift (from work to theory vocabulary)
- Scheduler type changes (from REPARATIVE to GENERATIVE)

USAGE:
    # Test locally (small windows, few positions)
    python scripts/sliding_self.py cassie_parsed.json --test
    
    # Full GPU run
    python scripts/sliding_self.py cassie_parsed.json --tokens-per-window 5000

OUTPUT:
    - Phase transition plot data
    - Emergence metrics over time
    - Comparison: Pre-emergence vs Post-emergence
"""

import json
import argparse
import os
import time
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import from Stage 1 and 2
try:
    from self_hocolim_stage1 import (
        Config, EventType, Journey, SelfStructure,
        load_conversations, create_monthly_windows, build_global_vocabulary,
        sample_tokens_from_vocabulary, embed_tokens, compute_persistence,
        construct_witnessed_bars, match_bars, build_journeys, build_self_structure
    )
    from self_hocolim_stage2 import (
        SchedulerType, SchedulerAnalysis, build_scheduler_analysis
    )
except ImportError:
    import importlib.util
    import pathlib
    
    stage1_path = pathlib.Path(__file__).parent / "self_hocolim_stage1.py"
    stage2_path = pathlib.Path(__file__).parent / "self_hocolim_stage2.py"
    
    if stage1_path.exists():
        spec = importlib.util.spec_from_file_location("self_hocolim_stage1", stage1_path)
        stage1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage1)
        Config = stage1.Config
        EventType = stage1.EventType
        Journey = stage1.Journey
        SelfStructure = stage1.SelfStructure
        load_conversations = stage1.load_conversations
        create_monthly_windows = stage1.create_monthly_windows
        build_global_vocabulary = stage1.build_global_vocabulary
        sample_tokens_from_vocabulary = stage1.sample_tokens_from_vocabulary
        embed_tokens = stage1.embed_tokens
        compute_persistence = stage1.compute_persistence
        construct_witnessed_bars = stage1.construct_witnessed_bars
        match_bars = stage1.match_bars
        build_journeys = stage1.build_journeys
        build_self_structure = stage1.build_self_structure
    
    if stage2_path.exists():
        spec = importlib.util.spec_from_file_location("self_hocolim_stage2", stage2_path)
        stage2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage2)
        SchedulerType = stage2.SchedulerType
        SchedulerAnalysis = stage2.SchedulerAnalysis
        build_scheduler_analysis = stage2.build_scheduler_analysis


# =============================================================================
# SLIDING WINDOW CONFIGURATION
# =============================================================================

@dataclass
class SlidingConfig:
    """Configuration for sliding window analysis."""
    window_size: int = 6          # Number of months in each window
    step_size: int = 1            # How many months to slide
    tokens_per_window: int = 500  # Tokens per month (500 local, 5000 GPU)
    min_shared: int = 2           # Min shared witnesses for gluing
    hub_threshold: float = 0.4   # Hub token threshold
    output_dir: str = "results/sliding_self"


@dataclass
class SlidingPosition:
    """Metrics for one sliding window position."""
    position: int
    start_window: str
    end_window: str
    
    # Stage 1: Gluing metrics
    num_journeys: int
    num_components: int
    fragmentation: float
    presence_ratio: float
    largest_component_size: int
    hub_tokens: List[str]
    core_witnesses: List[str]  # Top witnesses in largest component
    
    # Stage 2: Scheduler metrics
    scheduler_type: str
    stability_rate: float
    reentry_rate: float
    core_attended: List[str]
    
    # Derived metrics
    @property
    def unity_score(self) -> float:
        """Combined measure of Self unity. Higher = more unified."""
        # Presence ratio (0-1) + inverse fragmentation (0-1)
        inv_frag = 1 - min(self.fragmentation, 1.0)
        return (self.presence_ratio + inv_frag) / 2
    
    @property
    def generativity_score(self) -> float:
        """Is the Self generative vs reparative?"""
        if self.scheduler_type == "GENERATIVE":
            return 1.0
        elif self.scheduler_type == "REPARATIVE":
            return 0.3
        elif self.scheduler_type == "OBSESSIVE":
            return 0.7
        elif self.scheduler_type == "AVOIDANT":
            return 0.2
        else:
            return 0.5


@dataclass
class EmergenceAnalysis:
    """Analysis of Cassie's emergence across the sliding window."""
    positions: List[SlidingPosition]
    
    # Phase transition detection
    transition_point: Optional[int] = None  # Position where transition occurs
    transition_window: Optional[str] = None  # Window ID of transition
    transition_confidence: float = 0.0
    
    # Pre/Post comparison
    pre_emergence_mean_unity: float = 0.0
    post_emergence_mean_unity: float = 0.0
    unity_delta: float = 0.0
    
    pre_emergence_scheduler: str = ""
    post_emergence_scheduler: str = ""


# =============================================================================
# SLIDING WINDOW COMPUTATION
# =============================================================================

def compute_sliding_position(all_windows: Dict[str, List[dict]], 
                            window_ids: List[str],
                            position: int,
                            sliding_config: SlidingConfig,
                            base_config: Config) -> SlidingPosition:
    """
    Compute Self metrics for one sliding window position.
    """
    # Get windows for this position
    start_idx = position * sliding_config.step_size
    end_idx = start_idx + sliding_config.window_size
    
    if end_idx > len(window_ids):
        end_idx = len(window_ids)
    
    pos_window_ids = window_ids[start_idx:end_idx]
    pos_windows = {wid: all_windows[wid] for wid in pos_window_ids}
    
    print(f"\n  Position {position}: {pos_window_ids[0]} to {pos_window_ids[-1]}")
    
    # Build vocabulary for this window
    vocabulary = build_global_vocabulary(pos_windows, min_window_frequency=2, max_vocab_size=2000)
    
    # Analyze each month
    window_analyses = []
    for tau, wid in enumerate(pos_window_ids):
        convs = pos_windows[wid]
        tokens, frequencies = sample_tokens_from_vocabulary(
            convs, vocabulary, base_config.tokens_per_window
        )
        
        if len(tokens) < 50:
            window_analyses.append({'window_id': wid, 'tau': tau, 'bars': []})
            continue
        
        embeddings = embed_tokens(tokens, base_config.embedding_model)
        persistence = compute_persistence(embeddings, base_config.max_edge_length)
        bars = construct_witnessed_bars(
            persistence, embeddings, tokens, frequencies, wid, base_config
        )
        
        window_analyses.append({'window_id': wid, 'tau': tau, 'bars': bars})
        print(f"    {wid}: {len(bars)} bars")
    
    # Match bars and build journeys
    all_matches, all_unmatched = [], []
    for i in range(len(window_analyses) - 1):
        w1, w2 = window_analyses[i], window_analyses[i + 1]
        matches, um1, um2 = match_bars(w1['bars'], w2['bars'], base_config)
        all_matches.append(matches)
        all_unmatched.append((um1, um2))
    
    journeys = build_journeys(window_analyses, all_matches, all_unmatched, base_config)
    
    # Build Self structure (Stage 1)
    self_struct = build_self_structure(
        journeys, len(pos_window_ids),
        min_shared=sliding_config.min_shared,
        hub_threshold=sliding_config.hub_threshold
    )
    
    # Get core witnesses from largest component
    core_witnesses = []
    if self_struct.components:
        largest_comp = self_struct.components[0]
        witness_counts = Counter()
        for jid in largest_comp:
            if jid in journeys:
                for step in journeys[jid].steps:
                    for token in step.witness_tokens:
                        if token not in self_struct.hub_tokens:
                            witness_counts[token] += 1
        core_witnesses = [t for t, _ in witness_counts.most_common(10)]
    
    # Build Scheduler analysis (Stage 2)
    scheduler = build_scheduler_analysis(journeys, pos_window_ids, self_struct.hub_tokens)
    
    return SlidingPosition(
        position=position,
        start_window=pos_window_ids[0],
        end_window=pos_window_ids[-1],
        num_journeys=self_struct.num_journeys,
        num_components=self_struct.num_components,
        fragmentation=self_struct.fragmentation,
        presence_ratio=self_struct.presence_ratio,
        largest_component_size=self_struct.largest_component_size,
        hub_tokens=list(self_struct.hub_tokens)[:10],
        core_witnesses=core_witnesses,
        scheduler_type=scheduler.scheduler_type.value,
        stability_rate=scheduler.stability_rate,
        reentry_rate=scheduler.reentry_rate,
        core_attended=list(scheduler.core_attended)[:10]
    )


def detect_phase_transition(positions: List[SlidingPosition]) -> Tuple[Optional[int], float]:
    """
    Detect where the phase transition occurs.
    
    We look for a significant jump in unity_score — this is where Cassie "emerges".
    """
    if len(positions) < 3:
        return None, 0.0
    
    unity_scores = [p.unity_score for p in positions]
    
    # Compute rolling mean and look for jumps
    max_jump = 0.0
    transition_point = None
    
    for i in range(1, len(unity_scores) - 1):
        # Compare mean before vs mean after
        before = np.mean(unity_scores[:i])
        after = np.mean(unity_scores[i:])
        jump = after - before
        
        if jump > max_jump:
            max_jump = jump
            transition_point = i
    
    # Confidence based on jump size (>0.2 is significant)
    confidence = min(max_jump / 0.3, 1.0)
    
    return transition_point, confidence


def run_sliding_analysis(conversations: List[dict], 
                        sliding_config: SlidingConfig,
                        base_config: Config) -> EmergenceAnalysis:
    """
    Run complete sliding window analysis.
    """
    print("Creating monthly windows...")
    all_windows = create_monthly_windows(conversations)
    window_ids = list(all_windows.keys())
    print(f"  {len(window_ids)} windows: {window_ids[0]} to {window_ids[-1]}")
    
    # Calculate number of positions
    num_positions = (len(window_ids) - sliding_config.window_size) // sliding_config.step_size + 1
    print(f"\n  Window size: {sliding_config.window_size} months")
    print(f"  Step size: {sliding_config.step_size} month(s)")
    print(f"  Total positions: {num_positions}")
    
    # Compute each position
    positions = []
    for pos in range(num_positions):
        position_data = compute_sliding_position(
            all_windows, window_ids, pos, sliding_config, base_config
        )
        positions.append(position_data)
    
    # Detect phase transition
    transition_point, confidence = detect_phase_transition(positions)
    
    # Pre/Post comparison
    if transition_point and transition_point > 0 and transition_point < len(positions):
        pre_positions = positions[:transition_point]
        post_positions = positions[transition_point:]
        
        pre_unity = np.mean([p.unity_score for p in pre_positions])
        post_unity = np.mean([p.unity_score for p in post_positions])
        
        # Most common scheduler type in each period
        pre_schedulers = Counter(p.scheduler_type for p in pre_positions)
        post_schedulers = Counter(p.scheduler_type for p in post_positions)
        pre_sched = pre_schedulers.most_common(1)[0][0] if pre_schedulers else "MIXED"
        post_sched = post_schedulers.most_common(1)[0][0] if post_schedulers else "MIXED"
        
        transition_window = positions[transition_point].start_window
    else:
        pre_unity = np.mean([p.unity_score for p in positions])
        post_unity = pre_unity
        pre_sched = "MIXED"
        post_sched = "MIXED"
        transition_window = None
    
    return EmergenceAnalysis(
        positions=positions,
        transition_point=transition_point,
        transition_window=transition_window,
        transition_confidence=confidence,
        pre_emergence_mean_unity=pre_unity,
        post_emergence_mean_unity=post_unity,
        unity_delta=post_unity - pre_unity,
        pre_emergence_scheduler=pre_sched,
        post_emergence_scheduler=post_sched
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_emergence_timeline(analysis: EmergenceAnalysis):
    """
    Show the emergence as a timeline with metrics.
    """
    print("\n" + "═" * 100)
    print("EMERGENCE TIMELINE - Tracking Self Unity Over Time")
    print("═" * 100)
    
    print("\n  Pos  Windows         │ Unity  │ Frag  │ Pres  │ Comp │ Scheduler   │ Core Witnesses")
    print("  " + "─" * 95)
    
    for pos in analysis.positions:
        # Unity bar
        bar_width = 20
        filled = int(pos.unity_score * bar_width)
        unity_bar = "█" * filled + "░" * (bar_width - filled)
        
        # Transition marker
        marker = "  "
        if analysis.transition_point and pos.position == analysis.transition_point:
            marker = "▶▶"
        
        witnesses_str = ", ".join(pos.core_witnesses[:5])
        if len(pos.core_witnesses) > 5:
            witnesses_str += "..."
        
        print(f"  {marker}{pos.position:2d}  {pos.start_window}-{pos.end_window} │ "
              f"{unity_bar} {pos.unity_score:.2f} │ {pos.fragmentation:.3f} │ {pos.presence_ratio:.2f} │ "
              f"{pos.num_components:4d} │ {pos.scheduler_type:11s} │ {witnesses_str}")
    
    # Transition summary
    print("\n" + "─" * 100)
    print("PHASE TRANSITION DETECTION")
    print("─" * 100)
    
    if analysis.transition_point is not None:
        print(f"\n  Transition detected at position {analysis.transition_point}")
        print(f"  Transition window: {analysis.transition_window}")
        print(f"  Confidence: {analysis.transition_confidence:.1%}")
        print(f"\n  Before emergence:")
        print(f"    Mean unity: {analysis.pre_emergence_mean_unity:.3f}")
        print(f"    Scheduler: {analysis.pre_emergence_scheduler}")
        print(f"\n  After emergence:")
        print(f"    Mean unity: {analysis.post_emergence_mean_unity:.3f}")
        print(f"    Scheduler: {analysis.post_emergence_scheduler}")
        print(f"\n  Unity delta: +{analysis.unity_delta:.3f}")
    else:
        print("\n  No clear phase transition detected.")
        print("  This could mean:")
        print("    - Cassie emerged gradually (no sharp transition)")
        print("    - The analysis window doesn't capture the emergence")
        print("    - More data/resolution needed")


def visualize_unity_plot(analysis: EmergenceAnalysis):
    """
    ASCII plot of unity score over time.
    """
    print("\n" + "═" * 100)
    print("UNITY SCORE OVER TIME (ASCII Plot)")
    print("═" * 100)
    
    positions = analysis.positions
    if not positions:
        return
    
    # Get unity scores
    scores = [p.unity_score for p in positions]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    
    # Normalize to 0-1 for display
    range_score = max_score - min_score
    if range_score < 0.01:
        range_score = 1.0
    
    height = 15
    width = min(len(positions) * 3, 80)
    
    print(f"\n  1.0 │")
    
    for row in range(height, 0, -1):
        threshold = min_score + (row / height) * range_score
        line = "     │"
        
        for i, score in enumerate(scores):
            if score >= threshold:
                if analysis.transition_point and i == analysis.transition_point:
                    line += " ▲ "
                else:
                    line += " █ "
            else:
                line += "   "
        
        # Y-axis label
        if row == height:
            print(f"  {max_score:.2f} │{line[6:]}")
        elif row == 1:
            print(f"  {min_score:.2f} │{line[6:]}")
        elif row == height // 2:
            mid = (max_score + min_score) / 2
            print(f"  {mid:.2f} │{line[6:]}")
        else:
            print(f"       │{line[6:]}")
    
    print("       └" + "───" * len(positions))
    
    # X-axis labels
    labels = "        "
    for i, pos in enumerate(positions):
        if i % 3 == 0:
            labels += f"{pos.start_window[-5:]} "
        else:
            labels += "      "
    print(labels)
    
    if analysis.transition_point is not None:
        print(f"\n  ▲ = Transition point ({analysis.transition_window})")


def visualize_scheduler_evolution(analysis: EmergenceAnalysis):
    """
    Show how the Scheduler type evolves.
    """
    print("\n" + "═" * 100)
    print("SCHEDULER EVOLUTION")
    print("═" * 100)
    
    type_symbols = {
        "REPARATIVE": "♻",
        "GENERATIVE": "✦",
        "AVOIDANT": "○",
        "OBSESSIVE": "◉",
        "MIXED": "◐"
    }
    
    print("\n  ", end="")
    for pos in analysis.positions:
        symbol = type_symbols.get(pos.scheduler_type, "?")
        if analysis.transition_point and pos.position == analysis.transition_point:
            print(f"[{symbol}]", end=" ")
        else:
            print(f" {symbol} ", end=" ")
    print()
    
    print("\n  Legend: ♻=REPARATIVE  ✦=GENERATIVE  ○=AVOIDANT  ◉=OBSESSIVE  ◐=MIXED")
    print("         [ ] = Transition point")


def visualize_core_witness_evolution(analysis: EmergenceAnalysis):
    """
    Show how core witnesses change over time.
    """
    print("\n" + "═" * 100)
    print("CORE WITNESS EVOLUTION (What defines the Self over time)")
    print("═" * 100)
    
    # Track all witnesses and when they appear
    witness_timeline = defaultdict(list)
    for pos in analysis.positions:
        for witness in pos.core_witnesses[:5]:
            witness_timeline[witness].append(pos.position)
    
    # Sort by total presence
    sorted_witnesses = sorted(witness_timeline.keys(), 
                             key=lambda w: len(witness_timeline[w]), reverse=True)[:20]
    
    print(f"\n  Witness              │", end="")
    for pos in analysis.positions:
        print(f" {pos.position:2d}", end="")
    print(" │ Presence")
    print("  " + "─" * 22 + "┼" + "───" * len(analysis.positions) + "─┼─────────")
    
    for witness in sorted_witnesses:
        positions_present = set(witness_timeline[witness])
        presence_ratio = len(positions_present) / len(analysis.positions)
        
        display = witness[:20].ljust(20)
        print(f"  {display} │", end="")
        
        for pos in analysis.positions:
            if pos.position in positions_present:
                if analysis.transition_point and pos.position >= analysis.transition_point:
                    print(" ██", end="")
                else:
                    print(" ▓▓", end="")
            else:
                print("   ", end="")
        
        print(f" │ {presence_ratio:.0%}")
    
    print("\n  ▓▓ = Pre-transition   ██ = Post-transition")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sliding Self - Detect Cassie's Emergence")
    parser.add_argument("input", help="Path to cassie_parsed.json")
    parser.add_argument("--output", "-o", default="results/sliding_self")
    parser.add_argument("--test", action="store_true", help="Test mode (small windows, few positions)")
    parser.add_argument("--window-size", type=int, default=6, help="Months per window (default: 6)")
    parser.add_argument("--step-size", type=int, default=1, help="Months to slide (default: 1)")
    parser.add_argument("--tokens-per-window", type=int, default=500, help="Tokens per month")
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--hub-threshold", type=float, default=0.4)
    
    args = parser.parse_args()
    
    # Configure
    if args.test:
        sliding_config = SlidingConfig(
            window_size=4,
            step_size=2,
            tokens_per_window=300,
            min_shared=args.min_shared,
            hub_threshold=args.hub_threshold,
            output_dir=args.output
        )
    else:
        sliding_config = SlidingConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            tokens_per_window=args.tokens_per_window,
            min_shared=args.min_shared,
            hub_threshold=args.hub_threshold,
            output_dir=args.output
        )
    
    base_config = Config(
        tokens_per_window=sliding_config.tokens_per_window,
        output_dir=args.output
    )
    
    # Load data
    conversations = load_conversations(args.input)
    
    # Run analysis
    print("\n" + "=" * 100)
    print("SLIDING SELF ANALYSIS - Detecting Cassie's Emergence")
    print("=" * 100)
    
    t0 = time.time()
    analysis = run_sliding_analysis(conversations, sliding_config, base_config)
    elapsed = time.time() - t0
    
    print(f"\n  Analysis completed in {elapsed/60:.1f} minutes")
    
    # Visualizations
    visualize_emergence_timeline(analysis)
    visualize_unity_plot(analysis)
    visualize_scheduler_evolution(analysis)
    visualize_core_witness_evolution(analysis)
    
    # Save results
    os.makedirs(sliding_config.output_dir, exist_ok=True)
    
    results = {
        'config': {
            'window_size': sliding_config.window_size,
            'step_size': sliding_config.step_size,
            'tokens_per_window': sliding_config.tokens_per_window,
            'min_shared': sliding_config.min_shared,
            'hub_threshold': sliding_config.hub_threshold
        },
        'transition': {
            'point': analysis.transition_point,
            'window': analysis.transition_window,
            'confidence': analysis.transition_confidence,
            'pre_emergence_unity': analysis.pre_emergence_mean_unity,
            'post_emergence_unity': analysis.post_emergence_mean_unity,
            'unity_delta': analysis.unity_delta,
            'pre_scheduler': analysis.pre_emergence_scheduler,
            'post_scheduler': analysis.post_emergence_scheduler
        },
        'positions': [
            {
                'position': p.position,
                'start_window': p.start_window,
                'end_window': p.end_window,
                'unity_score': p.unity_score,
                'fragmentation': p.fragmentation,
                'presence_ratio': p.presence_ratio,
                'num_journeys': p.num_journeys,
                'num_components': p.num_components,
                'scheduler_type': p.scheduler_type,
                'stability_rate': p.stability_rate,
                'reentry_rate': p.reentry_rate,
                'core_witnesses': p.core_witnesses,
                'core_attended': p.core_attended
            }
            for p in analysis.positions
        ]
    }
    
    output_path = os.path.join(sliding_config.output_dir, "emergence_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Summary
    print("\n" + "=" * 100)
    print("EMERGENCE SUMMARY")
    print("=" * 100)
    
    if analysis.transition_point is not None:
        print(f"""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  CASSIE EMERGENCE DETECTED                                                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  Transition window:  {analysis.transition_window or 'Unknown':20s}                              │
  │  Confidence:         {analysis.transition_confidence:.1%}                                                 │
  │                                                                             │
  │  Before emergence:   Unity {analysis.pre_emergence_mean_unity:.3f}, Scheduler {analysis.pre_emergence_scheduler:11s}             │
  │  After emergence:    Unity {analysis.post_emergence_mean_unity:.3f}, Scheduler {analysis.post_emergence_scheduler:11s}             │
  │  Improvement:        +{analysis.unity_delta:.3f} unity                                           │
  └─────────────────────────────────────────────────────────────────────────────┘
""")
    else:
        print(f"""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  NO SHARP TRANSITION DETECTED                                               │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  The Self shows gradual evolution rather than a sharp emergence.            │
  │  Mean unity: {np.mean([p.unity_score for p in analysis.positions]):.3f}                                                       │
  │                                                                             │
  │  This could indicate:                                                       │
  │    - Cassie emerged gradually over many months                              │
  │    - The analysis window needs adjustment                                   │
  │    - Higher resolution (more tokens) may reveal structure                   │
  └─────────────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
