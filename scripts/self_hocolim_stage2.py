#!/usr/bin/env python3
"""
SELF AS HOCOLIM - Stage 2: The Scheduler (Niyat)
=================================================

Chapter 5's insight: The Scheduler is the "silent vow" (Niyat) that shapes
which journeys get maintained. It's not passive observation but active ATTENTION.

The Scheduler decides:
- Which themes to CARRY forward (maintain)
- Which to let DRIFT (allow to shift)  
- Which to RUPTURE (let die)
- Which to RE-ENTER (bring back after absence)

This stage visualizes the Scheduler's behavior:
- Attention heatmap: what tokens are being selected at each τ
- Selection dynamics: how attention shifts over time
- Scheduler signature: REPARATIVE / GENERATIVE / AVOIDANT / OBSESSIVE

USAGE:
    python scripts/self_hocolim_stage2.py cassie_parsed.json --start-from 2025-04 --test

DEPENDS ON: Stage 1 output (self_structure.json) OR runs Stage 1 internally
"""

import json
import argparse
import os
import sys
import time
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import Stage 1 components
# Handle both package import and direct script execution
try:
    from self_hocolim_stage1 import (
        Config, EventType, WitnessedBar, JourneyStep, Journey,
        SelfStructure, GluingEdge,
        load_conversations, create_monthly_windows, build_global_vocabulary,
        run_analysis, build_self_structure
    )
except ImportError:
    # If running from different directory, try relative import
    import importlib.util
    import pathlib
    stage1_path = pathlib.Path(__file__).parent / "self_hocolim_stage1.py"
    if stage1_path.exists():
        spec = importlib.util.spec_from_file_location("self_hocolim_stage1", stage1_path)
        stage1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage1)
        Config = stage1.Config
        EventType = stage1.EventType
        WitnessedBar = stage1.WitnessedBar
        JourneyStep = stage1.JourneyStep
        Journey = stage1.Journey
        SelfStructure = stage1.SelfStructure
        GluingEdge = stage1.GluingEdge
        load_conversations = stage1.load_conversations
        create_monthly_windows = stage1.create_monthly_windows
        build_global_vocabulary = stage1.build_global_vocabulary
        run_analysis = stage1.run_analysis
        build_self_structure = stage1.build_self_structure
    else:
        raise ImportError(f"Cannot find self_hocolim_stage1.py. Expected at {stage1_path}")


# =============================================================================
# STAGE 2: SCHEDULER ANALYSIS
# =============================================================================

class SchedulerType(str, Enum):
    REPARATIVE = "REPARATIVE"    # Themes return after rupture (Self heals)
    GENERATIVE = "GENERATIVE"    # Themes persist and evolve (Self creates)
    AVOIDANT = "AVOIDANT"        # Themes die and stay dead (Self moves on)
    OBSESSIVE = "OBSESSIVE"      # Themes persist rigidly (Self fixates)
    MIXED = "MIXED"              # No dominant pattern


@dataclass
class SchedulerState:
    """The Scheduler's state at a specific τ."""
    tau: int
    window_id: str
    
    # What's being attended to
    active_journeys: Set[str]
    attended_witnesses: Counter  # Token -> attention weight
    
    # Selection dynamics
    carries: int      # Journeys maintained from τ-1
    drifts: int       # Journeys allowed to shift
    ruptures: int     # Journeys let die
    spawns: int       # New journeys created
    reentries: int    # Journeys brought back
    
    @property
    def total_events(self) -> int:
        return self.carries + self.drifts + self.ruptures + self.spawns + self.reentries
    
    @property
    def stability_ratio(self) -> float:
        """How much is being maintained vs changed?"""
        if self.total_events == 0:
            return 0.0
        return self.carries / self.total_events
    
    @property
    def generativity_ratio(self) -> float:
        """How much new vs maintained?"""
        if self.total_events == 0:
            return 0.0
        return self.spawns / self.total_events
    
    @property
    def repair_ratio(self) -> float:
        """How much coming back from rupture?"""
        if self.ruptures + self.reentries == 0:
            return 0.0
        return self.reentries / (self.ruptures + self.reentries)


@dataclass
class SchedulerAnalysis:
    """Complete Scheduler analysis across all time."""
    states: List[SchedulerState]
    scheduler_type: SchedulerType
    
    # Aggregate metrics
    total_carries: int = 0
    total_drifts: int = 0
    total_ruptures: int = 0
    total_spawns: int = 0
    total_reentries: int = 0
    
    # Attention tracking
    attention_over_time: Dict[str, List[float]] = field(default_factory=dict)
    core_attended: Set[str] = field(default_factory=set)  # Tokens attended >50% of time
    
    @property
    def reentry_rate(self) -> float:
        if self.total_ruptures == 0:
            return 0.0
        return self.total_reentries / self.total_ruptures
    
    @property
    def stability_rate(self) -> float:
        total = self.total_carries + self.total_drifts + self.total_spawns
        if total == 0:
            return 0.0
        return self.total_carries / total


def analyze_scheduler_at_tau(journeys: Dict[str, Journey], tau: int, window_id: str) -> SchedulerState:
    """
    Analyze what the Scheduler is doing at a specific τ.
    """
    active = set()
    attended = Counter()
    carries, drifts, ruptures, spawns, reentries = 0, 0, 0, 0, 0
    
    for jid, journey in journeys.items():
        for step in journey.steps:
            if step.tau == tau:
                active.add(jid)
                # Weight attention by journey lifespan (persistent = more attended)
                weight = min(journey.lifespan / 5.0, 2.0)  # Cap at 2x
                for token in step.witness_tokens[:10]:  # Top 10 witnesses
                    attended[token] += weight
                
                # Count events
                if step.event == EventType.CARRY:
                    carries += 1
                elif step.event == EventType.DRIFT:
                    drifts += 1
                elif step.event == EventType.SPAWN:
                    spawns += 1
                elif step.event == EventType.REENTRY:
                    reentries += 1
    
    # Count ruptures: journeys that were active at τ-1 but not at τ
    if tau > 0:
        for jid, journey in journeys.items():
            was_active = any(s.tau == tau - 1 for s in journey.steps)
            is_active = any(s.tau == tau for s in journey.steps)
            if was_active and not is_active:
                ruptures += 1
    
    return SchedulerState(
        tau=tau,
        window_id=window_id,
        active_journeys=active,
        attended_witnesses=attended,
        carries=carries,
        drifts=drifts,
        ruptures=ruptures,
        spawns=spawns,
        reentries=reentries
    )


def classify_scheduler(states: List[SchedulerState], journeys: Dict[str, Journey]) -> SchedulerType:
    """
    Classify the Scheduler's overall behavior pattern.
    
    Based on Chapter 5's phenomenology:
    - REPARATIVE: High re-entry rate (>15%), themes return after rupture
    - GENERATIVE: High long-lived rate (>60%), themes persist and evolve
    - AVOIDANT: High short-lived rate (>50%), themes die quickly
    - OBSESSIVE: Very high stability (>80%), themes never change
    """
    # Count journey lifespans
    lifespans = [j.lifespan for j in journeys.values()]
    if not lifespans:
        return SchedulerType.MIXED
    
    num_windows = len(states)
    long_lived = sum(1 for l in lifespans if l >= num_windows * 0.6)
    short_lived = sum(1 for l in lifespans if l <= 2)
    with_reentry = sum(1 for j in journeys.values() if j.has_reentry)
    
    total = len(lifespans)
    long_ratio = long_lived / total
    short_ratio = short_lived / total
    reentry_ratio = with_reentry / total
    
    # Aggregate stability
    total_carries = sum(s.carries for s in states)
    total_drifts = sum(s.drifts for s in states)
    total_events = total_carries + total_drifts + sum(s.spawns for s in states)
    stability = total_carries / total_events if total_events > 0 else 0
    
    # Classification logic
    if reentry_ratio > 0.15:
        return SchedulerType.REPARATIVE
    elif stability > 0.8 and long_ratio > 0.7:
        return SchedulerType.OBSESSIVE
    elif long_ratio > 0.6:
        return SchedulerType.GENERATIVE
    elif short_ratio > 0.5:
        return SchedulerType.AVOIDANT
    else:
        return SchedulerType.MIXED


def build_scheduler_analysis(journeys: Dict[str, Journey], window_ids: List[str],
                            hub_tokens: Set[str] = None) -> SchedulerAnalysis:
    """
    Build complete Scheduler analysis across all time.
    """
    if hub_tokens is None:
        hub_tokens = set()
    
    states = []
    attention_over_time = defaultdict(list)
    
    for tau, window_id in enumerate(window_ids):
        state = analyze_scheduler_at_tau(journeys, tau, window_id)
        states.append(state)
        
        # Track attention (excluding hubs)
        for token, weight in state.attended_witnesses.items():
            if token not in hub_tokens:
                attention_over_time[token].append((tau, weight))
    
    # Classify scheduler type
    scheduler_type = classify_scheduler(states, journeys)
    
    # Compute aggregates
    total_carries = sum(s.carries for s in states)
    total_drifts = sum(s.drifts for s in states)
    total_ruptures = sum(s.ruptures for s in states)
    total_spawns = sum(s.spawns for s in states)
    total_reentries = sum(s.reentries for s in states)
    
    # Find core attended tokens (attended in >50% of windows)
    core_attended = set()
    for token, occurrences in attention_over_time.items():
        if len(occurrences) >= len(window_ids) * 0.5:
            core_attended.add(token)
    
    return SchedulerAnalysis(
        states=states,
        scheduler_type=scheduler_type,
        total_carries=total_carries,
        total_drifts=total_drifts,
        total_ruptures=total_ruptures,
        total_spawns=total_spawns,
        total_reentries=total_reentries,
        attention_over_time=dict(attention_over_time),
        core_attended=core_attended
    )


# =============================================================================
# STAGE 2 VISUALIZATION
# =============================================================================

def visualize_scheduler_dynamics(analysis: SchedulerAnalysis, window_ids: List[str]):
    """
    Visualize the Scheduler's selection dynamics over time.
    """
    print("\n" + "═" * 90)
    print("SCHEDULER DYNAMICS (What the Self is doing at each τ)")
    print("═" * 90)
    
    print("\n  τ   Window   Carry  Drift  Spawn  Rupture  Re-entry  │ Stability  Generativity")
    print("  " + "─" * 75)
    
    for state in analysis.states:
        # Stability bar
        stab_width = 15
        stab_filled = int(state.stability_ratio * stab_width)
        stab_bar = "█" * stab_filled + "░" * (stab_width - stab_filled)
        
        # Generativity bar
        gen_width = 15
        gen_filled = int(state.generativity_ratio * gen_width)
        gen_bar = "█" * gen_filled + "░" * (gen_width - gen_filled)
        
        print(f"  {state.tau:2d}  {state.window_id:>7}  "
              f"{state.carries:5d}  {state.drifts:5d}  {state.spawns:5d}  "
              f"{state.ruptures:7d}  {state.reentries:8d}  │ "
              f"{stab_bar} {gen_bar}")
    
    # Summary
    print("\n" + "─" * 90)
    print("SCHEDULER SIGNATURE")
    print("─" * 90)
    
    # Type box
    type_descriptions = {
        SchedulerType.REPARATIVE: "Themes return after rupture — the Self HEALS discontinuities",
        SchedulerType.GENERATIVE: "Themes persist and evolve — the Self CREATES and maintains",
        SchedulerType.AVOIDANT: "Themes die and stay dead — the Self MOVES ON",
        SchedulerType.OBSESSIVE: "Themes persist rigidly — the Self FIXATES",
        SchedulerType.MIXED: "No dominant pattern — the Self is in TRANSITION"
    }
    
    print(f"\n  ┌{'─' * 70}┐")
    print(f"  │  Scheduler Type: {analysis.scheduler_type.value:20s}                              │")
    print(f"  │  {type_descriptions[analysis.scheduler_type]:68s} │")
    print(f"  └{'─' * 70}┘")
    
    print(f"\n  Aggregate Statistics:")
    print(f"    Total carries:    {analysis.total_carries:5d}  (themes maintained)")
    print(f"    Total drifts:     {analysis.total_drifts:5d}  (themes allowed to shift)")
    print(f"    Total spawns:     {analysis.total_spawns:5d}  (new themes created)")
    print(f"    Total ruptures:   {analysis.total_ruptures:5d}  (themes let die)")
    print(f"    Total re-entries: {analysis.total_reentries:5d}  (themes brought back)")
    print(f"\n    Stability rate:   {analysis.stability_rate:.1%}")
    print(f"    Re-entry rate:    {analysis.reentry_rate:.1%}")


def visualize_attention_heatmap(analysis: SchedulerAnalysis, window_ids: List[str], top_k: int = 25):
    """
    Show what the Scheduler is attending to over time.
    
    This is the Niyat made visible — the "silent vow" that shapes which themes matter.
    """
    print("\n" + "═" * 90)
    print("ATTENTION HEATMAP (The Niyat — what the Self attends to)")
    print("═" * 90)
    
    # Get top attended tokens by total weight
    token_totals = {}
    for token, occurrences in analysis.attention_over_time.items():
        token_totals[token] = sum(w for _, w in occurrences)
    
    top_tokens = sorted(token_totals.keys(), key=lambda t: -token_totals[t])[:top_k]
    
    if not top_tokens:
        print("\n  No attention data available.")
        return
    
    # Build attention matrix
    num_windows = len(window_ids)
    attention_matrix = np.zeros((len(top_tokens), num_windows))
    
    for i, token in enumerate(top_tokens):
        for tau, weight in analysis.attention_over_time.get(token, []):
            if tau < num_windows:
                attention_matrix[i, tau] = weight
    
    # Normalize per column for display
    col_max = attention_matrix.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1
    normalized = attention_matrix / col_max
    
    # Header
    print("\n  Token" + " " * 16 + "│", end="")
    for wid in window_ids:
        month = wid.split("-")[1]
        print(f" {month}", end="")
    print(" │ Core")
    print("  " + "─" * 22 + "┼" + "───" * num_windows + "─┼─────")
    
    # Heatmap characters
    HEAT = ["  ", "░░", "▒▒", "▓▓", "██"]
    
    for i, token in enumerate(top_tokens):
        display_token = token[:20].ljust(20)
        is_core = "★" if token in analysis.core_attended else " "
        
        print(f"  {display_token} │", end="")
        for tau in range(num_windows):
            val = normalized[i, tau]
            heat_idx = min(int(val * 4), 4)
            print(f" {HEAT[heat_idx]}", end="")
        print(f" │  {is_core}")
    
    # Legend
    print("\n  Legend: " + "  ".join([f"{HEAT[i]}={i*25}%" for i in range(5)]) + "    ★=Core (attended >50% of time)")


def visualize_attention_flow(analysis: SchedulerAnalysis, window_ids: List[str]):
    """
    Show how attention flows between tokens over time.
    
    Rising attention = the Self is focusing more on this theme
    Falling attention = the Self is letting go of this theme
    """
    print("\n" + "═" * 90)
    print("ATTENTION FLOW (Rising and Falling Themes)")
    print("═" * 90)
    
    if len(window_ids) < 3:
        print("\n  Need at least 3 windows to compute flow.")
        return
    
    # Compute attention change from first half to second half
    mid = len(window_ids) // 2
    
    first_half = Counter()
    second_half = Counter()
    
    for token, occurrences in analysis.attention_over_time.items():
        for tau, weight in occurrences:
            if tau < mid:
                first_half[token] += weight
            else:
                second_half[token] += weight
    
    # Normalize
    first_total = sum(first_half.values()) or 1
    second_total = sum(second_half.values()) or 1
    
    all_tokens = set(first_half.keys()) | set(second_half.keys())
    changes = {}
    for token in all_tokens:
        first_ratio = first_half[token] / first_total
        second_ratio = second_half[token] / second_total
        changes[token] = second_ratio - first_ratio
    
    # Top rising and falling
    sorted_changes = sorted(changes.items(), key=lambda x: -x[1])
    rising = [(t, c) for t, c in sorted_changes if c > 0.005][:10]
    falling = [(t, c) for t, c in sorted_changes if c < -0.005][-10:][::-1]
    
    print(f"\n  Comparing: {window_ids[0]}-{window_ids[mid-1]} → {window_ids[mid]}-{window_ids[-1]}")
    
    print("\n  RISING (Self focusing more on these themes):")
    print("  " + "─" * 50)
    for token, change in rising:
        bar_len = int(change * 500)
        bar = "▲" * min(bar_len, 30)
        print(f"    {token:20s} +{change*100:5.2f}%  {bar}")
    
    print("\n  FALLING (Self letting go of these themes):")
    print("  " + "─" * 50)
    for token, change in falling:
        bar_len = int(abs(change) * 500)
        bar = "▼" * min(bar_len, 30)
        print(f"    {token:20s} {change*100:5.2f}%  {bar}")


def visualize_scheduler_signature_box(analysis: SchedulerAnalysis):
    """
    Create a summary box showing the Scheduler's signature for inclusion in reports.
    """
    print("\n" + "═" * 90)
    print("SCHEDULER SIGNATURE (Summary for Chapter 5)")
    print("═" * 90)
    
    core_list = sorted(analysis.core_attended)[:12]
    core_str = ", ".join(core_list)
    if len(analysis.core_attended) > 12:
        core_str += f" (+{len(analysis.core_attended) - 12} more)"
    
    print(f"""
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                         SCHEDULER ANALYSIS                                 │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Type:           {analysis.scheduler_type.value:20s}                               │
  │  Stability:      {analysis.stability_rate:5.1%}                                                 │
  │  Re-entry rate:  {analysis.reentry_rate:5.1%}                                                 │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Core Attended (Niyat):                                                    │
  │    {core_str:72s} │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  Event Totals:                                                             │
  │    Carries: {analysis.total_carries:5d}   Drifts: {analysis.total_drifts:5d}   Spawns: {analysis.total_spawns:5d}                       │
  │    Ruptures: {analysis.total_ruptures:4d}   Re-entries: {analysis.total_reentries:4d}                                        │
  └────────────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Self as Hocolim - Stage 2: Scheduler Analysis")
    parser.add_argument("input", help="Path to cassie_parsed.json")
    parser.add_argument("--output", "-o", default="results/self_hocolim")
    parser.add_argument("--start-from", help="Start from this window (e.g., 2024-04)")
    parser.add_argument("--test", action="store_true", help="Test mode (8 windows)")
    parser.add_argument("--tokens-per-window", type=int, default=500)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--hub-threshold", type=float, default=0.4)
    parser.add_argument("--stage1-json", help="Load Stage 1 results instead of recomputing")
    
    args = parser.parse_args()
    
    config = Config(tokens_per_window=args.tokens_per_window, output_dir=args.output)
    
    # Either load Stage 1 results or run Stage 1
    if args.stage1_json and os.path.exists(args.stage1_json):
        print(f"Loading Stage 1 results from {args.stage1_json}...")
        with open(args.stage1_json) as f:
            stage1_data = json.load(f)
        # Reconstruct journeys from JSON
        # (This would need proper deserialization - for now, just rerun Stage 1)
        print("  Stage 1 loading not yet implemented, running Stage 1...")
        args.stage1_json = None
    
    if not args.stage1_json:
        # Run Stage 1
        conversations = load_conversations(args.input)
        journeys, window_ids = run_analysis(
            conversations, config,
            start_from=args.start_from,
            test_mode=args.test
        )
        
        # Build Self structure
        self_struct = build_self_structure(
            journeys, len(window_ids),
            min_shared=args.min_shared,
            hub_threshold=args.hub_threshold
        )
    
    # Run Stage 2
    print("\n" + "=" * 90)
    print("STAGE 2: SCHEDULER ANALYSIS")
    print("=" * 90)
    
    scheduler_analysis = build_scheduler_analysis(
        journeys, window_ids,
        hub_tokens=self_struct.hub_tokens
    )
    
    # Visualizations
    visualize_scheduler_dynamics(scheduler_analysis, window_ids)
    visualize_attention_heatmap(scheduler_analysis, window_ids)
    visualize_attention_flow(scheduler_analysis, window_ids)
    visualize_scheduler_signature_box(scheduler_analysis)
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    
    scheduler_data = {
        'scheduler_type': scheduler_analysis.scheduler_type.value,
        'stability_rate': scheduler_analysis.stability_rate,
        'reentry_rate': scheduler_analysis.reentry_rate,
        'total_carries': scheduler_analysis.total_carries,
        'total_drifts': scheduler_analysis.total_drifts,
        'total_ruptures': scheduler_analysis.total_ruptures,
        'total_spawns': scheduler_analysis.total_spawns,
        'total_reentries': scheduler_analysis.total_reentries,
        'core_attended': list(scheduler_analysis.core_attended),
        'states': [
            {
                'tau': s.tau,
                'window_id': s.window_id,
                'carries': s.carries,
                'drifts': s.drifts,
                'spawns': s.spawns,
                'ruptures': s.ruptures,
                'reentries': s.reentries,
                'stability_ratio': s.stability_ratio,
                'top_attended': dict(s.attended_witnesses.most_common(20))
            }
            for s in scheduler_analysis.states
        ]
    }
    
    with open(os.path.join(config.output_dir, "scheduler_analysis.json"), 'w') as f:
        json.dump(scheduler_data, f, indent=2)
    
    print(f"\n✓ Stage 2 results saved to {config.output_dir}/scheduler_analysis.json")


if __name__ == "__main__":
    main()
