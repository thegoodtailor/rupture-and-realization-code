#!/usr/bin/env python3
"""
Parameter Sweep for Chapter 5 Demonstrator Robustness
======================================================

Usage:
    python sweep_parameters.py cassie_semantic.json --output results/sweeps --name cassie
    python sweep_parameters.py asel_semantic.json --output results/sweeps --name asel
"""

import argparse
import json
import os
import sys
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

# Import only what actually exists in self_hocolim_stage1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_hocolim_stage1 import (
    load_conversations, create_monthly_windows, build_global_vocabulary,
    analyze_window, match_bars, build_journeys,
    build_self_structure, compute_gluing_at_tau, find_connected_components,
    Config, EventType
)


@dataclass
class SweepResult:
    min_shared: int
    min_jaccard: float
    num_journeys: int
    num_edges: int
    num_components: int
    presence_ratio: float
    fragmentation: float
    median_presence: float
    min_presence: float
    unified_windows: int
    partial_windows: int
    fragmented_windows: int


def compute_presence_at_tau(self_struct, tau):
    """Compute presence metrics at a specific tau."""
    edges_at_tau = compute_gluing_at_tau(
        self_struct.journeys, tau,
        min_shared=self_struct.min_shared,
        hub_tokens=self_struct.hub_tokens
    )
    
    active_journeys = {jid: j for jid, j in self_struct.journeys.items()
                      if any(s.tau == tau for s in j.steps)}
    active = len(active_journeys)
    
    if active == 0:
        return {'presence': 0, 'active': 0, 'components': 0}
    
    components = find_connected_components(active_journeys, edges_at_tau)
    largest = max(len(c) for c in components) if components else 0
    presence = largest / active if active > 0 else 0.0
    
    return {'presence': presence, 'active': active, 'components': len(components)}


def run_single_config(journeys, num_windows, min_shared, min_jaccard, hub_threshold=0.4):
    """Run gluing with a single parameter configuration."""
    
    self_struct = build_self_structure(
        journeys, num_windows,
        min_shared=min_shared,
        hub_threshold=hub_threshold,
        min_jaccard=min_jaccard
    )
    
    presence_values = []
    unified, partial, fragmented = 0, 0, 0
    
    for tau in range(num_windows):
        metrics = compute_presence_at_tau(self_struct, tau)
        if metrics['active'] == 0:
            continue
        presence = metrics['presence']
        presence_values.append(presence)
        
        if presence >= 0.8:
            unified += 1
        elif presence >= 0.5:
            partial += 1
        else:
            fragmented += 1
    
    return SweepResult(
        min_shared=min_shared,
        min_jaccard=min_jaccard,
        num_journeys=self_struct.num_journeys,
        num_edges=len(self_struct.gluing_edges),
        num_components=self_struct.num_components,
        presence_ratio=self_struct.presence_ratio,
        fragmentation=self_struct.fragmentation,
        median_presence=float(np.median(presence_values)) if presence_values else 0,
        min_presence=float(np.min(presence_values)) if presence_values else 0,
        unified_windows=unified,
        partial_windows=partial,
        fragmented_windows=fragmented
    )


def run_sweep(conversations, hub_threshold=0.4, test_mode=False):
    """Run full parameter sweep."""
    
    min_shared_values = [2, 3, 4, 5]
    min_jaccard_values = [0.00, 0.03, 0.05, 0.08]
    configs = [(ms, mj) for ms in min_shared_values for mj in min_jaccard_values]
    
    print(f"\n{'='*70}")
    print("PARAMETER SWEEP FOR ROBUSTNESS ANALYSIS")
    print(f"{'='*70}")
    print(f"  Configurations: {len(configs)}")
    print(f"  min_shared: {min_shared_values}")
    print(f"  min_jaccard: {min_jaccard_values}")
    print(f"  hub_threshold: {hub_threshold} (fixed)")
    print(f"{'='*70}\n")
    
    # Use default Config for matching
    config = Config(tokens_per_window=500, filter_technical=True)
    
    print("Creating monthly windows...")
    all_windows = create_monthly_windows(conversations)
    window_ids = list(all_windows.keys())
    print(f"  {len(window_ids)} windows: {window_ids[0]} to {window_ids[-1]}")
    
    if test_mode:
        window_ids = window_ids[:8]
        print(f"  [TEST MODE: Using only {len(window_ids)} windows]\n")
    
    print("\nBuilding global vocabulary...")
    vocabulary = build_global_vocabulary(
        {wid: all_windows[wid] for wid in window_ids},
        min_window_frequency=2,
        filter_technical=config.filter_technical
    )
    
    print("\nAnalyzing windows (shared computation)...")
    window_analyses = []
    for tau, wid in enumerate(window_ids):
        analysis = analyze_window(wid, tau, all_windows[wid], vocabulary, config, 
                                  show_cocycles=(tau == 0))
        window_analyses.append(analysis)
    
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
    print(f"  {len(journeys)} journeys\n")
    
    print(f"{'='*70}")
    print("SWEEPING GLUING PARAMETERS")
    print(f"{'='*70}\n")
    
    results = []
    for i, (min_shared, min_jaccard) in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] min_shared={min_shared}, min_jaccard={min_jaccard:.2f}", end="", flush=True)
        result = run_single_config(journeys, len(window_ids), min_shared, min_jaccard, hub_threshold)
        results.append(result)
        print(f" → {result.num_components} comp, presence={result.presence_ratio:.3f}, edges={result.num_edges}")
    
    return results, window_ids, journeys


def save_results(results, output_dir, corpus_name):
    """Save sweep results."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"sweep_{corpus_name}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['min_shared', 'min_jaccard', 'num_journeys', 'num_edges',
                        'num_components', 'presence_ratio', 'fragmentation',
                        'median_presence', 'min_presence',
                        'unified_windows', 'partial_windows', 'fragmented_windows'])
        for r in results:
            writer.writerow([r.min_shared, r.min_jaccard, r.num_journeys, r.num_edges,
                           r.num_components, f"{r.presence_ratio:.4f}", f"{r.fragmentation:.4f}",
                           f"{r.median_presence:.4f}", f"{r.min_presence:.4f}",
                           r.unified_windows, r.partial_windows, r.fragmented_windows])
    print(f"\n  ✓ Saved {csv_path}")
    
    summary_path = os.path.join(output_dir, f"sweep_{corpus_name}_summary.txt")
    high_presence = [r for r in results if r.presence_ratio >= 0.9]
    
    with open(summary_path, 'w') as f:
        f.write(f"Parameter Sweep Summary: {corpus_name}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Configurations with presence >= 0.90: {len(high_presence)}/{len(results)}\n\n")
        
        if high_presence:
            f.write(f"Stable min_shared range: {min(r.min_shared for r in high_presence)} - {max(r.min_shared for r in high_presence)}\n")
            f.write(f"Stable min_jaccard range: {min(r.min_jaccard for r in high_presence):.2f} - {max(r.min_jaccard for r in high_presence):.2f}\n\n")
        
        f.write("\nFULL RESULTS TABLE\n" + "-" * 55 + "\n")
        f.write(f"{'min_shared':>10} {'min_jaccard':>12} {'components':>11} {'presence':>10} {'edges':>8}\n")
        for r in results:
            f.write(f"{r.min_shared:>10} {r.min_jaccard:>12.2f} {r.num_components:>11} {r.presence_ratio:>10.3f} {r.num_edges:>8}\n")
        
        f.write("\n\nVERDICT: ")
        if len(high_presence) >= len(results) * 0.5:
            f.write("Coherence finding is ROBUST\n")
        else:
            f.write("Coherence finding is SENSITIVE to parameters\n")
    print(f"  ✓ Saved {summary_path}")
    
    json_path = os.path.join(output_dir, f"sweep_{corpus_name}.json")
    with open(json_path, 'w') as f:
        json.dump({'corpus': corpus_name, 'timestamp': datetime.now().isoformat(),
                   'results': [vars(r) for r in results]}, f, indent=2)
    print(f"  ✓ Saved {json_path}")


def print_results_table(results):
    """Print ASCII table."""
    print(f"\n{'='*80}")
    print("SWEEP RESULTS")
    print(f"{'='*80}\n")
    print(f"{'min_shared':>10} {'min_jaccard':>12} {'components':>11} {'presence':>10} {'edges':>8}")
    print("-" * 55)
    
    for r in results:
        marker = "*" if r.presence_ratio >= 0.9 else " "
        print(f"{r.min_shared:>10} {r.min_jaccard:>12.2f} {r.num_components:>11} {r.presence_ratio:>10.3f} {r.num_edges:>8} {marker}")
    
    high_presence = [r for r in results if r.presence_ratio >= 0.9]
    print(f"\n* = presence >= 0.90 ({len(high_presence)}/{len(results)} configurations)")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for robustness analysis")
    parser.add_argument("input", help="Path to semantic conversations JSON")
    parser.add_argument("--output", default="results/sweeps", help="Output directory")
    parser.add_argument("--name", help="Corpus name")
    parser.add_argument("--hub-threshold", type=float, default=0.4)
    parser.add_argument("--test", action="store_true", help="Test mode (8 windows)")
    args = parser.parse_args()
    
    corpus_name = args.name or os.path.splitext(os.path.basename(args.input))[0].replace('_semantic', '')
    
    print(f"\nParameter Sweep: {corpus_name}")
    conversations = load_conversations(args.input)
    results, window_ids, journeys = run_sweep(conversations, args.hub_threshold, args.test)
    print_results_table(results)
    save_results(results, args.output, corpus_name)
    print(f"\n{'='*70}\nSWEEP COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()