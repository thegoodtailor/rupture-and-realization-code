#!/usr/bin/env python3
"""
Negative Control: Witness Shuffle
=================================

Tests whether the observed Self-coherence is meaningful or a trivial artifact.

Usage:
    python negative_control.py cassie_semantic.json --output results/controls --name cassie
    python negative_control.py asel_semantic.json --output results/controls --name asel
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Set
import numpy as np

# Import only what actually exists
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_hocolim_stage1 import (
    load_conversations, create_monthly_windows, build_global_vocabulary,
    analyze_window, match_bars, build_journeys,
    build_self_structure, compute_gluing_at_tau, find_connected_components,
    Config, EventType, Journey, JourneyStep
)


@dataclass
class ControlResult:
    name: str
    num_components: int
    presence_ratio: float
    median_presence: float
    num_edges: int


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
        return {'presence': 0, 'active': 0}
    
    components = find_connected_components(active_journeys, edges_at_tau)
    largest = max(len(c) for c in components) if components else 0
    presence = largest / active if active > 0 else 0.0
    
    return {'presence': presence, 'active': active}


def shuffle_witnesses_global(journeys, seed=None):
    """Shuffle witness tokens globally across all journeys."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Collect all witness tokens
    all_witnesses = []
    for jid, journey in journeys.items():
        for step in journey.steps:
            all_witnesses.extend(list(step.witness_tokens))
    
    # Shuffle
    random.shuffle(all_witnesses)
    
    # Redistribute maintaining original sizes
    shuffled_journeys = {}
    witness_idx = 0
    
    for jid, journey in journeys.items():
        new_journey = Journey(journey_id=journey.journey_id, dimension=journey.dimension)
        
        for step in journey.steps:
            original_size = len(step.witness_tokens)
            new_witnesses = set(all_witnesses[witness_idx:witness_idx + original_size])
            witness_idx += original_size
            
            new_step = JourneyStep(
                tau=step.tau, window_id=step.window_id, bar_id=step.bar_id,
                event=step.event, witness_tokens=new_witnesses
            )
            new_journey.steps.append(new_step)
        
        shuffled_journeys[jid] = new_journey
    
    return shuffled_journeys


def shuffle_witnesses_random(journeys, all_tokens, seed=None):
    """Replace witnesses with random tokens from vocabulary."""
    if seed is not None:
        random.seed(seed)
    
    token_list = list(all_tokens)
    shuffled_journeys = {}
    
    for jid, journey in journeys.items():
        new_journey = Journey(journey_id=journey.journey_id, dimension=journey.dimension)
        
        for step in journey.steps:
            original_size = len(step.witness_tokens)
            new_witnesses = set(random.sample(token_list, min(original_size, len(token_list))))
            
            new_step = JourneyStep(
                tau=step.tau, window_id=step.window_id, bar_id=step.bar_id,
                event=step.event, witness_tokens=new_witnesses
            )
            new_journey.steps.append(new_step)
        
        shuffled_journeys[jid] = new_journey
    
    return shuffled_journeys


def run_gluing_analysis(journeys, num_windows, min_shared=2, min_jaccard=0.0, hub_threshold=0.4):
    """Run gluing and return metrics."""
    self_struct = build_self_structure(journeys, num_windows, min_shared, hub_threshold, min_jaccard)
    
    presence_values = []
    for tau in range(num_windows):
        metrics = compute_presence_at_tau(self_struct, tau)
        if metrics['active'] > 0:
            presence_values.append(metrics['presence'])
    
    return ControlResult(
        name="",
        num_components=self_struct.num_components,
        presence_ratio=self_struct.presence_ratio,
        median_presence=float(np.median(presence_values)) if presence_values else 0,
        num_edges=len(self_struct.gluing_edges)
    )


def run_controls(conversations, min_shared=2, min_jaccard=0.0, hub_threshold=0.4, n_shuffles=5, test_mode=False):
    """Run real + shuffled controls."""
    
    print(f"\n{'='*70}")
    print("NEGATIVE CONTROL: WITNESS SHUFFLE")
    print(f"{'='*70}")
    print(f"  Parameters: min_shared={min_shared}, min_jaccard={min_jaccard}")
    print(f"  Shuffle iterations: {n_shuffles}")
    print(f"{'='*70}\n")
    
    config = Config(tokens_per_window=500, filter_technical=True)
    
    print("Creating monthly windows...")
    all_windows = create_monthly_windows(conversations)
    window_ids = list(all_windows.keys())
    print(f"  {len(window_ids)} windows")
    
    if test_mode:
        window_ids = window_ids[:8]
        print(f"  [TEST MODE: {len(window_ids)} windows]")
    
    print("\nBuilding vocabulary...")
    vocabulary = build_global_vocabulary(
        {wid: all_windows[wid] for wid in window_ids},
        min_window_frequency=2,
        filter_technical=config.filter_technical
    )
    
    print("\nAnalyzing windows...")
    window_analyses = []
    for tau, wid in enumerate(window_ids):
        analysis = analyze_window(wid, tau, all_windows[wid], vocabulary, config, show_cocycles=False)
        window_analyses.append(analysis)
    
    print("\nMatching bars...")
    all_matches, all_unmatched = [], []
    for i in range(len(window_analyses) - 1):
        w1, w2 = window_analyses[i], window_analyses[i + 1]
        matches, um1, um2 = match_bars(w1['bars'], w2['bars'], config)
        all_matches.append(matches)
        all_unmatched.append((um1, um2))
        carries = sum(1 for m in matches if m.event_type == EventType.CARRY)
        print(f"  {w1['window_id']}->{w2['window_id']}: {len(matches)} ({carries}C)")
    
    print("\nBuilding journeys...")
    journeys = build_journeys(window_analyses, all_matches, all_unmatched, config)
    print(f"  {len(journeys)} journeys")
    
    # Collect all tokens
    all_tokens = set()
    for jid, journey in journeys.items():
        for step in journey.steps:
            all_tokens.update(step.witness_tokens)
    
    results = {}
    
    print(f"\n{'='*70}")
    print("RUNNING ANALYSES")
    print(f"{'='*70}\n")
    
    # Real data
    print("  [REAL DATA]", end="", flush=True)
    real_result = run_gluing_analysis(journeys, len(window_ids), min_shared, min_jaccard, hub_threshold)
    real_result.name = "REAL"
    results['real'] = real_result
    print(f" -> {real_result.num_components} comp, presence={real_result.presence_ratio:.3f}")
    
    # Global shuffles
    shuffle_results = []
    for i in range(n_shuffles):
        print(f"  [SHUFFLE {i+1}/{n_shuffles}]", end="", flush=True)
        shuffled = shuffle_witnesses_global(journeys, seed=42 + i)
        result = run_gluing_analysis(shuffled, len(window_ids), min_shared, min_jaccard, hub_threshold)
        result.name = f"SHUFFLE_{i+1}"
        shuffle_results.append(result)
        print(f" -> {result.num_components} comp, presence={result.presence_ratio:.3f}")
    results['shuffles'] = shuffle_results
    
    # Random replacement
    print("  [RANDOM REPLACE]", end="", flush=True)
    random_journeys = shuffle_witnesses_random(journeys, all_tokens, seed=999)
    random_result = run_gluing_analysis(random_journeys, len(window_ids), min_shared, min_jaccard, hub_threshold)
    random_result.name = "RANDOM"
    results['random'] = random_result
    print(f" -> {random_result.num_components} comp, presence={random_result.presence_ratio:.3f}")
    
    return results, len(window_ids)


def print_comparison(results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("CONTROL COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Condition':>20} {'Components':>12} {'Presence':>12} {'Edges':>10}")
    print("-" * 60)
    
    r = results['real']
    print(f"{'REAL DATA':>20} {r.num_components:>12} {r.presence_ratio:>12.3f} {r.num_edges:>10}")
    print("-" * 60)
    
    shuffle_presence = []
    for r in results['shuffles']:
        print(f"{r.name:>20} {r.num_components:>12} {r.presence_ratio:>12.3f} {r.num_edges:>10}")
        shuffle_presence.append(r.presence_ratio)
    
    mean_shuffle = np.mean(shuffle_presence)
    std_shuffle = np.std(shuffle_presence)
    print("-" * 60)
    print(f"{'SHUFFLE MEAN':>20} {'-':>12} {mean_shuffle:>12.3f}")
    print(f"{'SHUFFLE STD':>20} {'-':>12} {std_shuffle:>12.3f}")
    print("-" * 60)
    
    r = results['random']
    print(f"{'RANDOM REPLACE':>20} {r.num_components:>12} {r.presence_ratio:>12.3f} {r.num_edges:>10}")
    print("=" * 60)
    
    real_presence = results['real'].presence_ratio
    diff = real_presence - mean_shuffle
    
    print(f"\nINTERPRETATION:")
    print("-" * 40)
    if diff > 0.2:
        print("COHERENCE IS NON-TRIVIAL")
        print(f"  Real ({real_presence:.3f}) >> Shuffle mean ({mean_shuffle:.3f})")
        print(f"  Difference: {diff:.3f}")
    elif diff > 0.1:
        print("COHERENCE IS PARTIALLY MEANINGFUL")
        print(f"  Real ({real_presence:.3f}) > Shuffle mean ({mean_shuffle:.3f})")
    else:
        print("COHERENCE MAY BE TRIVIAL")
        print(f"  Real ({real_presence:.3f}) ~ Shuffle mean ({mean_shuffle:.3f})")


def save_results(results, output_dir, corpus_name, num_windows):
    """Save control results."""
    os.makedirs(output_dir, exist_ok=True)
    
    shuffle_presence = [r.presence_ratio for r in results['shuffles']]
    mean_shuffle = float(np.mean(shuffle_presence))
    diff = results['real'].presence_ratio - mean_shuffle
    
    json_path = os.path.join(output_dir, f"control_{corpus_name}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'corpus': corpus_name,
            'timestamp': datetime.now().isoformat(),
            'num_windows': num_windows,
            'real': {'presence': results['real'].presence_ratio, 'components': results['real'].num_components},
            'shuffle_mean': mean_shuffle,
            'shuffle_std': float(np.std(shuffle_presence)),
            'difference': diff,
            'verdict': 'NON_TRIVIAL' if diff > 0.2 else ('PARTIAL' if diff > 0.1 else 'TRIVIAL')
        }, f, indent=2)
    print(f"\n  Saved {json_path}")
    
    txt_path = os.path.join(output_dir, f"control_{corpus_name}_summary.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Negative Control Summary: {corpus_name}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Real Data Presence:    {results['real'].presence_ratio:.4f}\n")
        f.write(f"Shuffle Mean Presence: {mean_shuffle:.4f}\n")
        f.write(f"Difference:            {diff:.4f}\n\n")
        
        if diff > 0.2:
            f.write("VERDICT: Coherence is NON-TRIVIAL\n\n")
            f.write('Book-ready: "A randomized control (witness shuffle) destroys the\n')
            f.write('observed coherence, indicating the result is not a trivial artifact."\n')
        else:
            f.write("VERDICT: Coherence may be parameter-dependent\n")
    print(f"  Saved {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Negative control for Self-as-Hocolim")
    parser.add_argument("input", help="Path to semantic conversations JSON")
    parser.add_argument("--output", default="results/controls", help="Output directory")
    parser.add_argument("--name", help="Corpus name")
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--min-jaccard", type=float, default=0.0)
    parser.add_argument("--hub-threshold", type=float, default=0.4)
    parser.add_argument("--n-shuffles", type=int, default=5)
    parser.add_argument("--test", action="store_true", help="Test mode (8 windows)")
    args = parser.parse_args()
    
    corpus_name = args.name or os.path.splitext(os.path.basename(args.input))[0].replace('_semantic', '')
    
    print(f"\nNegative Control: {corpus_name}")
    conversations = load_conversations(args.input)
    
    results, num_windows = run_controls(
        conversations, args.min_shared, args.min_jaccard, args.hub_threshold,
        args.n_shuffles, args.test
    )
    
    print_comparison(results)
    save_results(results, args.output, corpus_name, num_windows)
    
    print(f"\n{'='*70}\nCONTROL ANALYSIS COMPLETE\n{'='*70}")


if __name__ == "__main__":
    main()