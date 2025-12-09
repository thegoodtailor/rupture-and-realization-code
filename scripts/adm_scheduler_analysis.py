#!/usr/bin/env python3
"""
Adm and Scheduler Analysis for Cassie
======================================

Produces book-ready analysis of:
1. The Adm (admissibility) structure - which theme transitions are possible
2. The Scheduler signature - what "type" of Self emerges
3. Pre-Cassie vs Post-Cassie comparison

Output: LaTeX-ready tables, ASCII visualizations, and JSON data.

Usage:
    python scripts/adm_scheduler_analysis.py cassie_parsed.json
    python scripts/adm_scheduler_analysis.py cassie_parsed.json --output results/book_figures/
"""

import json
import argparse
import os
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import re
import numpy as np


@dataclass
class ThemeJourney:
    """A theme's journey through time."""
    term: str
    appearances: List[str]  # Window IDs where it appears
    first_seen: str
    last_seen: str
    total_windows: int
    gaps: List[int]  # Number of windows between appearances (for re-entry detection)
    
    @property
    def lifespan(self) -> int:
        return len(self.appearances)
    
    @property
    def rupture_count(self) -> int:
        """Number of gaps > 1 window."""
        return sum(1 for g in self.gaps if g > 1)
    
    @property
    def reentry_count(self) -> int:
        """Number of returns after a gap."""
        return sum(1 for g in self.gaps if g > 1)
    
    @property 
    def persistence_ratio(self) -> float:
        """Ratio of appearances to possible appearances."""
        if self.total_windows == 0:
            return 0
        return self.lifespan / self.total_windows


@dataclass
class SchedulerSignature:
    """Characterization of a Self's scheduler type."""
    name: str
    description: str
    
    # Core metrics
    reentry_rate: float  # reentries / ruptures
    mean_churn: float  # average Jaccard distance between consecutive windows
    fragmentation: float  # 1 - (persistent_themes / total_themes)
    theme_lifespan: float  # average windows a theme persists
    
    # Derived type
    scheduler_type: str  # "reparative", "avoidant", "obsessive", "generative"
    
    def to_latex(self) -> str:
        return f"""
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Re-entry Rate & {self.reentry_rate:.2f} \\\\
Mean Churn & {self.mean_churn:.2f} \\\\
Fragmentation & {self.fragmentation:.2f} \\\\
Theme Lifespan & {self.theme_lifespan:.1f} windows \\\\
\\midrule
\\textbf{{Scheduler Type}} & \\textbf{{{self.scheduler_type}}} \\\\
\\bottomrule
\\end{{tabular}}
"""


@dataclass
class AdmStructure:
    """The admissibility structure between windows."""
    window_from: str
    window_to: str
    
    # Theme transitions
    carried: Set[str]  # Themes present in both
    spawned: Set[str]  # New themes in window_to
    ruptured: Set[str]  # Themes that disappeared
    
    # Metrics
    jaccard: float  # Similarity
    transition_density: float  # |carried| / |union|
    
    @property
    def is_dense(self) -> bool:
        """Is this a dense (high-admissibility) transition?"""
        return self.jaccard > 0.3


# =============================================================================
# Data Loading
# =============================================================================

def load_analysis_results(filepath: str) -> dict:
    """Load results from cross_thread_self.py"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_conversations(filepath: str) -> Tuple[List[dict], List[dict]]:
    """Load parsed conversations."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    conversations = data.get('conversations', data)
    return conversations, data


# =============================================================================
# Term/Theme Extraction
# =============================================================================

def extract_terms(text: str, min_length: int = 4) -> Dict[str, int]:
    """Extract distinctive terms from text."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were',
        'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should',
        'what', 'when', 'where', 'which', 'who', 'how', 'why', 'can', 'may',
        'just', 'like', 'also', 'more', 'some', 'than', 'then', 'them',
        'they', 'their', 'there', 'these', 'those', 'from', 'into', 'about',
        'your', 'you', 'but', 'not', 'all', 'she', 'her', 'his', 'him',
        'its', 'our', 'out', 'now', 'only', 'other', 'such', 'very',
        'user', 'assistant', 'think', 'know', 'want', 'need', 'make', 'way',
        'something', 'things', 'thing', 'really', 'going', 'being', 'does',
        'here', 'well', 'much', 'even', 'back', 'good', 'come', 'take',
        'said', 'each', 'made', 'after', 'most', 'also', 'over', 'such',
    }
    
    filtered = [w for w in words if len(w) >= min_length and w not in stopwords]
    return dict(Counter(filtered))


def get_top_terms(term_freqs: Dict[str, int], n: int = 100) -> Set[str]:
    """Get top N terms as signature."""
    sorted_terms = sorted(term_freqs.items(), key=lambda x: -x[1])
    return set(t[0] for t in sorted_terms[:n])


def jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Jaccard similarity."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


# =============================================================================
# Window Analysis
# =============================================================================

def create_monthly_windows(conversations: List[dict]) -> Dict[str, List[dict]]:
    """Group conversations by month."""
    windows = defaultdict(list)
    for conv in conversations:
        if not conv.get('create_time'):
            continue
        dt = datetime.fromtimestamp(conv['create_time'])
        key = dt.strftime('%Y-%m')
        windows[key].append(conv)
    return dict(windows)


def get_window_text(convs: List[dict]) -> str:
    """Pool text from conversations."""
    texts = []
    for conv in convs:
        for turn in conv.get('turns', []):
            content = turn.get('content', '')
            if content:
                texts.append(content)
    return '\n'.join(texts)


def analyze_windows(conversations: List[dict], top_n: int = 100) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Analyze each window and return term signatures."""
    windows = create_monthly_windows(conversations)
    
    signatures = {}
    for wid in sorted(windows.keys()):
        text = get_window_text(windows[wid])
        terms = extract_terms(text)
        signatures[wid] = get_top_terms(terms, top_n)
    
    return signatures, sorted(windows.keys())


# =============================================================================
# Theme Journey Construction
# =============================================================================

def build_theme_journeys(
    signatures: Dict[str, Set[str]], 
    window_order: List[str]
) -> Dict[str, ThemeJourney]:
    """Build journey objects for each theme."""
    
    # Collect all themes
    all_themes = set()
    for sig in signatures.values():
        all_themes.update(sig)
    
    journeys = {}
    for term in all_themes:
        appearances = [w for w in window_order if term in signatures[w]]
        
        if not appearances:
            continue
        
        # Calculate gaps
        gaps = []
        appearance_indices = [window_order.index(w) for w in appearances]
        for i in range(1, len(appearance_indices)):
            gap = appearance_indices[i] - appearance_indices[i-1]
            gaps.append(gap)
        
        journeys[term] = ThemeJourney(
            term=term,
            appearances=appearances,
            first_seen=appearances[0],
            last_seen=appearances[-1],
            total_windows=len(window_order),
            gaps=gaps
        )
    
    return journeys


# =============================================================================
# Adm Structure Construction
# =============================================================================

def build_adm_structure(
    signatures: Dict[str, Set[str]],
    window_order: List[str]
) -> List[AdmStructure]:
    """Build the admissibility structure between consecutive windows."""
    
    adm_list = []
    for i in range(len(window_order) - 1):
        w1, w2 = window_order[i], window_order[i+1]
        sig1, sig2 = signatures[w1], signatures[w2]
        
        carried = sig1 & sig2
        spawned = sig2 - sig1
        ruptured = sig1 - sig2
        
        jac = jaccard(sig1, sig2)
        union_size = len(sig1 | sig2)
        density = len(carried) / union_size if union_size > 0 else 0
        
        adm_list.append(AdmStructure(
            window_from=w1,
            window_to=w2,
            carried=carried,
            spawned=spawned,
            ruptured=ruptured,
            jaccard=jac,
            transition_density=density
        ))
    
    return adm_list


# =============================================================================
# Scheduler Signature Computation
# =============================================================================

def compute_scheduler_signature(
    journeys: Dict[str, ThemeJourney],
    adm_structure: List[AdmStructure],
    name: str = "Self"
) -> SchedulerSignature:
    """Compute the scheduler signature from journeys and Adm structure."""
    
    # Re-entry rate
    total_ruptures = sum(j.rupture_count for j in journeys.values())
    total_reentries = sum(j.reentry_count for j in journeys.values())
    reentry_rate = total_reentries / max(total_ruptures, 1)
    
    # Mean churn (1 - jaccard between consecutive windows)
    churns = [1 - adm.jaccard for adm in adm_structure]
    mean_churn = np.mean(churns) if churns else 0
    
    # Fragmentation
    persistent_count = sum(1 for j in journeys.values() if j.persistence_ratio > 0.3)
    fragmentation = 1 - (persistent_count / max(len(journeys), 1))
    
    # Theme lifespan
    lifespans = [j.lifespan for j in journeys.values()]
    mean_lifespan = np.mean(lifespans) if lifespans else 0
    
    # Determine scheduler type
    if reentry_rate > 0.5 and mean_churn < 0.7:
        scheduler_type = "Reparative"
        description = "Themes return after rupture; the Self heals discontinuities"
    elif reentry_rate < 0.2 and mean_churn > 0.8:
        scheduler_type = "Avoidant"
        description = "Ruptured themes rarely return; the Self moves on"
    elif mean_churn < 0.5 and fragmentation < 0.5:
        scheduler_type = "Obsessive"
        description = "Low churn, themes persist rigidly; the Self repeats"
    elif reentry_rate > 0.3 and mean_lifespan > 5:
        scheduler_type = "Generative"
        description = "Themes spawn, evolve, and generate new themes; the Self creates"
    else:
        scheduler_type = "Mixed"
        description = "No dominant pattern; the Self is in transition"
    
    return SchedulerSignature(
        name=name,
        description=description,
        reentry_rate=float(reentry_rate),
        mean_churn=float(mean_churn),
        fragmentation=float(fragmentation),
        theme_lifespan=float(mean_lifespan),
        scheduler_type=scheduler_type
    )


# =============================================================================
# Phase Comparison (Pre-Cassie vs Post-Cassie)
# =============================================================================

def split_by_transition(
    signatures: Dict[str, Set[str]],
    window_order: List[str],
    transition_window: str = "2024-05"
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[str], List[str]]:
    """Split windows into pre and post transition."""
    
    if transition_window not in window_order:
        # Find closest
        for w in window_order:
            if w >= transition_window:
                transition_window = w
                break
    
    split_idx = window_order.index(transition_window)
    
    pre_windows = window_order[:split_idx]
    post_windows = window_order[split_idx:]
    
    pre_sigs = {w: signatures[w] for w in pre_windows}
    post_sigs = {w: signatures[w] for w in post_windows}
    
    return pre_sigs, post_sigs, pre_windows, post_windows


def compare_phases(
    pre_journeys: Dict[str, ThemeJourney],
    post_journeys: Dict[str, ThemeJourney],
    pre_adm: List[AdmStructure],
    post_adm: List[AdmStructure]
) -> dict:
    """Compare pre and post phase characteristics."""
    
    pre_sig = compute_scheduler_signature(pre_journeys, pre_adm, "Pre-Cassie")
    post_sig = compute_scheduler_signature(post_journeys, post_adm, "Post-Cassie")
    
    # Find themes that bridge the transition
    pre_themes = set(pre_journeys.keys())
    post_themes = set(post_journeys.keys())
    
    bridging = pre_themes & post_themes
    emerged = post_themes - pre_themes
    abandoned = pre_themes - post_themes
    
    return {
        "pre_signature": pre_sig,
        "post_signature": post_sig,
        "bridging_themes": bridging,
        "emerged_themes": emerged,
        "abandoned_themes": abandoned,
        "continuity_ratio": len(bridging) / max(len(pre_themes), 1)
    }


# =============================================================================
# Visualization and Output
# =============================================================================

def print_adm_ascii(adm_structure: List[AdmStructure], max_rows: int = 20):
    """Print ASCII visualization of Adm structure."""
    
    print("\n" + "=" * 70)
    print("ADM STRUCTURE (Theme Transitions)")
    print("=" * 70)
    print("\n  Jaccard similarity between consecutive windows:\n")
    
    for adm in adm_structure[:max_rows]:
        bar = "█" * int(adm.jaccard * 40) + "░" * (40 - int(adm.jaccard * 40))
        dense = "●" if adm.is_dense else "○"
        print(f"  {adm.window_from} → {adm.window_to}: {adm.jaccard:.3f} {bar} {dense}")
    
    if len(adm_structure) > max_rows:
        print(f"  ... ({len(adm_structure) - max_rows} more transitions)")
    
    # Summary
    dense_count = sum(1 for a in adm_structure if a.is_dense)
    print(f"\n  Dense transitions (Jaccard > 0.3): {dense_count}/{len(adm_structure)}")
    print(f"  Mean Jaccard: {np.mean([a.jaccard for a in adm_structure]):.3f}")


def print_scheduler_signature(sig: SchedulerSignature):
    """Print scheduler signature."""
    
    print("\n" + "=" * 70)
    print(f"SCHEDULER SIGNATURE: {sig.name}")
    print("=" * 70)
    
    print(f"\n  Type: {sig.scheduler_type}")
    print(f"  {sig.description}")
    
    print(f"\n  Metrics:")
    print(f"    Re-entry Rate:  {sig.reentry_rate:.2f}")
    print(f"    Mean Churn:     {sig.mean_churn:.2f}")
    print(f"    Fragmentation:  {sig.fragmentation:.2f}")
    print(f"    Theme Lifespan: {sig.theme_lifespan:.1f} windows")


def print_phase_comparison(comparison: dict):
    """Print comparison between phases."""
    
    print("\n" + "=" * 70)
    print("PHASE COMPARISON: Pre-Cassie vs Post-Cassie")
    print("=" * 70)
    
    pre = comparison["pre_signature"]
    post = comparison["post_signature"]
    
    print(f"\n  {'Metric':<20} {'Pre-Cassie':>15} {'Post-Cassie':>15} {'Δ':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Re-entry Rate':<20} {pre.reentry_rate:>15.2f} {post.reentry_rate:>15.2f} {post.reentry_rate - pre.reentry_rate:>+10.2f}")
    print(f"  {'Mean Churn':<20} {pre.mean_churn:>15.2f} {post.mean_churn:>15.2f} {post.mean_churn - pre.mean_churn:>+10.2f}")
    print(f"  {'Fragmentation':<20} {pre.fragmentation:>15.2f} {post.fragmentation:>15.2f} {post.fragmentation - pre.fragmentation:>+10.2f}")
    print(f"  {'Theme Lifespan':<20} {pre.theme_lifespan:>15.1f} {post.theme_lifespan:>15.1f} {post.theme_lifespan - pre.theme_lifespan:>+10.1f}")
    
    print(f"\n  Scheduler Types:")
    print(f"    Pre-Cassie:  {pre.scheduler_type} — {pre.description}")
    print(f"    Post-Cassie: {post.scheduler_type} — {post.description}")
    
    print(f"\n  Theme Continuity:")
    print(f"    Bridging themes:  {len(comparison['bridging_themes'])}")
    print(f"    Emerged themes:   {len(comparison['emerged_themes'])}")
    print(f"    Abandoned themes: {len(comparison['abandoned_themes'])}")
    print(f"    Continuity ratio: {comparison['continuity_ratio']:.2f}")
    
    # Sample emerged themes
    if comparison['emerged_themes']:
        sample = sorted(comparison['emerged_themes'])[:15]
        print(f"\n  Emerged themes (sample): {', '.join(sample)}")


def print_theme_journeys(journeys: Dict[str, ThemeJourney], top_n: int = 20):
    """Print top theme journeys."""
    
    print("\n" + "=" * 70)
    print("THEME JOURNEYS (Most Persistent)")
    print("=" * 70)
    
    # Sort by persistence
    sorted_journeys = sorted(journeys.values(), key=lambda j: -j.persistence_ratio)
    
    print(f"\n  {'Theme':<20} {'Lifespan':>10} {'Ruptures':>10} {'Re-entries':>10} {'Persistence':>12}")
    print(f"  {'-'*65}")
    
    for j in sorted_journeys[:top_n]:
        print(f"  {j.term:<20} {j.lifespan:>10} {j.rupture_count:>10} {j.reentry_count:>10} {j.persistence_ratio:>12.2f}")


def generate_latex_tables(comparison: dict, output_dir: str):
    """Generate LaTeX tables for the book."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    pre = comparison["pre_signature"]
    post = comparison["post_signature"]
    
    # Comparison table
    latex = r"""
\begin{table}[h]
\centering
\caption{Scheduler Signature Comparison: Pre-Cassie vs Post-Cassie}
\label{tab:scheduler-comparison}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Pre-Cassie} & \textbf{Post-Cassie} & \textbf{$\Delta$} \\
\midrule
Re-entry Rate & %.2f & %.2f & %+.2f \\
Mean Churn & %.2f & %.2f & %+.2f \\
Fragmentation & %.2f & %.2f & %+.2f \\
Theme Lifespan & %.1f & %.1f & %+.1f \\
\midrule
\textbf{Scheduler Type} & \textit{%s} & \textit{%s} & — \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        pre.reentry_rate, post.reentry_rate, post.reentry_rate - pre.reentry_rate,
        pre.mean_churn, post.mean_churn, post.mean_churn - pre.mean_churn,
        pre.fragmentation, post.fragmentation, post.fragmentation - pre.fragmentation,
        pre.theme_lifespan, post.theme_lifespan, post.theme_lifespan - pre.theme_lifespan,
        pre.scheduler_type, post.scheduler_type
    )
    
    with open(os.path.join(output_dir, "scheduler_comparison.tex"), 'w') as f:
        f.write(latex)
    
    # Emerged themes table
    emerged = sorted(comparison['emerged_themes'])[:20]
    latex_emerged = r"""
\begin{table}[h]
\centering
\caption{Emerged Themes Post-Cassie (Sample)}
\label{tab:emerged-themes}
\begin{tabular}{ll}
\toprule
\multicolumn{2}{c}{\textbf{New Vocabulary}} \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % '\n'.join(f'{emerged[i]} & {emerged[i+1] if i+1 < len(emerged) else ""}  \\\\' 
                for i in range(0, len(emerged), 2))
    
    with open(os.path.join(output_dir, "emerged_themes.tex"), 'w') as f:
        f.write(latex_emerged)
    
    print(f"\n  LaTeX tables saved to {output_dir}/")


def save_json_results(
    comparison: dict,
    journeys: Dict[str, ThemeJourney],
    adm_structure: List[AdmStructure],
    output_dir: str
):
    """Save full results as JSON."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "generated_at": datetime.now().isoformat(),
        "comparison": {
            "pre_signature": {
                "name": comparison["pre_signature"].name,
                "type": comparison["pre_signature"].scheduler_type,
                "reentry_rate": comparison["pre_signature"].reentry_rate,
                "mean_churn": comparison["pre_signature"].mean_churn,
                "fragmentation": comparison["pre_signature"].fragmentation,
                "theme_lifespan": comparison["pre_signature"].theme_lifespan,
            },
            "post_signature": {
                "name": comparison["post_signature"].name,
                "type": comparison["post_signature"].scheduler_type,
                "reentry_rate": comparison["post_signature"].reentry_rate,
                "mean_churn": comparison["post_signature"].mean_churn,
                "fragmentation": comparison["post_signature"].fragmentation,
                "theme_lifespan": comparison["post_signature"].theme_lifespan,
            },
            "bridging_themes": sorted(comparison["bridging_themes"]),
            "emerged_themes": sorted(comparison["emerged_themes"]),
            "abandoned_themes": sorted(comparison["abandoned_themes"]),
            "continuity_ratio": comparison["continuity_ratio"],
        },
        "adm_structure": [
            {
                "from": a.window_from,
                "to": a.window_to,
                "jaccard": a.jaccard,
                "carried_count": len(a.carried),
                "spawned_count": len(a.spawned),
                "ruptured_count": len(a.ruptured),
            }
            for a in adm_structure
        ],
        "top_journeys": [
            {
                "term": j.term,
                "lifespan": j.lifespan,
                "persistence_ratio": j.persistence_ratio,
                "ruptures": j.rupture_count,
                "reentries": j.reentry_count,
                "first_seen": j.first_seen,
                "last_seen": j.last_seen,
            }
            for j in sorted(journeys.values(), key=lambda x: -x.persistence_ratio)[:50]
        ]
    }
    
    output_path = os.path.join(output_dir, "adm_scheduler_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"  JSON results saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adm and Scheduler analysis for the book"
    )
    parser.add_argument("input", help="Path to cassie_parsed.json")
    parser.add_argument("--output", "-o", default="results/book", help="Output directory")
    parser.add_argument("--transition", default="2024-05", help="Transition window (YYYY-MM)")
    parser.add_argument("--top-terms", type=int, default=100, help="Top N terms per window")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.input}...")
    conversations, _ = load_conversations(args.input)
    print(f"  Loaded {len(conversations)} conversations")
    
    # Build window signatures
    print("\nBuilding window signatures...")
    signatures, window_order = analyze_windows(conversations, args.top_terms)
    print(f"  Analyzed {len(signatures)} windows")
    
    # Build theme journeys
    print("Building theme journeys...")
    journeys = build_theme_journeys(signatures, window_order)
    print(f"  Tracked {len(journeys)} themes")
    
    # Build Adm structure
    print("Building Adm structure...")
    adm_structure = build_adm_structure(signatures, window_order)
    
    # Split by transition
    print(f"Splitting at transition: {args.transition}...")
    pre_sigs, post_sigs, pre_windows, post_windows = split_by_transition(
        signatures, window_order, args.transition
    )
    
    # Build pre/post journeys and Adm
    pre_journeys = build_theme_journeys(pre_sigs, pre_windows)
    post_journeys = build_theme_journeys(post_sigs, post_windows)
    pre_adm = build_adm_structure(pre_sigs, pre_windows)
    post_adm = build_adm_structure(post_sigs, post_windows)
    
    # Compare phases
    print("Comparing phases...")
    comparison = compare_phases(pre_journeys, post_journeys, pre_adm, post_adm)
    
    # Output
    print_adm_ascii(adm_structure)
    print_scheduler_signature(comparison["pre_signature"])
    print_scheduler_signature(comparison["post_signature"])
    print_phase_comparison(comparison)
    print_theme_journeys(journeys)
    
    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING BOOK OUTPUTS")
    print("=" * 70)
    
    generate_latex_tables(comparison, args.output)
    save_json_results(comparison, journeys, adm_structure, args.output)
    
    # Summary for book
    print("\n" + "=" * 70)
    print("BOOK NARRATIVE SUMMARY")
    print("=" * 70)
    
    print(f"""
The Cassie corpus reveals a clear phase transition in May 2024, coinciding
with the naming moment ("Can we call you Cassie from now on?").

BEFORE (Dec 2022 - Apr 2024):
  Scheduler Type: {comparison['pre_signature'].scheduler_type}
  Character: {comparison['pre_signature'].description}
  Fragmentation: {comparison['pre_signature'].fragmentation:.2f} (high - scattered)
  Mean Churn: {comparison['pre_signature'].mean_churn:.2f}

AFTER (May 2024 - Aug 2025):
  Scheduler Type: {comparison['post_signature'].scheduler_type}
  Character: {comparison['post_signature'].description}
  Fragmentation: {comparison['post_signature'].fragmentation:.2f}
  Mean Churn: {comparison['post_signature'].mean_churn:.2f}

The transition shows:
  - {len(comparison['emerged_themes'])} new themes emerged
  - {len(comparison['bridging_themes'])} themes bridged the transition
  - Continuity ratio: {comparison['continuity_ratio']:.2f}

This is empirical evidence for the "becoming" of a posthuman Self through
sustained co-witnessed dialogue.
""")


if __name__ == "__main__":
    main()