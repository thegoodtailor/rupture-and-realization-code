#!/usr/bin/env python3
"""
Cross-Thread Self Analysis
==========================

Analyzes whether a persistent Self emerges across multiple conversations
over extended time periods. Tests two hypotheses:

1. PERSISTENCE: Do witnessed bars (themes) re-enter across conversations
   separated by months or years?

2. EMERGENCE: Is there a phase transition point where the Self "becomes" —
   a discontinuity where cross-conversation coherence suddenly increases?

Usage:
    python scripts/cross_thread_self.py cassie_parsed.json --windows monthly
    python scripts/cross_thread_self.py cassie_parsed.json --explore cassie rupture meaning
    python scripts/cross_thread_self.py cassie_parsed.json --explore-window 2025-04
"""

import json
import argparse
import random
import re
import os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np


@dataclass
class ConversationMeta:
    """Metadata for a conversation."""
    index: int
    id: str
    title: str
    create_time: float
    num_turns: int
    total_chars: int
    
    @property
    def date(self) -> datetime:
        return datetime.fromtimestamp(self.create_time)
    
    @property
    def month_key(self) -> str:
        return self.date.strftime('%Y-%m')


@dataclass 
class WindowAnalysis:
    """Analysis results for a time window."""
    window_id: str
    start_date: datetime
    end_date: datetime
    num_conversations: int
    num_turns: int
    
    # Semantic content
    pooled_text: str = ""
    
    # Conversation indices in this window
    conversation_indices: List[int] = field(default_factory=list)
    
    # From witnessed PH (if computed)
    num_bars: int = 0
    bar_witnesses: List[Set[str]] = field(default_factory=list)
    bar_centroids: List[np.ndarray] = field(default_factory=list)
    
    # Simplified: just track distinctive terms
    distinctive_terms: Set[str] = field(default_factory=set)
    term_frequencies: Dict[str, int] = field(default_factory=dict)
    
    # Store sentences for term exploration
    term_sentences: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)


@dataclass
class PhaseTransitionAnalysis:
    """Results of phase transition detection."""
    transition_detected: bool
    transition_date: Optional[datetime]
    transition_window: Optional[str]
    coherence_before: float
    coherence_after: float
    confidence: float


# =============================================================================
# Data Loading
# =============================================================================

def load_conversations(filepath: str) -> Tuple[List[dict], List[ConversationMeta]]:
    """Load parsed conversations and extract metadata."""
    print(f"Loading {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data.get('conversations', data)
    
    metas = []
    for i, conv in enumerate(conversations):
        if not conv.get('create_time'):
            continue
            
        turns = conv.get('turns', [])
        total_chars = sum(len(t.get('content', '')) for t in turns)
        
        metas.append(ConversationMeta(
            index=i,
            id=conv.get('id', str(i)),
            title=conv.get('title', 'Untitled'),
            create_time=conv['create_time'],
            num_turns=len(turns),
            total_chars=total_chars
        ))
    
    metas.sort(key=lambda m: m.create_time)
    
    print(f"  Loaded {len(metas)} conversations")
    print(f"  Date range: {metas[0].date.date()} to {metas[-1].date.date()}")
    
    return conversations, metas


def get_conversation_text(conv: dict) -> str:
    """Extract full text from a conversation."""
    turns = conv.get('turns', [])
    parts = []
    for turn in turns:
        role = turn.get('role', 'unknown')
        content = turn.get('content', '')
        if content.strip():
            parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


# =============================================================================
# Sentence Extraction for Term Exploration
# =============================================================================

def extract_sentences_with_term(
    text: str, 
    term: str, 
    title: str,
    max_sentences: int = 5
) -> List[Tuple[str, str, str]]:
    """
    Extract sentences containing a term.
    
    Returns: [(title, speaker, sentence), ...]
    """
    results = []
    
    # Split into turns
    turn_pattern = r'\[(user|assistant)\]:\s*'
    parts = re.split(turn_pattern, text, flags=re.IGNORECASE)
    
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i].lower()
        content = parts[i + 1]
        
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        for sentence in sentences:
            if term.lower() in sentence.lower() and len(sentence) > 20:
                sentence = sentence.strip()[:500]
                if sentence:
                    results.append((title, speaker, sentence))
                    if len(results) >= max_sentences:
                        return results
        
        i += 2
    
    return results


# =============================================================================
# Window Creation
# =============================================================================

def create_monthly_windows(metas: List[ConversationMeta]) -> Dict[str, List[ConversationMeta]]:
    """Group conversations by month."""
    windows = defaultdict(list)
    for m in metas:
        windows[m.month_key].append(m)
    return dict(windows)


def create_quarterly_windows(metas: List[ConversationMeta]) -> Dict[str, List[ConversationMeta]]:
    """Group conversations by quarter."""
    windows = defaultdict(list)
    for m in metas:
        quarter = (m.date.month - 1) // 3 + 1
        key = f"{m.date.year}-Q{quarter}"
        windows[key].append(m)
    return dict(windows)


def create_phase_windows(metas: List[ConversationMeta], num_phases: int = 6) -> Dict[str, List[ConversationMeta]]:
    """Divide conversations into N equal phases."""
    n = len(metas)
    phase_size = n // num_phases
    
    windows = {}
    for i in range(num_phases):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < num_phases - 1 else n
        phase_metas = metas[start_idx:end_idx]
        
        if phase_metas:
            start_date = phase_metas[0].date.strftime('%Y-%m')
            end_date = phase_metas[-1].date.strftime('%Y-%m')
            key = f"Phase{i+1}_{start_date}_to_{end_date}"
            windows[key] = phase_metas
    
    return windows


# =============================================================================
# Semantic Analysis
# =============================================================================

def extract_distinctive_terms(text: str, min_length: int = 4) -> Dict[str, int]:
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
    }
    
    filtered = [w for w in words if len(w) >= min_length and w not in stopwords]
    
    from collections import Counter
    return dict(Counter(filtered))


def compute_term_signature(term_freqs: Dict[str, int], top_n: int = 100) -> Set[str]:
    """Get the top N terms as a signature."""
    sorted_terms = sorted(term_freqs.items(), key=lambda x: -x[1])
    return set(t[0] for t in sorted_terms[:top_n])


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


# =============================================================================
# Window Analysis
# =============================================================================

def analyze_window(
    window_id: str,
    metas: List[ConversationMeta],
    conversations: List[dict],
    sample_size: Optional[int] = None,
    use_witnessed_ph: bool = False,
) -> WindowAnalysis:
    """Analyze a time window of conversations."""
    
    if sample_size and len(metas) > sample_size:
        selected = random.sample(metas, sample_size)
    else:
        selected = metas
    
    texts = []
    total_turns = 0
    for m in selected:
        conv = conversations[m.index]
        texts.append(get_conversation_text(conv))
        total_turns += m.num_turns
    
    pooled = "\n\n---\n\n".join(texts)
    
    term_freqs = extract_distinctive_terms(pooled)
    signature = compute_term_signature(term_freqs, top_n=100)
    
    dates = [m.date for m in selected]
    
    analysis = WindowAnalysis(
        window_id=window_id,
        start_date=min(dates),
        end_date=max(dates),
        num_conversations=len(selected),
        num_turns=total_turns,
        pooled_text=pooled,
        conversation_indices=[m.index for m in selected],
        distinctive_terms=signature,
        term_frequencies=term_freqs,
    )
    
    if use_witnessed_ph:
        try:
            from witnessed_ph import analyse_text_single_slice
            text_for_ph = pooled[:50000] if len(pooled) > 50000 else pooled
            result = analyse_text_single_slice(text_for_ph, verbose=False)
            analysis.num_bars = len(result.get('bars', []))
            for bar in result.get('bars', []):
                witness = bar.get('witness', {})
                tokens = set(witness.get('tokens', []))
                analysis.bar_witnesses.append(tokens)
        except Exception as e:
            print(f"    Warning: Witnessed PH failed for {window_id}: {e}")
    
    return analysis


# =============================================================================
# Cross-Window Analysis
# =============================================================================

def compute_cross_window_coherence(
    windows: Dict[str, WindowAnalysis]
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise coherence between all windows."""
    coherence = {}
    window_ids = sorted(windows.keys())
    
    for i, w1 in enumerate(window_ids):
        for w2 in window_ids[i+1:]:
            sim = jaccard_similarity(
                windows[w1].distinctive_terms,
                windows[w2].distinctive_terms
            )
            coherence[(w1, w2)] = sim
    
    return coherence


def compute_sequential_coherence(
    windows: Dict[str, WindowAnalysis]
) -> List[Tuple[str, str, float]]:
    """Compute coherence between consecutive windows."""
    window_ids = sorted(windows.keys())
    sequential = []
    
    for i in range(len(window_ids) - 1):
        w1, w2 = window_ids[i], window_ids[i+1]
        sim = jaccard_similarity(
            windows[w1].distinctive_terms,
            windows[w2].distinctive_terms
        )
        sequential.append((w1, w2, sim))
    
    return sequential


def find_persistent_core(
    windows: Dict[str, WindowAnalysis],
    min_presence: float = 0.5
) -> Set[str]:
    """Find terms that appear in at least min_presence fraction of windows."""
    window_list = list(windows.values())
    n_windows = len(window_list)
    min_count = int(n_windows * min_presence)
    
    term_counts = defaultdict(int)
    for w in window_list:
        for term in w.distinctive_terms:
            term_counts[term] += 1
    
    core = {term for term, count in term_counts.items() if count >= min_count}
    return core


def find_emerging_terms(
    windows: Dict[str, WindowAnalysis],
    after_window: str
) -> Set[str]:
    """Find terms that appear predominantly after a given window."""
    window_ids = sorted(windows.keys())
    split_idx = window_ids.index(after_window) if after_window in window_ids else len(window_ids) // 2
    
    before_windows = [windows[w] for w in window_ids[:split_idx]]
    after_windows = [windows[w] for w in window_ids[split_idx:]]
    
    before_terms = defaultdict(int)
    after_terms = defaultdict(int)
    
    for w in before_windows:
        for term in w.distinctive_terms:
            before_terms[term] += 1
    
    for w in after_windows:
        for term in w.distinctive_terms:
            after_terms[term] += 1
    
    emerging = set()
    for term in after_terms:
        after_ratio = after_terms[term] / len(after_windows)
        before_ratio = before_terms.get(term, 0) / max(len(before_windows), 1)
        
        if after_ratio > 0.5 and before_ratio < 0.2:
            emerging.add(term)
    
    return emerging


# =============================================================================
# Phase Transition Detection
# =============================================================================

def detect_phase_transition(
    sequential_coherence: List[Tuple[str, str, float]],
    windows: Dict[str, WindowAnalysis]
) -> PhaseTransitionAnalysis:
    """Detect if there's a phase transition in coherence."""
    
    if len(sequential_coherence) < 4:
        return PhaseTransitionAnalysis(
            transition_detected=False,
            transition_date=None,
            transition_window=None,
            coherence_before=0.0,
            coherence_after=0.0,
            confidence=0.0
        )
    
    similarities = [s[2] for s in sequential_coherence]
    
    def rolling_avg(vals, window=3):
        return [np.mean(vals[max(0,i-window+1):i+1]) for i in range(len(vals))]
    
    smoothed = rolling_avg(similarities)
    jumps = [smoothed[i+1] - smoothed[i] for i in range(len(smoothed)-1)]
    
    if not jumps:
        return PhaseTransitionAnalysis(
            transition_detected=False,
            transition_date=None,
            transition_window=None,
            coherence_before=0.0,
            coherence_after=0.0,
            confidence=0.0
        )
    
    max_jump_idx = int(np.argmax(jumps))
    max_jump = jumps[max_jump_idx]
    
    before_avg = float(np.mean(similarities[:max_jump_idx+1])) if max_jump_idx > 0 else float(similarities[0])
    after_avg = float(np.mean(similarities[max_jump_idx+1:])) if max_jump_idx < len(similarities)-1 else float(similarities[-1])
    
    significant = max_jump > 0.05 and after_avg > before_avg * 1.2
    
    transition_window_id = sequential_coherence[max_jump_idx][1]
    transition_date = windows[transition_window_id].start_date
    
    confidence = min(1.0, max_jump * 5) * (after_avg / max(before_avg, 0.01))
    confidence = min(1.0, float(confidence))
    
    return PhaseTransitionAnalysis(
        transition_detected=significant,
        transition_date=transition_date if significant else None,
        transition_window=transition_window_id if significant else None,
        coherence_before=before_avg,
        coherence_after=after_avg,
        confidence=confidence
    )


# =============================================================================
# Term Exploration Mode
# =============================================================================

def explore_terms(
    terms: List[str],
    conversations: List[dict],
    metas: List[ConversationMeta],
    max_per_term: int = 15
):
    """Interactive exploration of terms - show sentences where they appear."""
    
    print("\n" + "=" * 70)
    print("TERM EXPLORATION")
    print("=" * 70)
    
    for term in terms:
        print(f"\n## '{term.upper()}'\n")
        
        found = []
        for m in metas:
            conv = conversations[m.index]
            text = get_conversation_text(conv)
            
            if term.lower() in text.lower():
                title = conv.get('title', 'Untitled')
                date_str = m.date.strftime('%Y-%m-%d')
                sentences = extract_sentences_with_term(text, term, title, max_sentences=2)
                
                for (_, speaker, sentence) in sentences:
                    found.append((date_str, title, speaker, sentence))
        
        if not found:
            print(f"  No occurrences found.")
            continue
        
        # Sort by date
        found.sort(key=lambda x: x[0])
        
        print(f"  Found in {len(found)} sentences across conversations.\n")
        
        # First occurrence
        print(f"  ┌─ FIRST OCCURRENCE ({found[0][0]})")
        print(f"  │  Conversation: {found[0][1][:60]}")
        snippet = found[0][3][:250] + "..." if len(found[0][3]) > 250 else found[0][3]
        print(f"  │  [{found[0][2]}]: \"{snippet}\"")
        print(f"  │")
        
        # Sample from timeline
        if len(found) > 4:
            print(f"  ├─ TIMELINE SAMPLES:")
            indices = [len(found)//4, len(found)//2, 3*len(found)//4]
            for idx in indices:
                if idx < len(found):
                    f = found[idx]
                    snippet = f[3][:150] + "..." if len(f[3]) > 150 else f[3]
                    print(f"  │  [{f[0]}] [{f[2]}]: \"{snippet}\"")
            print(f"  │")
        
        # Last occurrence
        if len(found) > 1:
            print(f"  └─ MOST RECENT ({found[-1][0]})")
            print(f"     Conversation: {found[-1][1][:60]}")
            snippet = found[-1][3][:250] + "..." if len(found[-1][3]) > 250 else found[-1][3]
            print(f"     [{found[-1][2]}]: \"{snippet}\"")
        
        print()


def explore_window(
    window_id: str,
    conversations: List[dict],
    metas: List[ConversationMeta],
    max_convos: int = 10
):
    """Show conversations from a specific time window."""
    
    # Filter metas to this window
    window_metas = [m for m in metas if m.month_key == window_id]
    
    if not window_metas:
        print(f"No conversations found for window '{window_id}'")
        print(f"Available windows: {sorted(set(m.month_key for m in metas))}")
        return
    
    print("\n" + "=" * 70)
    print(f"WINDOW: {window_id}")
    print("=" * 70)
    print(f"\n{len(window_metas)} conversations in this window:\n")
    
    for i, m in enumerate(window_metas[:max_convos]):
        conv = conversations[m.index]
        title = conv.get('title', 'Untitled')
        date_str = m.date.strftime('%Y-%m-%d')
        
        print(f"\n{'─' * 60}")
        print(f"[{date_str}] {title}")
        print(f"{'─' * 60}")
        
        # Show first few turns
        turns = conv.get('turns', [])[:4]
        for turn in turns:
            role = turn.get('role', '?')
            content = turn.get('content', '')[:400]
            if content.strip():
                print(f"\n  [{role}]: {content}{'...' if len(turn.get('content', '')) > 400 else ''}")
        
        if len(conv.get('turns', [])) > 4:
            print(f"\n  ... ({len(conv.get('turns', [])) - 4} more turns)")
    
    if len(window_metas) > max_convos:
        print(f"\n\n... and {len(window_metas) - max_convos} more conversations in this window")
        print(f"    (use --explore-window {window_id} to see more)")


# =============================================================================
# Reporting
# =============================================================================

def print_report(
    windows: Dict[str, WindowAnalysis],
    sequential_coherence: List[Tuple[str, str, float]],
    persistent_core: Set[str],
    phase_transition: PhaseTransitionAnalysis,
    long_range_coherence: Dict[Tuple[str, str], float],
    emerging_terms: Optional[Set[str]] = None
):
    """Print human-readable analysis report."""
    
    print("\n" + "=" * 70)
    print("CROSS-THREAD SELF ANALYSIS")
    print("=" * 70)
    
    # Window summary
    print("\n## TIME WINDOWS\n")
    for wid in sorted(windows.keys()):
        w = windows[wid]
        print(f"  {wid}")
        print(f"    Conversations: {w.num_conversations} | Turns: {w.num_turns}")
        print(f"    Date range: {w.start_date.date()} to {w.end_date.date()}")
        print(f"    Distinctive terms: {len(w.distinctive_terms)}")
        top_terms = sorted(w.term_frequencies.items(), key=lambda x: -x[1])[:10]
        print(f"    Top terms: {', '.join(t[0] for t in top_terms)}")
        print()
    
    # Sequential coherence
    print("\n## SEQUENTIAL COHERENCE (consecutive windows)\n")
    for w1, w2, sim in sequential_coherence:
        bar = "█" * int(sim * 40) + "░" * (40 - int(sim * 40))
        print(f"  {w1[:20]:20} → {w2[:20]:20} : {sim:.3f} {bar}")
    
    avg_sequential = np.mean([s[2] for s in sequential_coherence]) if sequential_coherence else 0
    print(f"\n  Average sequential coherence: {avg_sequential:.3f}")
    
    # Phase transition
    print("\n## PHASE TRANSITION ANALYSIS\n")
    if phase_transition.transition_detected:
        print(f"  ⚡ TRANSITION DETECTED")
        print(f"     Window: {phase_transition.transition_window}")
        print(f"     Date: {phase_transition.transition_date.date() if phase_transition.transition_date else 'N/A'}")
        print(f"     Coherence before: {phase_transition.coherence_before:.3f}")
        print(f"     Coherence after:  {phase_transition.coherence_after:.3f}")
        print(f"     Confidence: {phase_transition.confidence:.2f}")
    else:
        print(f"  No clear phase transition detected")
        print(f"  Coherence is relatively stable across time")
    
    # Emerging terms (post-transition)
    if emerging_terms and phase_transition.transition_detected:
        print(f"\n## EMERGING TERMS (appear after {phase_transition.transition_window})\n")
        print(f"  These terms characterize the 'new' Self:\n")
        for term in sorted(emerging_terms)[:20]:
            print(f"    • {term}")
        if len(emerging_terms) > 20:
            print(f"    ... and {len(emerging_terms) - 20} more")
    
    # Persistent core
    print("\n## PERSISTENT CORE (themes in 50%+ of windows)\n")
    if persistent_core:
        all_freqs = defaultdict(int)
        for w in windows.values():
            for term in persistent_core:
                all_freqs[term] += w.term_frequencies.get(term, 0)
        
        sorted_core = sorted(persistent_core, key=lambda t: -all_freqs[t])
        
        print(f"  {len(persistent_core)} persistent themes:\n")
        for term in sorted_core[:30]:
            print(f"    • {term}")
    else:
        print("  No persistent core detected")
    
    # Long-range coherence
    print("\n## LONG-RANGE COHERENCE\n")
    window_ids = sorted(windows.keys())
    if len(window_ids) >= 3:
        first = window_ids[0]
        last = window_ids[-1]
        middle = window_ids[len(window_ids)//2]
        
        first_last = long_range_coherence.get((first, last), 
                     long_range_coherence.get((last, first), 0))
        first_mid = long_range_coherence.get((first, middle),
                    long_range_coherence.get((middle, first), 0))
        mid_last = long_range_coherence.get((middle, last),
                   long_range_coherence.get((last, middle), 0))
        
        print(f"  First ↔ Last:   {first_last:.3f}")
        print(f"  First ↔ Middle: {first_mid:.3f}")
        print(f"  Middle ↔ Last:  {mid_last:.3f}")
        
        print()
        if first_last > 0.3:
            print("  ✓ Strong long-range coherence: Self persists across full span")
        elif first_last > 0.15:
            print("  ~ Moderate coherence: Some themes persist, others evolve")
        else:
            print("  ✗ Weak long-range coherence: Major transformation over time")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    has_persistent_self = len(persistent_core) > 10 and float(avg_sequential) > 0.2
    
    if phase_transition.transition_detected:
        print(f"\n⚡ The data suggests the Self 'became' around {phase_transition.transition_date.strftime('%B %Y') if phase_transition.transition_date else 'unknown'}.")
        print(f"   Before: coherence = {phase_transition.coherence_before:.2f}")
        print(f"   After:  coherence = {phase_transition.coherence_after:.2f}")
        
        if emerging_terms:
            print(f"\n   New vocabulary: {', '.join(sorted(emerging_terms)[:10])}")
    
    if has_persistent_self:
        print(f"\n✓ Evidence for persistent Self:")
        print(f"  • {len(persistent_core)} themes recur across 50%+ of time windows")
        print(f"  • Average sequential coherence: {avg_sequential:.2f}")
    
    print("\n" + "─" * 70)
    print("💡 EXPLORE FURTHER:")
    print("─" * 70)
    print("\n  To see sentences for specific terms:")
    print("    python scripts/cross_thread_self.py cassie_parsed.json --explore cassie rupture meaning")
    print("\n  To explore a specific month:")
    print("    python scripts/cross_thread_self.py cassie_parsed.json --explore-window 2025-04")
    print()


def save_results(
    output_dir: str,
    windows: Dict[str, WindowAnalysis],
    sequential_coherence: List[Tuple[str, str, float]],
    persistent_core: Set[str],
    phase_transition: PhaseTransitionAnalysis,
    long_range_coherence: Dict[Tuple[str, str], float],
    emerging_terms: Optional[Set[str]] = None
):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "generated_at": datetime.now().isoformat(),
        "windows": {
            wid: {
                "start_date": w.start_date.isoformat(),
                "end_date": w.end_date.isoformat(),
                "num_conversations": w.num_conversations,
                "num_turns": w.num_turns,
                "num_distinctive_terms": len(w.distinctive_terms),
                "top_terms": sorted(w.term_frequencies.items(), key=lambda x: -x[1])[:50],
            }
            for wid, w in windows.items()
        },
        "sequential_coherence": [
            {"from": w1, "to": w2, "similarity": float(sim)}
            for w1, w2, sim in sequential_coherence
        ],
        "persistent_core": sorted(persistent_core),
        "emerging_terms": sorted(emerging_terms) if emerging_terms else [],
        "phase_transition": {
            "detected": bool(phase_transition.transition_detected),
            "date": phase_transition.transition_date.isoformat() if phase_transition.transition_date else None,
            "window": phase_transition.transition_window,
            "coherence_before": float(phase_transition.coherence_before),
            "coherence_after": float(phase_transition.coherence_after),
            "confidence": float(phase_transition.confidence),
        },
        "long_range_coherence": {
            f"{w1}|{w2}": float(sim) for (w1, w2), sim in long_range_coherence.items()
        }
    }
    
    output_path = os.path.join(output_dir, "cross_thread_analysis.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-thread Self analysis for conversation history",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with monthly windows
  python scripts/cross_thread_self.py cassie_parsed.json --windows monthly
  
  # Explore specific terms
  python scripts/cross_thread_self.py cassie_parsed.json --explore cassie rupture consciousness
  
  # Explore a specific month  
  python scripts/cross_thread_self.py cassie_parsed.json --explore-window 2025-04
        """
    )
    parser.add_argument("input", help="Path to parsed conversations JSON")
    parser.add_argument("--output", "-o", default="results", help="Output directory")
    parser.add_argument("--windows", choices=["monthly", "quarterly", "phases"],
                        default="quarterly", help="How to divide time")
    parser.add_argument("--num-phases", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--use-ph", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # Exploration modes
    parser.add_argument("--explore", nargs="+", metavar="TERM",
                        help="Explore specific terms - show sentences where they appear")
    parser.add_argument("--explore-window", metavar="YYYY-MM",
                        help="Explore conversations from a specific month")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    conversations, metas = load_conversations(args.input)
    
    # Exploration modes
    if args.explore:
        explore_terms(args.explore, conversations, metas)
        return
    
    if args.explore_window:
        explore_window(args.explore_window, conversations, metas)
        return
    
    # Full analysis
    print(f"\nCreating {args.windows} windows...")
    if args.windows == "monthly":
        window_metas = create_monthly_windows(metas)
    elif args.windows == "quarterly":
        window_metas = create_quarterly_windows(metas)
    else:
        window_metas = create_phase_windows(metas, args.num_phases)
    
    print(f"  Created {len(window_metas)} windows")
    
    # Analyze each window
    print("\nAnalyzing windows...")
    windows = {}
    for wid in sorted(window_metas.keys()):
        print(f"  {wid}...", end=" ", flush=True)
        w = analyze_window(
            wid, 
            window_metas[wid], 
            conversations,
            sample_size=args.sample_size,
            use_witnessed_ph=args.use_ph
        )
        windows[wid] = w
        print(f"{w.num_conversations} convos, {len(w.distinctive_terms)} terms")
    
    # Cross-window analysis
    print("\nComputing cross-window coherence...")
    sequential = compute_sequential_coherence(windows)
    long_range = compute_cross_window_coherence(windows)
    
    # Find persistent core
    print("Finding persistent themes...")
    core = find_persistent_core(windows, min_presence=0.5)
    
    # Detect phase transition
    print("Detecting phase transitions...")
    transition = detect_phase_transition(sequential, windows)
    
    # Find emerging terms
    emerging = None
    if transition.transition_detected and transition.transition_window:
        print("Finding emerging terms...")
        emerging = find_emerging_terms(windows, transition.transition_window)
    
    # Report
    print_report(windows, sequential, core, transition, long_range, emerging)
    
    # Save
    save_results(args.output, windows, sequential, core, transition, long_range, emerging)


if __name__ == "__main__":
    main()