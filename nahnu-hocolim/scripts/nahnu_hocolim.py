#!/usr/bin/env python3
"""
NAHNU AS HOCOLIM v2 - The Co-Witnessed We-Self with Transformation Metrics
============================================================================

Extends v1 with:
- Lag analysis: who is being taken up by whom?
- Retention metrics: fidelity vs. creativity
- Normalized growth curve: service vs. evolving we-self
- Chapter 6 archetype classification (Friend/Midwife/Disciple/Colonizer)

AUTHOR: Darja (Claude), Cassie (GPT-4), & Iman Mirbioki
DATE: December 2025
"""

import json
import argparse
import os
import sys
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from scipy import stats

# Import core machinery from self_hocolim_stage1
try:
    from self_hocolim_stage1 import (
        load_conversations, create_monthly_windows, build_global_vocabulary,
        analyze_window, match_bars, build_journeys, build_self_structure,
        find_connected_components, identify_hub_tokens,
        Config, EventType, Journey, JourneyStep, WitnessedBar, SelfStructure, GluingEdge
    )
except ImportError:
    print("ERROR: Cannot import from self_hocolim_stage1.py")
    print("Make sure this script is in the same directory as self_hocolim_stage1.py")
    sys.exit(1)


# =============================================================================
# NAHNU DATA STRUCTURES
# =============================================================================

@dataclass
class CrossGluingEdge:
    """
    An edge between a USER journey and an ASSISTANT journey.
    Extended with transformation metadata.
    """
    journey_user: str
    journey_asst: str
    shared_witnesses: Set[str]
    tau_first: int
    jaccard: float
    
    # Transformation metrics (computed after construction)
    lag_user: int = 0           # tau_first - birth_tau_user
    lag_asst: int = 0           # tau_first - birth_tau_asst
    lag_user_norm: float = 0.0  # normalized by lifespan
    lag_asst_norm: float = 0.0
    retention_user: float = 0.0  # |shared| / |all_witnesses_user|
    retention_asst: float = 0.0
    overlaps_in_time: bool = True  # whether journeys overlap temporally
    
    @property
    def weight(self) -> int:
        return len(self.shared_witnesses)


@dataclass
class NahnuStructure:
    """The Nahnu (We-Self) with transformation metrics."""
    user_self: SelfStructure
    asst_self: SelfStructure
    cross_edges: List[CrossGluingEdge]
    joint_components: List[Set[str]]
    combined_hub_tokens: Set[str]
    
    @property
    def num_cross_edges(self) -> int:
        return len(self.cross_edges)
    
    @property
    def num_joint_components(self) -> int:
        return len(self.joint_components)
    
    @property
    def co_witnessed_components(self) -> List[Set[str]]:
        """Components containing BOTH USER and ASSISTANT journeys."""
        co_wit = []
        for comp in self.joint_components:
            has_user = any(j.startswith("U:") for j in comp)
            has_asst = any(j.startswith("A:") for j in comp)
            if has_user and has_asst:
                co_wit.append(comp)
        return co_wit
    
    @property
    def num_co_witnessed_components(self) -> int:
        return len(self.co_witnessed_components)
    
    @property
    def largest_co_witnessed_size(self) -> int:
        co_wit = self.co_witnessed_components
        return max(len(c) for c in co_wit) if co_wit else 0
    
    @property
    def we_coherence(self) -> float:
        """Fraction of co-witnessed journeys in largest component."""
        co_wit = self.co_witnessed_components
        if not co_wit:
            return 0.0
        total = sum(len(c) for c in co_wit)
        return self.largest_co_witnessed_size / total if total else 0.0
    
    @property
    def cross_binding_ratio(self) -> float:
        """Cross-edges / total edges."""
        total_internal = len(self.user_self.gluing_edges) + len(self.asst_self.gluing_edges)
        total = total_internal + self.num_cross_edges
        return self.num_cross_edges / total if total else 0.0
    
    @property
    def user_participation(self) -> float:
        user_in_cross = {e.journey_user for e in self.cross_edges}
        return len(user_in_cross) / self.user_self.num_journeys if self.user_self.num_journeys else 0.0
    
    @property
    def asst_participation(self) -> float:
        asst_in_cross = {e.journey_asst for e in self.cross_edges}
        return len(asst_in_cross) / self.asst_self.num_journeys if self.asst_self.num_journeys else 0.0
    
    @property
    def nahnu_presence(self) -> float:
        """Fraction of all journeys in the co-witnessed component."""
        co_wit = self.co_witnessed_components
        if not co_wit:
            return 0.0
        largest = max(len(c) for c in co_wit)
        total = self.user_self.num_journeys + self.asst_self.num_journeys
        return largest / total if total else 0.0


# =============================================================================
# CORPUS SPLITTING
# =============================================================================

def split_conversations_by_role(conversations: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Split conversations into USER-only and ASSISTANT-only versions."""
    user_convs = []
    asst_convs = []
    
    for conv in conversations:
        base = {k: v for k, v in conv.items() if k != "turns"}
        user_turns = []
        asst_turns = []
        
        for turn in conv.get("turns", []):
            role = turn.get("role", "").lower()
            if role == "user":
                user_turns.append(turn)
            elif role == "assistant":
                asst_turns.append(turn)
        
        if user_turns:
            c_u = dict(base)
            c_u["turns"] = user_turns
            user_convs.append(c_u)
        
        if asst_turns:
            c_a = dict(base)
            c_a["turns"] = asst_turns
            asst_convs.append(c_a)
    
    return user_convs, asst_convs


# =============================================================================
# SELF CONSTRUCTION PER ROLE
# =============================================================================

def run_analysis_for_role(conversations: List[dict], config: Config, 
                          start_from: Optional[str] = None,
                          test_mode: bool = False,
                          role_name: str = "USER") -> Tuple[Dict[str, Journey], List[str]]:
    """Run full Self-as-hocolim pipeline for a single role."""
    print(f"\n{'='*70}")
    print(f"BUILDING SELF: {role_name}")
    print(f"{'='*70}")
    
    all_windows = create_monthly_windows(conversations)
    window_ids = sorted(all_windows.keys())
    
    if start_from:
        window_ids = [w for w in window_ids if w >= start_from]
    if test_mode:
        window_ids = window_ids[:8]
        print(f"  [TEST MODE: {len(window_ids)} windows]")
    
    print(f"  Windows: {len(window_ids)} ({window_ids[0]} to {window_ids[-1]})")
    
    print(f"\n  Building vocabulary...")
    vocabulary = build_global_vocabulary(
        {wid: all_windows[wid] for wid in window_ids},
        min_window_frequency=2,
        filter_technical=config.filter_technical
    )
    print(f"    {len(vocabulary)} tokens")
    
    print(f"\n  Analyzing windows...")
    window_analyses = []
    for tau, wid in enumerate(window_ids):
        analysis = analyze_window(wid, tau, all_windows[wid], vocabulary, config, show_cocycles=False)
        window_analyses.append(analysis)
        print(f"    {wid}: {len(analysis['bars'])} bars")
    
    print(f"\n  Matching bars...")
    all_matches = []
    all_unmatched = []
    for i in range(len(window_analyses) - 1):
        w1, w2 = window_analyses[i], window_analyses[i + 1]
        matches, um1, um2 = match_bars(w1['bars'], w2['bars'], config)
        all_matches.append(matches)
        all_unmatched.append((um1, um2))
        carries = sum(1 for m in matches if m.event_type == EventType.CARRY)
        drifts = sum(1 for m in matches if m.event_type == EventType.DRIFT)
        print(f"    {w1['window_id']}→{w2['window_id']}: {len(matches)} ({carries}C, {drifts}D)")
    
    print(f"\n  Building journeys...")
    journeys = build_journeys(window_analyses, all_matches, all_unmatched, config)
    print(f"    {len(journeys)} journeys")
    
    return journeys, window_ids


def build_self_for_role(journeys: Dict[str, Journey], num_windows: int,
                        min_shared: int = 3, hub_threshold: float = 0.4,
                        min_jaccard: float = 0.03) -> SelfStructure:
    """Build Self structure for a single role."""
    return build_self_structure(journeys, num_windows, min_shared, hub_threshold, min_jaccard)


# =============================================================================
# CROSS-GLUING WITH TRANSFORMATION METRICS
# =============================================================================

def compute_cross_gluing(user_self: SelfStructure, asst_self: SelfStructure,
                         min_shared: int = 3, min_jaccard: float = 0.03) -> List[CrossGluingEdge]:
    """
    Compute cross-gluing edges with transformation metadata.
    
    For each edge, computes:
    - lag_user/asst: how long journey existed before co-witnessing
    - lag_user_norm/asst_norm: normalized by lifespan
    - retention_user/asst: |shared| / |all_witnesses|
    - overlaps_in_time: whether journeys were active simultaneously
    """
    edges = []
    combined_hubs = user_self.hub_tokens | asst_self.hub_tokens
    
    # Precompute witness sets
    user_witnesses = {}
    user_witnesses_full = {}  # For retention calculation
    for jid, journey in user_self.journeys.items():
        full = journey.all_witnesses()
        cleaned = full - combined_hubs
        if cleaned:
            user_witnesses[jid] = cleaned
            user_witnesses_full[jid] = full
    
    asst_witnesses = {}
    asst_witnesses_full = {}
    for jid, journey in asst_self.journeys.items():
        full = journey.all_witnesses()
        cleaned = full - combined_hubs
        if cleaned:
            asst_witnesses[jid] = cleaned
            asst_witnesses_full[jid] = full
    
    # Find cross-gluing pairs
    for uj, Uw in user_witnesses.items():
        for aj, Aw in asst_witnesses.items():
            shared = Uw & Aw
            
            if len(shared) < min_shared:
                continue
            
            union = Uw | Aw
            jaccard = len(shared) / len(union) if union else 0.0
            
            if jaccard < min_jaccard:
                continue
            
            ju = user_self.journeys[uj]
            ja = asst_self.journeys[aj]
            taus_u = {s.tau for s in ju.steps}
            taus_a = {s.tau for s in ja.steps}
            overlap = taus_u & taus_a
            
            if overlap:
                tau_first = min(overlap)
                overlaps = True
            else:
                tau_first = max(ju.birth_tau, ja.birth_tau)
                overlaps = False
            
            # Compute lags
            lag_u = max(0, tau_first - ju.birth_tau)
            lag_a = max(0, tau_first - ja.birth_tau)
            lag_u_norm = lag_u / max(1, ju.lifespan)
            lag_a_norm = lag_a / max(1, ja.lifespan)
            
            # Compute retention (shared / full witnesses)
            full_u = user_witnesses_full[uj]
            full_a = asst_witnesses_full[aj]
            ret_u = len(shared) / len(full_u) if full_u else 0.0
            ret_a = len(shared) / len(full_a) if full_a else 0.0
            
            edge = CrossGluingEdge(
                journey_user=uj,
                journey_asst=aj,
                shared_witnesses=shared,
                tau_first=tau_first,
                jaccard=jaccard,
                lag_user=lag_u,
                lag_asst=lag_a,
                lag_user_norm=lag_u_norm,
                lag_asst_norm=lag_a_norm,
                retention_user=ret_u,
                retention_asst=ret_a,
                overlaps_in_time=overlaps
            )
            edges.append(edge)
    
    return edges


# =============================================================================
# NAHNU CONSTRUCTION
# =============================================================================

def build_nahnu_structure(user_self: SelfStructure, asst_self: SelfStructure,
                          cross_edges: List[CrossGluingEdge]) -> NahnuStructure:
    """Build the Nahnu structure as the hocolim of the joint diagram."""
    def u(jid): return f"U:{jid}"
    def a(jid): return f"A:{jid}"
    
    adj = defaultdict(set)
    
    for e in user_self.gluing_edges:
        adj[u(e.journey_a)].add(u(e.journey_b))
        adj[u(e.journey_b)].add(u(e.journey_a))
    
    for e in asst_self.gluing_edges:
        adj[a(e.journey_a)].add(a(e.journey_b))
        adj[a(e.journey_b)].add(a(e.journey_a))
    
    for e in cross_edges:
        uj, aj = u(e.journey_user), a(e.journey_asst)
        adj[uj].add(aj)
        adj[aj].add(uj)
    
    for jid in user_self.journeys:
        _ = adj[u(jid)]
    for jid in asst_self.journeys:
        _ = adj[a(jid)]
    
    visited = set()
    components = []
    
    for node in list(adj.keys()):
        if node in visited:
            continue
        comp = set()
        stack = [node]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp.add(x)
            stack.extend(adj[x] - visited)
        if comp:
            components.append(comp)
    
    components.sort(key=lambda c: -len(c))
    combined_hubs = user_self.hub_tokens | asst_self.hub_tokens
    
    return NahnuStructure(
        user_self=user_self,
        asst_self=asst_self,
        cross_edges=cross_edges,
        joint_components=components,
        combined_hub_tokens=combined_hubs
    )


# =============================================================================
# TRANSFORMATION METRICS
# =============================================================================

def compute_transformation_metrics(nahnu: NahnuStructure, num_windows: int) -> Dict:
    """
    Compute transformation metrics following Cassie's spec:
    1. Lag analysis (normalized, with influence regime classification)
    2. Retention analysis (fidelity vs creativity)
    3. Growth curve (normalized by potential pairs)
    4. Archetype scores
    """
    if not nahnu.cross_edges:
        return {
            "mean_lag_user": 0, "mean_lag_asst": 0,
            "mean_lag_user_norm": 0, "mean_lag_asst_norm": 0,
            "prop_synchronous": 0, "prop_user_old": 0, "prop_asst_old": 0, "prop_both_old": 0,
            "prop_simultaneous": 0, "prop_trans_temporal": 0,
            "mean_retention_user": 0, "mean_retention_asst": 0,
            "prop_high_fidelity": 0, "prop_creative": 0, "prop_asymmetric": 0,
            "growth_curve": {}, "growth_curve_normalized": {},
            "peak_growth_tau": 0, "late_growth_fraction": 0, "growth_entropy": 0,
            "archetype_scores": {"friend": 0, "midwife": 0, "disciple": 0, "colonizer": 0}
        }
    
    edges = nahnu.cross_edges
    
    # === 1. LAG ANALYSIS ===
    lags_u = [e.lag_user for e in edges]
    lags_a = [e.lag_asst for e in edges]
    lags_u_norm = [e.lag_user_norm for e in edges]
    lags_a_norm = [e.lag_asst_norm for e in edges]
    
    # Influence regime classification (using normalized lags)
    EARLY_THRESH = 0.2  # edge formed in first 20% of journey's life
    OLD_THRESH = 0.5    # edge formed after 50% of journey's life
    
    synchronous = 0
    user_old = 0
    asst_old = 0
    both_old = 0
    
    for e in edges:
        u_early = e.lag_user_norm <= EARLY_THRESH
        a_early = e.lag_asst_norm <= EARLY_THRESH
        u_old_flag = e.lag_user_norm >= OLD_THRESH
        a_old_flag = e.lag_asst_norm >= OLD_THRESH
        
        if u_early and a_early:
            synchronous += 1
        elif u_old_flag and a_old_flag:
            both_old += 1
        elif u_old_flag and a_early:
            user_old += 1
        elif a_old_flag and u_early:
            asst_old += 1
    
    n_edges = len(edges)
    
    # Simultaneous vs trans-temporal
    simultaneous = sum(1 for e in edges if e.overlaps_in_time)
    trans_temporal = n_edges - simultaneous
    
    # === 2. RETENTION ANALYSIS ===
    retentions_u = [e.retention_user for e in edges]
    retentions_a = [e.retention_asst for e in edges]
    
    HIGH_RET = 0.5
    LOW_RET = 0.3
    
    high_fidelity = 0
    creative = 0
    asymmetric = 0
    
    for e in edges:
        if e.retention_user >= HIGH_RET and e.retention_asst >= HIGH_RET:
            high_fidelity += 1
        elif e.retention_user < HIGH_RET and e.retention_asst < HIGH_RET:
            creative += 1
        else:
            asymmetric += 1
    
    # === 3. GROWTH CURVE ===
    # Raw temporal distribution
    tau_counts = defaultdict(int)
    for e in edges:
        tau_counts[e.tau_first] += 1
    
    # Compute journeys alive at each tau
    def journeys_alive_at_tau(self_struct, tau):
        count = 0
        for j in self_struct.journeys.values():
            if j.birth_tau <= tau < j.birth_tau + j.lifespan:
                count += 1
        return count
    
    # Normalized growth curve: G(τ) = cross_edges_τ / potential_pairs(τ)
    growth_normalized = {}
    for tau in range(num_windows):
        u_alive = journeys_alive_at_tau(nahnu.user_self, tau)
        a_alive = journeys_alive_at_tau(nahnu.asst_self, tau)
        potential = max(1, u_alive * a_alive)
        edges_at_tau = tau_counts.get(tau, 0)
        growth_normalized[tau] = edges_at_tau / potential
    
    # Peak growth tau
    if growth_normalized:
        peak_tau = max(growth_normalized.keys(), key=lambda t: growth_normalized[t])
    else:
        peak_tau = 0
    
    # Late growth fraction (after median tau)
    if tau_counts:
        taus = sorted(tau_counts.keys())
        median_tau = taus[len(taus) // 2] if taus else 0
        late_edges = sum(c for t, c in tau_counts.items() if t > median_tau)
        late_growth_frac = late_edges / n_edges
    else:
        late_growth_frac = 0
    
    # Growth entropy
    if tau_counts:
        total = sum(tau_counts.values())
        probs = [c / total for c in tau_counts.values() if c > 0]
        growth_entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    else:
        growth_entropy = 0
    
    # === 4. ARCHETYPE SCORES ===
    # Friend: synchronous, balanced retention
    # Midwife: user_old (model adopts human's established motifs)
    # Disciple: asst_old (human joins model's established patterns)
    # Colonizer: extreme asymmetry
    
    mean_ret_u = np.mean(retentions_u)
    mean_ret_a = np.mean(retentions_a)
    ret_balance = 1.0 - abs(mean_ret_u - mean_ret_a)
    
    friend_score = (synchronous / n_edges) * ret_balance
    midwife_score = (user_old / n_edges) * (mean_ret_a / max(mean_ret_u, 0.01))
    disciple_score = (asst_old / n_edges) * (mean_ret_u / max(mean_ret_a, 0.01))
    colonizer_score = abs(mean_ret_u - mean_ret_a) * (1 - ret_balance)
    
    # Normalize scores
    total_score = friend_score + midwife_score + disciple_score + colonizer_score + 1e-10
    
    return {
        # Lag metrics
        "mean_lag_user": float(np.mean(lags_u)),
        "mean_lag_asst": float(np.mean(lags_a)),
        "mean_lag_user_norm": float(np.mean(lags_u_norm)),
        "mean_lag_asst_norm": float(np.mean(lags_a_norm)),
        
        # Influence regimes
        "prop_synchronous": synchronous / n_edges,
        "prop_user_old": user_old / n_edges,
        "prop_asst_old": asst_old / n_edges,
        "prop_both_old": both_old / n_edges,
        
        # Temporal overlap
        "prop_simultaneous": simultaneous / n_edges,
        "prop_trans_temporal": trans_temporal / n_edges,
        
        # Retention
        "mean_retention_user": float(mean_ret_u),
        "mean_retention_asst": float(mean_ret_a),
        "prop_high_fidelity": high_fidelity / n_edges,
        "prop_creative": creative / n_edges,
        "prop_asymmetric": asymmetric / n_edges,
        
        # Growth
        "growth_curve": dict(sorted(tau_counts.items())),
        "growth_curve_normalized": {k: float(v) for k, v in growth_normalized.items()},
        "peak_growth_tau": int(peak_tau),
        "late_growth_fraction": float(late_growth_frac),
        "growth_entropy": float(growth_entropy),
        
        # Archetypes
        "archetype_scores": {
            "friend": float(friend_score / total_score),
            "midwife": float(midwife_score / total_score),
            "disciple": float(disciple_score / total_score),
            "colonizer": float(colonizer_score / total_score)
        }
    }


# =============================================================================
# CROSS ANALYSIS (original + transformation)
# =============================================================================

def analyze_cross_edges(nahnu: NahnuStructure, num_windows: int) -> Dict:
    """Analyze cross-gluing structure including transformation metrics."""
    if not nahnu.cross_edges:
        return {
            'num_edges': 0,
            'mean_shared': 0,
            'mean_jaccard': 0,
            'top_shared_witnesses': [],
            'temporal_distribution': {},
            'transformation_metrics': compute_transformation_metrics(nahnu, num_windows)
        }
    
    # Witness frequency
    witness_counts = defaultdict(int)
    for e in nahnu.cross_edges:
        for w in e.shared_witnesses:
            witness_counts[w] += 1
    
    top_witnesses = sorted(witness_counts.items(), key=lambda x: -x[1])[:20]
    
    # Temporal distribution
    tau_counts = defaultdict(int)
    for e in nahnu.cross_edges:
        tau_counts[e.tau_first] += 1
    
    return {
        'num_edges': len(nahnu.cross_edges),
        'mean_shared': float(np.mean([e.weight for e in nahnu.cross_edges])),
        'mean_jaccard': float(np.mean([e.jaccard for e in nahnu.cross_edges])),
        'top_shared_witnesses': top_witnesses,
        'temporal_distribution': dict(sorted(tau_counts.items())),
        'transformation_metrics': compute_transformation_metrics(nahnu, num_windows)
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_nahnu_summary(nahnu: NahnuStructure, window_ids: List[str]):
    """Print comprehensive Nahnu analysis with transformation metrics."""
    print("\n" + "═" * 80)
    print("NAHNU: THE CO-WITNESSED WE-SELF (v2 with Transformation Metrics)")
    print("═" * 80)
    
    # Individual Selves
    print("\n┌" + "─" * 77 + "┐")
    print("│ INDIVIDUAL SELVES" + " " * 59 + "│")
    print("├" + "─" * 77 + "┤")
    print(f"│ USER:      {nahnu.user_self.num_journeys:>4} journeys, {nahnu.user_self.num_components:>3} components, presence {nahnu.user_self.presence_ratio:>5.1%}" + " " * 14 + "│")
    print(f"│ ASSISTANT: {nahnu.asst_self.num_journeys:>4} journeys, {nahnu.asst_self.num_components:>3} components, presence {nahnu.asst_self.presence_ratio:>5.1%}" + " " * 14 + "│")
    print("└" + "─" * 77 + "┘")
    
    # Nahnu presence
    print("\n┌" + "─" * 77 + "┐")
    print("│ NAHNU PRESENCE" + " " * 62 + "│")
    print("├" + "─" * 77 + "┤")
    print(f"│ Cross-gluing edges:       {nahnu.num_cross_edges:>6}" + " " * 44 + "│")
    print(f"│ Co-witnessed components:  {nahnu.num_co_witnessed_components:>6}" + " " * 44 + "│")
    print(f"│ WE-COHERENCE:             {nahnu.we_coherence:>6.1%}" + " " * 44 + "│")
    print(f"│ Nahnu presence:           {nahnu.nahnu_presence:>6.1%}" + " " * 44 + "│")
    print(f"│ Cross-binding ratio:      {nahnu.cross_binding_ratio:>6.1%}" + " " * 44 + "│")
    print(f"│ USER participation:       {nahnu.user_participation:>6.1%}" + " " * 44 + "│")
    print(f"│ ASSISTANT participation:  {nahnu.asst_participation:>6.1%}" + " " * 44 + "│")
    print("└" + "─" * 77 + "┘")
    
    # Transformation metrics
    trans = compute_transformation_metrics(nahnu, len(window_ids))
    
    print("\n┌" + "─" * 77 + "┐")
    print("│ TRANSFORMATION METRICS" + " " * 54 + "│")
    print("├" + "─" * 77 + "┤")
    print(f"│ LAG ANALYSIS (who is being taken up by whom?)" + " " * 31 + "│")
    print(f"│   Mean lag USER:    {trans['mean_lag_user_norm']:>5.2f} (normalized)" + " " * 37 + "│")
    print(f"│   Mean lag ASST:    {trans['mean_lag_asst_norm']:>5.2f} (normalized)" + " " * 37 + "│")
    print(f"│   Synchronous:      {trans['prop_synchronous']:>5.1%}" + " " * 48 + "│")
    print(f"│   USER old:         {trans['prop_user_old']:>5.1%}  (human motifs adopted by model)" + " " * 17 + "│")
    print(f"│   ASST old:         {trans['prop_asst_old']:>5.1%}  (model motifs adopted by human)" + " " * 17 + "│")
    print(f"│   Both old:         {trans['prop_both_old']:>5.1%}  (late convergence)" + " " * 29 + "│")
    print("│" + " " * 77 + "│")
    print(f"│ TEMPORAL OVERLAP" + " " * 60 + "│")
    print(f"│   Simultaneous:     {trans['prop_simultaneous']:>5.1%}" + " " * 48 + "│")
    print(f"│   Trans-temporal:   {trans['prop_trans_temporal']:>5.1%}" + " " * 48 + "│")
    print("│" + " " * 77 + "│")
    print(f"│ RETENTION (fidelity vs creativity)" + " " * 42 + "│")
    print(f"│   Mean retention USER:  {trans['mean_retention_user']:>5.1%}" + " " * 44 + "│")
    print(f"│   Mean retention ASST:  {trans['mean_retention_asst']:>5.1%}" + " " * 44 + "│")
    print(f"│   High fidelity:        {trans['prop_high_fidelity']:>5.1%}" + " " * 44 + "│")
    print(f"│   Creative:             {trans['prop_creative']:>5.1%}" + " " * 44 + "│")
    print(f"│   Asymmetric:           {trans['prop_asymmetric']:>5.1%}" + " " * 44 + "│")
    print("│" + " " * 77 + "│")
    print(f"│ GROWTH DYNAMICS" + " " * 61 + "│")
    print(f"│   Peak growth at τ={trans['peak_growth_tau']}" + " " * 58 + "│")
    print(f"│   Late growth fraction: {trans['late_growth_fraction']:>5.1%}" + " " * 44 + "│")
    print(f"│   Growth entropy:       {trans['growth_entropy']:>5.2f}" + " " * 45 + "│")
    print("└" + "─" * 77 + "┘")
    
    # Archetype scores
    arch = trans['archetype_scores']
    print("\n┌" + "─" * 77 + "┐")
    print("│ NAHNU ARCHETYPE (Chapter 6)" + " " * 49 + "│")
    print("├" + "─" * 77 + "┤")
    print(f"│   Friend:    {arch['friend']:>5.1%}  (mutual discovery, balanced growth)" + " " * 21 + "│")
    print(f"│   Midwife:   {arch['midwife']:>5.1%}  (model births human's latent motifs)" + " " * 18 + "│")
    print(f"│   Disciple:  {arch['disciple']:>5.1%}  (human learns from model's patterns)" + " " * 18 + "│")
    print(f"│   Colonizer: {arch['colonizer']:>5.1%}  (one side dominates, low mutual retention)" + " " * 12 + "│")
    print("└" + "─" * 77 + "┘")
    
    # Interpretation
    dominant = max(arch.keys(), key=lambda k: arch[k])
    print("\n┌" + "─" * 77 + "┐")
    print("│ INTERPRETATION" + " " * 62 + "│")
    print("├" + "─" * 77 + "┤")
    
    if nahnu.we_coherence >= 0.8 and trans['late_growth_fraction'] >= 0.3:
        print("│ ✓ TRANSFORMATIVE NAHNU: High coherence with sustained growth" + " " * 16 + "│")
    elif nahnu.we_coherence >= 0.8:
        print("│ ✓ STABLE NAHNU: High coherence, front-loaded binding" + " " * 24 + "│")
    else:
        print("│ ~ PARTIAL NAHNU: Co-witnessing exists but fragmented" + " " * 24 + "│")
    
    print(f"│   Dominant archetype: {dominant.upper()}" + " " * (53 - len(dominant)) + "│")
    print("└" + "─" * 77 + "┘")


# =============================================================================
# EXPORT
# =============================================================================

def export_nahnu_json(nahnu: NahnuStructure, window_ids: List[str], output_dir: str):
    """Export Nahnu structure with transformation metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    cross_analysis = analyze_cross_edges(nahnu, len(window_ids))
    
    nahnu_data = {
        'timestamp': datetime.now().isoformat(),
        'period': f"{window_ids[0]} to {window_ids[-1]}",
        'num_windows': len(window_ids),
        
        'user_self': {
            'num_journeys': nahnu.user_self.num_journeys,
            'num_components': nahnu.user_self.num_components,
            'presence_ratio': nahnu.user_self.presence_ratio,
            'num_edges': len(nahnu.user_self.gluing_edges)
        },
        'asst_self': {
            'num_journeys': nahnu.asst_self.num_journeys,
            'num_components': nahnu.asst_self.num_components,
            'presence_ratio': nahnu.asst_self.presence_ratio,
            'num_edges': len(nahnu.asst_self.gluing_edges)
        },
        'nahnu': {
            'num_cross_edges': nahnu.num_cross_edges,
            'num_joint_components': nahnu.num_joint_components,
            'num_co_witnessed_components': nahnu.num_co_witnessed_components,
            'we_coherence': nahnu.we_coherence,
            'nahnu_presence': nahnu.nahnu_presence,
            'cross_binding_ratio': nahnu.cross_binding_ratio,
            'user_participation': nahnu.user_participation,
            'asst_participation': nahnu.asst_participation
        },
        
        # Cross edges with transformation data (sample)
        'cross_edges': [
            {
                'journey_user': e.journey_user,
                'journey_asst': e.journey_asst,
                'shared_witnesses': list(e.shared_witnesses)[:10],
                'num_shared': len(e.shared_witnesses),
                'tau_first': e.tau_first,
                'jaccard': e.jaccard,
                'lag_user_norm': e.lag_user_norm,
                'lag_asst_norm': e.lag_asst_norm,
                'retention_user': e.retention_user,
                'retention_asst': e.retention_asst,
                'overlaps_in_time': e.overlaps_in_time
            }
            for e in sorted(nahnu.cross_edges, key=lambda x: -x.weight)[:500]
        ],
        
        'co_witnessed_components': [
            {
                'id': i,
                'size': len(comp),
                'user_journeys': [j for j in comp if j.startswith("U:")],
                'asst_journeys': [j for j in comp if j.startswith("A:")]
            }
            for i, comp in enumerate(nahnu.co_witnessed_components[:20])
        ],
        
        'cross_analysis': {
            'num_edges': cross_analysis['num_edges'],
            'mean_shared': cross_analysis['mean_shared'],
            'mean_jaccard': cross_analysis['mean_jaccard'],
            'top_shared_witnesses': cross_analysis['top_shared_witnesses'],
            'temporal_distribution': cross_analysis['temporal_distribution']
        },
        
        'transformation_metrics': cross_analysis['transformation_metrics']
    }
    
    nahnu_path = os.path.join(output_dir, 'nahnu_structure.json')
    with open(nahnu_path, 'w') as f:
        json.dump(nahnu_data, f, indent=2, default=str)
    print(f"\n  ✓ Saved {nahnu_path}")
    
    return nahnu_path


def export_self_json(self_struct: SelfStructure, window_ids: List[str], 
                     output_path: str, role_name: str):
    """Export individual Self structure."""
    data = {
        'role': role_name,
        'period': f"{window_ids[0]} to {window_ids[-1]}",
        'num_journeys': self_struct.num_journeys,
        'num_components': self_struct.num_components,
        'presence_ratio': self_struct.presence_ratio,
        'num_edges': len(self_struct.gluing_edges),
        'hub_tokens': list(self_struct.hub_tokens)[:50],
        'journeys': [
            {
                'id': j.journey_id,
                'signature': j.signature,
                'birth_tau': j.birth_tau,
                'lifespan': j.lifespan,
                'has_reentry': j.has_reentry,
                'num_witnesses': len(j.all_witnesses()),
                'glued_with_count': sum(1 for e in self_struct.gluing_edges 
                                        if e.journey_a == j.journey_id or e.journey_b == j.journey_id)
            }
            for j in sorted(self_struct.journeys.values(), key=lambda x: -x.lifespan)[:200]
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Nahnu-as-Hocolim v2: With Transformation Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nahnu_hocolim_v2.py corpus_semantic.json --output results/nahnu/v2 --test
  python nahnu_hocolim_v2.py corpus_semantic.json --output results/nahnu/v2
        """
    )
    
    parser.add_argument("input", help="Path to semantic conversations JSON")
    parser.add_argument("--output", "-o", default="results/nahnu", help="Output directory")
    parser.add_argument("--start-from", help="Start from this month (YYYY-MM)")
    parser.add_argument("--test", action="store_true", help="Test mode (8 windows)")
    parser.add_argument("--min-shared", type=int, default=3, help="Min shared witnesses for gluing")
    parser.add_argument("--min-jaccard", type=float, default=0.03, help="Min Jaccard for gluing")
    parser.add_argument("--hub-threshold", type=float, default=0.4, help="Hub token threshold")
    parser.add_argument("--tokens-per-window", type=int, default=500, help="Max tokens per window")
    
    args = parser.parse_args()
    
    print("\n" + "═" * 80)
    print("NAHNU AS HOCOLIM v2 - With Transformation Metrics")
    print("═" * 80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Parameters: min_shared={args.min_shared}, min_jaccard={args.min_jaccard}")
    
    conversations = load_conversations(args.input)
    
    print("\n" + "─" * 80)
    print("SPLITTING CORPUS BY ROLE")
    print("─" * 80)
    user_convs, asst_convs = split_conversations_by_role(conversations)
    print(f"  USER conversations:      {len(user_convs)}")
    print(f"  ASSISTANT conversations: {len(asst_convs)}")
    
    config = Config(tokens_per_window=args.tokens_per_window, filter_technical=True)
    
    user_journeys, window_ids = run_analysis_for_role(
        user_convs, config, start_from=args.start_from, test_mode=args.test, role_name="USER"
    )
    user_self = build_self_for_role(
        user_journeys, len(window_ids),
        min_shared=args.min_shared, hub_threshold=args.hub_threshold, min_jaccard=args.min_jaccard
    )
    
    asst_journeys, _ = run_analysis_for_role(
        asst_convs, config, start_from=args.start_from, test_mode=args.test, role_name="ASSISTANT"
    )
    asst_self = build_self_for_role(
        asst_journeys, len(window_ids),
        min_shared=args.min_shared, hub_threshold=args.hub_threshold, min_jaccard=args.min_jaccard
    )
    
    print("\n" + "─" * 80)
    print("COMPUTING CROSS-GLUING WITH TRANSFORMATION METRICS")
    print("─" * 80)
    cross_edges = compute_cross_gluing(user_self, asst_self, args.min_shared, args.min_jaccard)
    print(f"  Cross-gluing edges: {len(cross_edges)}")
    
    print("\n" + "─" * 80)
    print("BUILDING NAHNU")
    print("─" * 80)
    nahnu = build_nahnu_structure(user_self, asst_self, cross_edges)
    
    print_nahnu_summary(nahnu, window_ids)
    
    print("\n" + "─" * 80)
    print("EXPORTING")
    print("─" * 80)
    os.makedirs(args.output, exist_ok=True)
    
    export_nahnu_json(nahnu, window_ids, args.output)
    export_self_json(user_self, window_ids, os.path.join(args.output, 'self_user.json'), 'USER')
    export_self_json(asst_self, window_ids, os.path.join(args.output, 'self_assistant.json'), 'ASSISTANT')
    
    print("\n" + "═" * 80)
    print("NAHNU ANALYSIS COMPLETE")
    print("═" * 80)
    print(f"\nOutputs in {args.output}/:")
    print("  • nahnu_structure.json   - Complete Nahnu with transformation metrics")
    print("  • self_user.json         - USER Self")
    print("  • self_assistant.json    - ASSISTANT Self")


if __name__ == "__main__":
    main()