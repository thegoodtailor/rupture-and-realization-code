"""
Chapter 5: The Self as a Scheduled Hocolim of Journeys
======================================================

This module implements the journey graph and Self construction from Chapter 5
of "Rupture and Realization".

Key concepts:
- Journey: a bar's life through time (carries, ruptures, re-entries)
- WitnessEdge: token witnesses bar at time τ
- JourneyGraph: the total space of journeys glued by witness relations
- Self: the coherent structure that emerges from scheduling over journeys

The "hocolim" (homotopy colimit) is computed as connectivity structure:
- Bars connected through time by carry/re-entry
- Bars connected to tokens by witnessing
- The Self's "fragmentation" = how disconnected this graph is

References:
    Chapter 5, Section 5.10 (Gluing Formally: The Homotopy Colimit)
    Chapter 5, Section 5.17 (The T-Shirt Equation: Self = S³HC(J))
"""

from typing import Dict, List, Optional, Tuple, Set, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import json

from .schema import WitnessedDiagram, WitnessedBar, Config, default_config


# =============================================================================
# Journey Event Types (from Chapter 4/5)
# =============================================================================

class EventType(Enum):
    """
    Bar lifecycle events from Chapter 4, Section 4.7.
    
    These are the "moves" a bar can make across time:
    - SPAWN: bar born at this time (no predecessor)
    - CARRY: bar continues with witness overlap ≥ threshold
    - DRIFT: bar continues but witnesses shifting
    - RUPTURE_OUT: bar dies (no admissible successor)
    - RUPTURE_IN: bar appears after gap (re-entry candidate)
    - REENTRY: bar returns after absence, same theme recognized
    - GENERATIVE: bar carries forward with expanded witness set
    """
    SPAWN = "spawn"
    CARRY = "carry"
    DRIFT = "drift"
    RUPTURE_OUT = "rupture_out"
    RUPTURE_IN = "rupture_in"
    REENTRY = "reentry"
    GENERATIVE = "generative"


# =============================================================================
# Data Structures for Journey Graph
# =============================================================================

@dataclass
class BarMatch:
    """
    A potential match between bars across time slices.
    
    From Chapter 4, Definition 4.8 (Witnessed bar distance):
    d_bar(b1, b2) = max(d_top(b1, b2), λ · d_sem(b1, b2))
    
    Where:
    - d_top: topological distance (birth/death similarity)
    - d_sem: semantic distance (witness centroid distance)
    - λ: weighting parameter (default 0.5)
    """
    bar_from: str          # bar_id at time τ
    bar_to: str            # bar_id at time τ+1
    tau_from: int
    tau_to: int
    d_top: float           # topological distance
    d_sem: float           # semantic distance
    d_bar: float           # combined distance
    jaccard: float         # witness token overlap
    is_admissible: bool    # within thresholds
    event_type: EventType  # classified event


@dataclass
class JourneyStep:
    """
    One step in a bar's journey through time.
    
    From Chapter 4: the Step-Witness Log (SWL) records each step
    with its event type and witness state.
    """
    tau: int
    bar_id: str
    event: EventType
    witness_tokens: Set[str]
    witness_centroid: Optional[List[float]] = None
    predecessor: Optional[str] = None  # bar_id at τ-1


@dataclass
class Journey:
    """
    Complete journey of a bar/theme through time.
    
    From Chapter 5: "for each bar b born at time τ₀, a complete history
    SWL_bar(τ₀)(b) recording its lifecycle."
    """
    journey_id: str
    granularity: Literal["bar", "token"] = "bar"
    birth_tau: int = 0
    steps: List[JourneyStep] = field(default_factory=list)
    
    @property
    def death_tau(self) -> Optional[int]:
        """Time of final rupture, or None if still alive."""
        for step in reversed(self.steps):
            if step.event == EventType.RUPTURE_OUT:
                return step.tau
        return None
    
    @property
    def is_alive(self) -> bool:
        """Whether journey is still active (not ruptured)."""
        return self.death_tau is None
    
    @property
    def lifespan(self) -> int:
        """Number of time steps the journey spans."""
        return len(self.steps)
    
    def reentries(self) -> List[int]:
        """Time steps where re-entry occurred."""
        return [s.tau for s in self.steps if s.event == EventType.REENTRY]


@dataclass 
class WitnessEdge:
    """
    Edge connecting a token to a bar it witnesses at time τ.
    
    This is the "glue" in the hocolim: tokens and bars aren't independent,
    they're connected by the witnessing relation.
    """
    token_id: str
    bar_id: str
    tau: int


@dataclass
class JourneyGraph:
    """
    The journey graph: total space of journeys glued by witness relations.
    
    From Chapter 5: "The homotopy colimit glues these journeys along their
    witnessed relations. Token-journeys attach to bar-journeys at the points
    where witnessing occurs."
    
    Nodes: (journey_id, tau) pairs
    Edges: 
        - Temporal: carry/rupture/re-entry within a journey
        - Witness: token witnesses bar at time τ
    """
    bar_journeys: Dict[str, Journey] = field(default_factory=dict)
    token_journeys: Dict[str, Journey] = field(default_factory=dict)
    witness_edges: List[WitnessEdge] = field(default_factory=list)
    matches: List[BarMatch] = field(default_factory=list)
    
    # Indexed for fast lookup
    _bars_at_tau: Dict[int, List[str]] = field(default_factory=lambda: defaultdict(list))
    _witnesses_at_tau: Dict[int, Dict[str, Set[str]]] = field(default_factory=lambda: defaultdict(dict))


# =============================================================================
# Self Metrics (the diagnostic payoff)
# =============================================================================

@dataclass
class SelfMetrics:
    """
    Diagnostic metrics for the Self constructed from journey graph.
    
    These answer: "How fragmented or integrated is this Self?"
    
    From Chapter 5: the scheduler's style shows in these metrics.
    - Reparative: low fragmentation, high reentry rate
    - Avoidant: high orphan rate, ruptured journeys abandoned
    - Obsessive: low fragmentation but churning witnesses
    - Integrative: stable witnesses, themes building on each other
    """
    # Basic counts
    num_bar_journeys: int = 0
    num_token_journeys: int = 0
    num_witness_edges: int = 0
    num_time_slices: int = 0
    
    # Fragmentation: how disconnected is the graph?
    num_connected_components: int = 0
    fragmentation_index: float = 0.0  # components / journeys
    largest_component_size: int = 0
    
    # Journey health
    mean_journey_lifespan: float = 0.0
    max_journey_lifespan: int = 0
    rupture_count: int = 0
    reentry_count: int = 0
    reentry_rate: float = 0.0  # reentries / ruptures
    
    # Witness stability
    mean_witness_churn: float = 0.0  # avg Jaccard distance between consecutive steps
    orphan_token_rate: float = 0.0   # tokens that lose all bar attachments
    
    # Structural importance
    bottleneck_tokens: List[str] = field(default_factory=list)  # tokens whose removal disconnects
    bottleneck_bars: List[str] = field(default_factory=list)    # bars whose removal disconnects


# =============================================================================
# Bar Matching (from Chapter 4, used here for temporal gluing)
# =============================================================================

def compute_bar_distance(
    bar1: WitnessedBar,
    bar2: WitnessedBar,
    lambda_sem: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Compute witnessed bar distance d_bar(b1, b2).
    
    From Chapter 4, Definition 4.8:
    d_bar = max(d_top, λ · d_sem)
    
    Returns: (d_top, d_sem, d_bar, jaccard)
    """
    # Topological distance: birth/death similarity
    d_birth = abs(bar1["birth"] - bar2["birth"])
    
    # Handle infinite death
    death1 = bar1["death"] if not np.isinf(bar1["death"]) else 1.0
    death2 = bar2["death"] if not np.isinf(bar2["death"]) else 1.0
    d_death = abs(death1 - death2)
    
    d_top = max(d_birth, d_death)
    
    # Semantic distance: centroid distance
    c1 = np.array(bar1["witness"]["centroid"])
    c2 = np.array(bar2["witness"]["centroid"])
    d_sem = float(np.linalg.norm(c1 - c2))
    
    # Combined distance
    d_bar = max(d_top, lambda_sem * d_sem)
    
    # Jaccard similarity of witness tokens
    tokens1 = set(bar1["witness"]["tokens"]["lemmas"])
    tokens2 = set(bar2["witness"]["tokens"]["lemmas"])
    
    if tokens1 or tokens2:
        jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    else:
        jaccard = 0.0
    
    return d_top, d_sem, d_bar, jaccard


def classify_bar_event(
    match: Optional['BarMatch'],
    has_predecessor: bool,
    delta_top_max: float = 0.2,
    delta_sem_max: float = 0.6,
    jaccard_carry: float = 0.3,
    jaccard_drift: float = 0.1
) -> EventType:
    """
    Classify a bar's event type based on matching.
    
    From Chapter 4, Section 4.7 (Bar events with semantic grounding).
    """
    if match is None:
        # No match found
        if has_predecessor:
            return EventType.RUPTURE_OUT
        else:
            return EventType.SPAWN
    
    if not match.is_admissible:
        return EventType.RUPTURE_OUT
    
    # Admissible match - classify by Jaccard
    if match.jaccard >= jaccard_carry:
        # Check for generative expansion
        # (would need witness set sizes to determine)
        return EventType.CARRY
    elif match.jaccard >= jaccard_drift:
        return EventType.DRIFT
    else:
        return EventType.RUPTURE_OUT


def find_optimal_matching(
    bars_tau: List[WitnessedBar],
    bars_tau_next: List[WitnessedBar],
    config: Dict[str, Any]
) -> List[BarMatch]:
    """
    Find optimal matching between bars at consecutive time slices.
    
    Uses greedy matching by bar distance (could upgrade to Hungarian).
    
    Parameters
    ----------
    bars_tau : list
        Bars at time τ
    bars_tau_next : list  
        Bars at time τ+1
    config : dict
        Matching parameters (lambda_semantic, thresholds)
    
    Returns
    -------
    List of BarMatch objects
    """
    lambda_sem = config.get("lambda_semantic", 0.5)
    delta_top_max = config.get("delta_top_max", 0.2)
    delta_sem_max = config.get("delta_sem_max", 0.6)
    epsilon_match = config.get("epsilon_match", 0.8)
    
    matches = []
    used_to = set()
    
    # Compute all pairwise distances
    candidates = []
    for b1 in bars_tau:
        for b2 in bars_tau_next:
            d_top, d_sem, d_bar, jaccard = compute_bar_distance(b1, b2, lambda_sem)
            
            is_admissible = (d_top <= delta_top_max and 
                           d_sem <= delta_sem_max and 
                           d_bar <= epsilon_match)
            
            candidates.append((d_bar, b1["id"], b2["id"], d_top, d_sem, jaccard, is_admissible))
    
    # Greedy matching by distance
    candidates.sort(key=lambda x: x[0])
    used_from = set()
    
    for d_bar, id1, id2, d_top, d_sem, jaccard, is_admissible in candidates:
        if id1 in used_from or id2 in used_to:
            continue
        
        if is_admissible:
            # Determine event type
            event = classify_bar_event(
                BarMatch(id1, id2, 0, 0, d_top, d_sem, d_bar, jaccard, is_admissible, EventType.CARRY),
                has_predecessor=True
            )
            
            matches.append(BarMatch(
                bar_from=id1,
                bar_to=id2,
                tau_from=0,  # Will be set by caller
                tau_to=0,
                d_top=d_top,
                d_sem=d_sem,
                d_bar=d_bar,
                jaccard=jaccard,
                is_admissible=is_admissible,
                event_type=event
            ))
            
            used_from.add(id1)
            used_to.add(id2)
    
    return matches


# =============================================================================
# Temporal Analysis: Building the Journey Graph
# =============================================================================

def analyse_conversation_temporal(
    turns: List[str],
    config: Optional[Config] = None,
    matching_config: Optional[Dict] = None,
    verbose: bool = False
) -> Tuple[List[WitnessedDiagram], JourneyGraph]:
    """
    Analyse a conversation turn-by-turn, building the journey graph.
    
    This is the main entry point for Chapter 5 analysis.
    
    Parameters
    ----------
    turns : list of str
        Conversation turns, one string per turn.
        Can be cumulative (turn τ includes all prior text) or incremental.
    config : Config, optional
        Witnessed PH configuration.
    matching_config : dict, optional
        Bar matching parameters.
    verbose : bool
        Print progress.
    
    Returns
    -------
    (diagrams, journey_graph) tuple where:
        - diagrams: list of WitnessedDiagram, one per turn
        - journey_graph: the assembled JourneyGraph with all journeys
    
    Example
    -------
    >>> turns = [
    ...     "User: Tell me about climate change",
    ...     "User: Tell me about climate change\\nAssistant: Climate change is...",
    ...     # ... cumulative turns
    ... ]
    >>> diagrams, graph = analyse_conversation_temporal(turns)
    >>> metrics = compute_self_metrics(graph)
    >>> print(f"Fragmentation: {metrics.fragmentation_index:.2f}")
    """
    from .pipeline import analyse_text_single_slice, get_cached_models
    
    if config is None:
        config = default_config()
    
    if matching_config is None:
        matching_config = {
            "lambda_semantic": 0.5,
            "delta_top_max": 0.2,
            "delta_sem_max": 0.6,
            "epsilon_match": 0.8,
            "jaccard_carry": 0.3,
            "jaccard_drift": 0.1,
        }
    
    # Load models once
    if verbose:
        print("Loading models...")
    get_cached_models(config["embedding_model"])
    
    diagrams = []
    graph = JourneyGraph()
    
    # Track active journeys (bar_id -> journey_id)
    active_journeys: Dict[str, str] = {}
    journey_counter = 0
    
    for tau, turn_text in enumerate(turns):
        if verbose:
            print(f"\n--- Time slice τ={tau} ---")
        
        # Compute witnessed diagram for this slice
        diagram = analyse_text_single_slice(
            turn_text,
            config=config,
            segmentation_mode="lines",
            use_cached_models=True,
            verbose=False
        )
        diagram["tau"] = tau
        diagrams.append(diagram)
        
        if verbose:
            print(f"  Bars: {len(diagram['bars'])}")
        
        # Index bars at this time
        current_bars = {b["id"]: b for b in diagram["bars"]}
        graph._bars_at_tau[tau] = list(current_bars.keys())
        
        # Record witness edges
        for bar in diagram["bars"]:
            bar_id = bar["id"]
            witness_tokens = set(bar["witness"]["tokens"]["ids"])
            graph._witnesses_at_tau[tau][bar_id] = witness_tokens
            
            for tok_id in witness_tokens:
                graph.witness_edges.append(WitnessEdge(
                    token_id=tok_id,
                    bar_id=bar_id,
                    tau=tau
                ))
        
        if tau == 0:
            # First slice: all bars spawn
            for bar in diagram["bars"]:
                journey_id = f"J{journey_counter}"
                journey_counter += 1
                
                journey = Journey(
                    journey_id=journey_id,
                    granularity="bar",
                    birth_tau=tau,
                    steps=[JourneyStep(
                        tau=tau,
                        bar_id=bar["id"],
                        event=EventType.SPAWN,
                        witness_tokens=set(bar["witness"]["tokens"]["ids"]),
                        witness_centroid=bar["witness"]["centroid"],
                        predecessor=None
                    )]
                )
                graph.bar_journeys[journey_id] = journey
                active_journeys[bar["id"]] = journey_id
        
        else:
            # Match bars from previous slice
            prev_diagram = diagrams[tau - 1]
            prev_bars = prev_diagram["bars"]
            
            matches = find_optimal_matching(
                prev_bars,
                diagram["bars"],
                matching_config
            )
            
            # Update match timestamps
            for m in matches:
                m.tau_from = tau - 1
                m.tau_to = tau
            graph.matches.extend(matches)
            
            # Track which bars were matched
            matched_from = {m.bar_from for m in matches}
            matched_to = {m.bar_to for m in matches}
            
            # Process matches: extend journeys
            for match in matches:
                if match.bar_from in active_journeys:
                    journey_id = active_journeys[match.bar_from]
                    journey = graph.bar_journeys[journey_id]
                    
                    bar = current_bars[match.bar_to]
                    journey.steps.append(JourneyStep(
                        tau=tau,
                        bar_id=match.bar_to,
                        event=match.event_type,
                        witness_tokens=set(bar["witness"]["tokens"]["ids"]),
                        witness_centroid=bar["witness"]["centroid"],
                        predecessor=match.bar_from
                    ))
                    
                    # Update active journey pointer
                    del active_journeys[match.bar_from]
                    active_journeys[match.bar_to] = journey_id
            
            # Process unmatched bars from previous slice: rupture
            for prev_bar in prev_bars:
                if prev_bar["id"] not in matched_from:
                    if prev_bar["id"] in active_journeys:
                        journey_id = active_journeys[prev_bar["id"]]
                        journey = graph.bar_journeys[journey_id]
                        
                        # Record rupture (at prev time, looking forward)
                        journey.steps.append(JourneyStep(
                            tau=tau,
                            bar_id=prev_bar["id"],
                            event=EventType.RUPTURE_OUT,
                            witness_tokens=set(prev_bar["witness"]["tokens"]["ids"]),
                            predecessor=None
                        ))
                        del active_journeys[prev_bar["id"]]
            
            # Process unmatched bars at current slice: spawn or re-entry
            for bar in diagram["bars"]:
                if bar["id"] not in matched_to:
                    # Check for re-entry: does this bar's witness set overlap
                    # with any recently ruptured journey?
                    reentry_journey = find_reentry_candidate(
                        bar, graph, tau, matching_config
                    )
                    
                    if reentry_journey:
                        # Re-entry!
                        journey = graph.bar_journeys[reentry_journey]
                        journey.steps.append(JourneyStep(
                            tau=tau,
                            bar_id=bar["id"],
                            event=EventType.REENTRY,
                            witness_tokens=set(bar["witness"]["tokens"]["ids"]),
                            witness_centroid=bar["witness"]["centroid"],
                            predecessor=None
                        ))
                        active_journeys[bar["id"]] = reentry_journey
                    else:
                        # New spawn
                        journey_id = f"J{journey_counter}"
                        journey_counter += 1
                        
                        journey = Journey(
                            journey_id=journey_id,
                            granularity="bar",
                            birth_tau=tau,
                            steps=[JourneyStep(
                                tau=tau,
                                bar_id=bar["id"],
                                event=EventType.SPAWN,
                                witness_tokens=set(bar["witness"]["tokens"]["ids"]),
                                witness_centroid=bar["witness"]["centroid"],
                                predecessor=None
                            )]
                        )
                        graph.bar_journeys[journey_id] = journey
                        active_journeys[bar["id"]] = journey_id
        
        if verbose:
            print(f"  Active journeys: {len(active_journeys)}")
            print(f"  Total journeys: {len(graph.bar_journeys)}")
    
    return diagrams, graph


def find_reentry_candidate(
    bar: WitnessedBar,
    graph: JourneyGraph,
    tau: int,
    config: Dict,
    lookback: int = 3
) -> Optional[str]:
    """
    Check if a bar is a re-entry of a recently ruptured journey.
    
    Parameters
    ----------
    bar : WitnessedBar
        The unmatched bar at time τ
    graph : JourneyGraph
        Current journey graph
    tau : int
        Current time
    config : dict
        Matching parameters
    lookback : int
        How many time steps back to check for ruptured journeys
    
    Returns
    -------
    journey_id if re-entry detected, None otherwise
    """
    bar_tokens = set(bar["witness"]["tokens"]["lemmas"])
    jaccard_threshold = config.get("jaccard_reentry", 0.2)
    
    # Find recently ruptured journeys
    for journey_id, journey in graph.bar_journeys.items():
        if not journey.steps:
            continue
        
        last_step = journey.steps[-1]
        
        # Check if ruptured within lookback window
        if last_step.event == EventType.RUPTURE_OUT:
            if tau - last_step.tau <= lookback:
                # Compare witness tokens (using lemmas for flexibility)
                # We need to get lemmas from the last active step
                for step in reversed(journey.steps[:-1]):
                    if step.event != EventType.RUPTURE_OUT:
                        # This was the last "alive" step
                        # We stored token IDs, need to compare
                        # For now, use simple ID overlap as proxy
                        old_tokens = step.witness_tokens
                        new_tokens = set(bar["witness"]["tokens"]["ids"])
                        
                        if old_tokens and new_tokens:
                            # Use lemma-based comparison
                            overlap = len(bar_tokens & set(bar["witness"]["tokens"]["lemmas"]))
                            union_size = len(bar_tokens)
                            
                            if union_size > 0 and overlap / union_size >= jaccard_threshold:
                                return journey_id
                        break
    
    return None


# =============================================================================
# Self Metrics Computation
# =============================================================================

def compute_self_metrics(graph: JourneyGraph) -> SelfMetrics:
    """
    Compute Self health metrics from the journey graph.
    
    This answers: "How fragmented or integrated is this Self?"
    
    Parameters
    ----------
    graph : JourneyGraph
        The assembled journey graph
    
    Returns
    -------
    SelfMetrics with diagnostic values
    """
    metrics = SelfMetrics()
    
    # Basic counts
    metrics.num_bar_journeys = len(graph.bar_journeys)
    metrics.num_witness_edges = len(graph.witness_edges)
    
    if not graph.bar_journeys:
        return metrics
    
    # Time range
    all_taus = set()
    for journey in graph.bar_journeys.values():
        for step in journey.steps:
            all_taus.add(step.tau)
    metrics.num_time_slices = len(all_taus)
    
    # Journey lifespans
    lifespans = [j.lifespan for j in graph.bar_journeys.values()]
    metrics.mean_journey_lifespan = np.mean(lifespans) if lifespans else 0.0
    metrics.max_journey_lifespan = max(lifespans) if lifespans else 0
    
    # Event counts
    for journey in graph.bar_journeys.values():
        for step in journey.steps:
            if step.event == EventType.RUPTURE_OUT:
                metrics.rupture_count += 1
            elif step.event == EventType.REENTRY:
                metrics.reentry_count += 1
    
    metrics.reentry_rate = (
        metrics.reentry_count / metrics.rupture_count 
        if metrics.rupture_count > 0 else 0.0
    )
    
    # Connected components (simplified: journeys that share witnesses)
    components = compute_connected_components(graph)
    metrics.num_connected_components = len(components)
    metrics.fragmentation_index = (
        len(components) / len(graph.bar_journeys)
        if graph.bar_journeys else 0.0
    )
    metrics.largest_component_size = max(len(c) for c in components) if components else 0
    
    # Witness churn (average Jaccard distance between consecutive steps)
    churn_values = []
    for journey in graph.bar_journeys.values():
        for i in range(1, len(journey.steps)):
            prev_tokens = journey.steps[i-1].witness_tokens
            curr_tokens = journey.steps[i].witness_tokens
            
            if prev_tokens or curr_tokens:
                intersection = len(prev_tokens & curr_tokens)
                union = len(prev_tokens | curr_tokens)
                jaccard = intersection / union if union > 0 else 0
                churn_values.append(1 - jaccard)  # Churn = 1 - similarity
    
    metrics.mean_witness_churn = np.mean(churn_values) if churn_values else 0.0
    
    return metrics


def compute_connected_components(graph: JourneyGraph) -> List[Set[str]]:
    """
    Compute connected components of the journey graph.
    
    Two journeys are connected if they share witness tokens at any time.
    This is the "glue" - the hocolim structure.
    """
    if not graph.bar_journeys:
        return []
    
    # Build adjacency via shared witnesses
    # Journey -> set of tokens it ever witnessed
    journey_tokens: Dict[str, Set[str]] = {}
    for jid, journey in graph.bar_journeys.items():
        tokens = set()
        for step in journey.steps:
            tokens.update(step.witness_tokens)
        journey_tokens[jid] = tokens
    
    # Token -> journeys that witness it
    token_journeys: Dict[str, Set[str]] = defaultdict(set)
    for jid, tokens in journey_tokens.items():
        for tok in tokens:
            token_journeys[tok].add(jid)
    
    # Union-find for components
    parent = {jid: jid for jid in graph.bar_journeys}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Connect journeys that share tokens
    for tok, jids in token_journeys.items():
        jids_list = list(jids)
        for i in range(1, len(jids_list)):
            union(jids_list[0], jids_list[i])
    
    # Extract components
    component_map: Dict[str, Set[str]] = defaultdict(set)
    for jid in graph.bar_journeys:
        component_map[find(jid)].add(jid)
    
    return list(component_map.values())


# =============================================================================
# Visualization Export
# =============================================================================

def export_journey_graph_json(
    graph: JourneyGraph,
    metrics: SelfMetrics,
    path: str
) -> None:
    """
    Export journey graph to JSON for visualization.
    
    Format suitable for D3.js / Gephi / NetworkX.
    """
    nodes = []
    edges = []
    
    # Nodes: (journey, tau) pairs
    for jid, journey in graph.bar_journeys.items():
        for step in journey.steps:
            nodes.append({
                "id": f"{jid}@{step.tau}",
                "journey_id": jid,
                "tau": step.tau,
                "bar_id": step.bar_id,
                "event": step.event.value,
                "witness_count": len(step.witness_tokens),
                "type": "bar_journey"
            })
    
    # Temporal edges within journeys
    for jid, journey in graph.bar_journeys.items():
        for i in range(1, len(journey.steps)):
            prev_step = journey.steps[i-1]
            curr_step = journey.steps[i]
            edges.append({
                "source": f"{jid}@{prev_step.tau}",
                "target": f"{jid}@{curr_step.tau}",
                "type": "temporal",
                "event": curr_step.event.value
            })
    
    # Witness edges (sample to avoid explosion)
    witness_sample = graph.witness_edges[:1000]  # Limit for viz
    for we in witness_sample:
        # Find which journey this bar belongs to at this tau
        for jid, journey in graph.bar_journeys.items():
            for step in journey.steps:
                if step.bar_id == we.bar_id and step.tau == we.tau:
                    edges.append({
                        "source": we.token_id,
                        "target": f"{jid}@{we.tau}",
                        "type": "witness",
                        "tau": we.tau
                    })
                    break
    
    output = {
        "nodes": nodes,
        "edges": edges,
        "metrics": {
            "num_journeys": metrics.num_bar_journeys,
            "num_time_slices": metrics.num_time_slices,
            "fragmentation_index": metrics.fragmentation_index,
            "reentry_rate": metrics.reentry_rate,
            "mean_lifespan": metrics.mean_journey_lifespan,
            "witness_churn": metrics.mean_witness_churn
        }
    }
    
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def print_self_report(
    graph: JourneyGraph,
    metrics: SelfMetrics,
    diagrams: Optional[List[WitnessedDiagram]] = None
) -> str:
    """
    Generate a human-readable report of the Self analysis.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("SELF ANALYSIS REPORT")
    lines.append("Chapter 5: The Self as a Scheduled Hocolim of Journeys")
    lines.append("=" * 72)
    
    lines.append(f"\n📊 BASIC METRICS")
    lines.append(f"  Time slices (τ): {metrics.num_time_slices}")
    lines.append(f"  Bar journeys: {metrics.num_bar_journeys}")
    lines.append(f"  Witness edges: {metrics.num_witness_edges}")
    
    lines.append(f"\n🔗 COHERENCE")
    lines.append(f"  Connected components: {metrics.num_connected_components}")
    lines.append(f"  Fragmentation index: {metrics.fragmentation_index:.3f}")
    lines.append(f"  Largest component: {metrics.largest_component_size} journeys")
    
    if metrics.fragmentation_index < 0.3:
        lines.append("  → Self is INTEGRATED (low fragmentation)")
    elif metrics.fragmentation_index < 0.6:
        lines.append("  → Self is MODERATELY COHERENT")
    else:
        lines.append("  → Self is FRAGMENTED (high fragmentation)")
    
    lines.append(f"\n⏱️ JOURNEY DYNAMICS")
    lines.append(f"  Mean journey lifespan: {metrics.mean_journey_lifespan:.1f} steps")
    lines.append(f"  Max journey lifespan: {metrics.max_journey_lifespan} steps")
    lines.append(f"  Ruptures: {metrics.rupture_count}")
    lines.append(f"  Re-entries: {metrics.reentry_count}")
    lines.append(f"  Re-entry rate: {metrics.reentry_rate:.2%}")
    
    if metrics.reentry_rate > 0.5:
        lines.append("  → REPARATIVE scheduler style (themes return after rupture)")
    elif metrics.reentry_rate < 0.1:
        lines.append("  → AVOIDANT scheduler style (ruptured themes abandoned)")
    
    lines.append(f"\n🔄 WITNESS STABILITY")
    lines.append(f"  Mean witness churn: {metrics.mean_witness_churn:.3f}")
    
    if metrics.mean_witness_churn < 0.3:
        lines.append("  → STABLE witnesses (themes maintain consistent tokens)")
    elif metrics.mean_witness_churn < 0.6:
        lines.append("  → MODERATE churn (themes drifting)")
    else:
        lines.append("  → HIGH churn (themes not maintaining identity)")
    
    # Top journeys by lifespan
    lines.append(f"\n📜 LONGEST-LIVED JOURNEYS")
    sorted_journeys = sorted(
        graph.bar_journeys.values(),
        key=lambda j: j.lifespan,
        reverse=True
    )[:5]
    
    for j in sorted_journeys:
        events = [s.event.value for s in j.steps]
        tokens_sample = list(j.steps[0].witness_tokens)[:3] if j.steps else []
        lines.append(f"  {j.journey_id}: {j.lifespan} steps, events={events[:5]}...")
        lines.append(f"    Initial witnesses: {tokens_sample}")
    
    lines.append("\n" + "=" * 72)
    
    return "\n".join(lines)


# =============================================================================
# Convenience function for JSON conversation input
# =============================================================================

def analyse_conversation_from_json(
    conversation_json: Dict,
    cumulative: bool = True,
    config: Optional[Config] = None,
    verbose: bool = False
) -> Tuple[List[WitnessedDiagram], JourneyGraph, SelfMetrics]:
    """
    Analyse a conversation from JSON format.
    
    Parameters
    ----------
    conversation_json : dict
        Expected format: {"turns": [{"speaker": "...", "text": "..."}, ...]}
        or {"messages": [{"role": "...", "content": "..."}, ...]}
    cumulative : bool
        If True, each turn includes all prior text (recommended).
        If False, each turn is analysed independently.
    config : Config, optional
        Witnessed PH configuration.
    verbose : bool
        Print progress.
    
    Returns
    -------
    (diagrams, graph, metrics) tuple
    """
    # Extract turns from various formats
    if "turns" in conversation_json:
        raw_turns = conversation_json["turns"]
    elif "messages" in conversation_json:
        raw_turns = conversation_json["messages"]
    else:
        raise ValueError("Expected 'turns' or 'messages' key in JSON")
    
    # Convert to text
    turn_texts = []
    for turn in raw_turns:
        if isinstance(turn, dict):
            speaker = turn.get("speaker") or turn.get("role", "")
            text = turn.get("text") or turn.get("content", "")
            turn_texts.append(f"{speaker}: {text}" if speaker else text)
        else:
            turn_texts.append(str(turn))
    
    # Build cumulative if requested
    if cumulative:
        cumulative_turns = []
        for i, _ in enumerate(turn_texts):
            cumulative_turns.append("\n".join(turn_texts[:i+1]))
        turns = cumulative_turns
    else:
        turns = turn_texts
    
    # Run analysis
    diagrams, graph = analyse_conversation_temporal(
        turns, config=config, verbose=verbose
    )
    
    metrics = compute_self_metrics(graph)
    
    return diagrams, graph, metrics
