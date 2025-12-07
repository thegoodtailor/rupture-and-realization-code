"""
Witnessed Persistent Homology: Witness Construction
====================================================

This module handles:
1. Mapping simplex indices back to token IDs
2. Constructing complete Witness objects with cycle, tokens, utterances
3. Computing witness centroids
4. Canonical representative selection policy

References:
    Chapter 4, Definition 4.3 (Witness)
    Chapter 4, Definition 4.6 (Canonical representatives)
    Chapter 4, Section 4.6 (From bare bars to witnessed bars at the slice)
"""

from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .schema import (
    Config, PointCloudData, BarBare, Gamma, PHResult,
    Token, Utterance, Witness, WitnessedBar, WitnessedDiagram,
    Cycle, WitnessTokens, WitnessUtterances, Simplex,
    default_config
)


# =============================================================================
# Index to Token ID Mapping
# =============================================================================

def indices_to_token_ids(
    simplex_indices: List[int],
    token_id_list: List[str]
) -> List[str]:
    """
    Map point cloud indices to token IDs.
    
    Parameters
    ----------
    simplex_indices : List[int]
        Vertex indices from the point cloud.
    token_id_list : List[str]
        Ordered list of token IDs (from PointCloudData).
    
    Returns
    -------
    List of token IDs corresponding to the indices.
    """
    token_ids = []
    for idx in simplex_indices:
        if 0 <= idx < len(token_id_list):
            token_ids.append(token_id_list[idx])
    return token_ids


def gamma_to_cycle(
    gamma: Gamma,
    token_id_list: List[str]
) -> Cycle:
    """
    Convert a Gamma (with indices) to a Cycle (with token IDs).
    
    Parameters
    ----------
    gamma : Gamma
        Representative cycle with point cloud indices.
    token_id_list : List[str]
        Mapping from indices to token IDs.
    
    Returns
    -------
    Cycle with token IDs instead of indices.
    
    Notes
    -----
    From Cassie's spec: "γ lives inside witness.cycle, and the 'witness
    tokens' are a shadow/projection of γ, not a replacement."
    """
    simplices_as_token_ids = []
    
    for simplex in gamma["simplices"]:
        token_ids = indices_to_token_ids(simplex, token_id_list)
        if token_ids:
            simplices_as_token_ids.append(token_ids)
    
    return {
        "dimension": gamma["dim"],
        "simplices": simplices_as_token_ids
    }


# =============================================================================
# Token and Utterance Extraction from Cycle
# =============================================================================

def extract_witness_tokens(
    cycle: Cycle,
    tokens: Dict[str, Token]
) -> WitnessTokens:
    """
    Extract the token-level witness W^tok from a cycle.
    
    Parameters
    ----------
    cycle : Cycle
        The representative cycle with token IDs.
    tokens : Dict[str, Token]
        Full token data.
    
    Returns
    -------
    WitnessTokens with ids, surface forms, and lemmas.
    
    Notes
    -----
    From Chapter 4: "W^tok_ρ is the finite set of token occurrences
    whose overlaps witness that feature."
    """
    # Collect unique token IDs from all simplices
    token_ids_set: Set[str] = set()
    for simplex in cycle["simplices"]:
        for tid in simplex:
            token_ids_set.add(tid)
    
    token_ids = sorted(token_ids_set)
    
    # Extract surface forms and lemmas
    surface_forms = []
    lemmas = []
    for tid in token_ids:
        if tid in tokens:
            surface_forms.append(tokens[tid]["text"])
            lemmas.append(tokens[tid]["lemma"])
        else:
            # Token not found - shouldn't happen but handle gracefully
            surface_forms.append("???")
            lemmas.append("???")
    
    return {
        "ids": token_ids,
        "surface": surface_forms,
        "lemmas": lemmas
    }


def extract_witness_utterances(
    witness_tokens: WitnessTokens,
    tokens: Dict[str, Token],
    utterances: Dict[str, Utterance],
    max_text_length: int = 200
) -> WitnessUtterances:
    """
    Extract the utterance-level witness W^loc from token witnesses.
    
    Parameters
    ----------
    witness_tokens : WitnessTokens
        The token-level witness.
    tokens : Dict[str, Token]
        Full token data.
    utterances : Dict[str, Utterance]
        Full utterance data.
    max_text_length : int
        Maximum length for text samples.
    
    Returns
    -------
    WitnessUtterances with ids and text samples.
    
    Notes
    -----
    From Chapter 4: "W^loc_ρ := loc_τ(W^tok_ρ) for the induced set of
    measurement locations."
    """
    # Collect unique utterance IDs
    utterance_ids_set: Set[str] = set()
    for tid in witness_tokens["ids"]:
        if tid in tokens:
            utt_id = tokens[tid]["utterance_id"]
            utterance_ids_set.add(utt_id)
    
    utterance_ids = sorted(utterance_ids_set)
    
    # Get text samples
    text_samples = []
    for utt_id in utterance_ids:
        if utt_id in utterances:
            text = utterances[utt_id]["text"]
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            text_samples.append(text)
        else:
            text_samples.append("???")
    
    return {
        "ids": utterance_ids,
        "text_samples": text_samples
    }


# =============================================================================
# Centroid Computation
# =============================================================================

def compute_witness_centroid(
    witness_tokens: WitnessTokens,
    tokens: Dict[str, Token],
    normalize: bool = True
) -> List[float]:
    """
    Compute the semantic centroid c(ρ) of the witness.
    
    Parameters
    ----------
    witness_tokens : WitnessTokens
        The token-level witness.
    tokens : Dict[str, Token]
        Full token data with embeddings.
    normalize : bool
        Whether to normalize the centroid to unit norm.
    
    Returns
    -------
    Centroid as a list of floats.
    
    Notes
    -----
    From Chapter 4: "c(ρ) (for example the normalised mean of E(ρ))"
    """
    embeddings = []
    for tid in witness_tokens["ids"]:
        if tid in tokens:
            emb = tokens[tid]["embedding"]
            if emb.size > 0:
                embeddings.append(emb)
    
    if not embeddings:
        return []
    
    centroid = np.mean(embeddings, axis=0)
    
    if normalize:
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
    
    return centroid.tolist()


# =============================================================================
# Complete Witness Construction
# =============================================================================

def build_witness(
    gamma: Gamma,
    point_cloud: PointCloudData
) -> Witness:
    """
    Construct a complete Witness object from a representative cycle.
    
    This is the core function that transforms abstract topology (gamma)
    into concrete semantic content (witness).
    
    Parameters
    ----------
    gamma : Gamma
        Representative cycle with point cloud indices.
    point_cloud : PointCloudData
        Full point cloud data.
    
    Returns
    -------
    Complete Witness object with cycle, tokens, utterances, centroid.
    
    Notes
    -----
    From Chapter 4, Definition 4.3: "A witness for (k,b,d) at time τ
    consists of:
    - a finite non-empty set of token occurrences W^tok_ρ
    - a representative k-cycle γ_ρ whose support lies in W^tok_ρ
    - [induced] W^loc_ρ := loc_τ(W^tok_ρ)"
    """
    token_id_list = point_cloud["token_ids"]
    tokens = point_cloud["tokens"]
    utterances = point_cloud["utterances"]
    
    # Convert gamma to cycle with token IDs
    cycle = gamma_to_cycle(gamma, token_id_list)
    
    # Extract token-level witness
    witness_tokens = extract_witness_tokens(cycle, tokens)
    
    # Extract utterance-level witness
    witness_utterances = extract_witness_utterances(
        witness_tokens, tokens, utterances
    )
    
    # Compute centroid
    centroid = compute_witness_centroid(witness_tokens, tokens)
    
    return {
        "cycle": cycle,
        "tokens": witness_tokens,
        "utterances": witness_utterances,
        "centroid": centroid
    }


# =============================================================================
# Canonical Representative Selection
# =============================================================================

def choose_canonical_gamma(
    candidate_gammas: List[Gamma],
    token_id_list: List[str],
    policy: Dict
) -> Gamma:
    """
    Choose the canonical representative cycle from candidates.
    
    Parameters
    ----------
    candidate_gammas : List[Gamma]
        Multiple candidate representative cycles.
    token_id_list : List[str]
        For lexicographic ordering.
    policy : dict
        CanonicalCyclePolicy with selection criteria.
    
    Returns
    -------
    The chosen canonical Gamma.
    
    Notes
    -----
    From Chapter 4, Definition 4.6: "We fix a canonical witness by
    applying a deterministic policy such as:
    1. among all k-cycles, choose one of minimal length
    2. among such minimal cycles, select one that appears earliest
    3. if ambiguity remains, choose lexicographically first"
    """
    if not candidate_gammas:
        raise ValueError("No candidate cycles provided")
    
    if len(candidate_gammas) == 1:
        return candidate_gammas[0]
    
    candidates = list(candidate_gammas)
    
    # Step 1: Minimal length
    if policy.get("minimal_length", True):
        min_length = min(len(g["simplices"]) for g in candidates)
        candidates = [g for g in candidates if len(g["simplices"]) == min_length]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Step 2: Earliest birth (approximated by smallest min index)
    if policy.get("earliest_birth", True):
        def min_index(g: Gamma) -> int:
            indices = []
            for simp in g["simplices"]:
                indices.extend(simp)
            return min(indices) if indices else float('inf')
        
        earliest = min(min_index(g) for g in candidates)
        candidates = [g for g in candidates if min_index(g) == earliest]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Step 3: Lexicographic tie-break
    if policy.get("lexicographic_tie_break", True):
        def lex_key(g: Gamma) -> Tuple:
            # Sort simplices and flatten for comparison
            sorted_simplices = sorted(
                tuple(sorted(simp)) for simp in g["simplices"]
            )
            return sorted_simplices
        
        candidates.sort(key=lex_key)
    
    return candidates[0]


# =============================================================================
# Witnessed Bar Construction
# =============================================================================

def build_witnessed_bar(
    bar: BarBare,
    gamma: Gamma,
    point_cloud: PointCloudData
) -> WitnessedBar:
    """
    Construct a complete WitnessedBar from a bare bar and its generator.
    
    Parameters
    ----------
    bar : BarBare
        The bare persistence bar.
    gamma : Gamma
        The representative cycle.
    point_cloud : PointCloudData
        Full point cloud data.
    
    Returns
    -------
    Complete WitnessedBar.
    """
    witness = build_witness(gamma, point_cloud)
    
    return {
        "id": bar["id"],
        "dim": bar["dim"],
        "birth": bar["birth"],
        "death": bar["death"],
        "persistence": bar["persistence"],
        "witness": witness
    }


# =============================================================================
# Filtering and Validation
# =============================================================================

def filter_by_witness_size(
    witnessed_bars: List[WitnessedBar],
    min_tokens: int = 2
) -> List[WitnessedBar]:
    """
    Filter bars by minimum witness token count.
    
    Parameters
    ----------
    witnessed_bars : List[WitnessedBar]
        Bars to filter.
    min_tokens : int
        Minimum number of tokens required.
    
    Returns
    -------
    Filtered list of bars.
    
    Notes
    -----
    From Chapter 4: "we keep only those bars whose canonical witnesses
    contain at least three distinct content tokens."
    
    This prevents singleton bars that lack semantic richness.
    """
    return [
        bar for bar in witnessed_bars
        if len(bar["witness"]["tokens"]["ids"]) >= min_tokens
    ]


def validate_witness(witness: Witness) -> List[str]:
    """
    Validate a witness object and return any issues.
    
    Parameters
    ----------
    witness : Witness
        The witness to validate.
    
    Returns
    -------
    List of warning messages (empty if valid).
    """
    warnings = []
    
    # Check for empty token set
    if not witness["tokens"]["ids"]:
        warnings.append("Empty token witness set")
    
    # Check for empty utterance set
    if not witness["utterances"]["ids"]:
        warnings.append("Empty utterance witness set")
    
    # Check for empty cycle
    if not witness["cycle"]["simplices"]:
        warnings.append("Empty cycle simplices")
    
    # Check for empty centroid
    if not witness["centroid"]:
        warnings.append("Empty centroid")
    
    # Check consistency: all cycle tokens should be in tokens.ids
    cycle_tokens = set()
    for simp in witness["cycle"]["simplices"]:
        for tid in simp:
            cycle_tokens.add(tid)
    
    witness_token_set = set(witness["tokens"]["ids"])
    if cycle_tokens - witness_token_set:
        warnings.append(
            f"Cycle contains tokens not in witness set: {cycle_tokens - witness_token_set}"
        )
    
    return warnings


# =============================================================================
# Main Witness Construction Pipeline
# =============================================================================

def build_witnessed_diagram(
    point_cloud: PointCloudData,
    ph_result: PHResult,
    config: Optional[Config] = None
) -> WitnessedDiagram:
    """
    Construct the complete witnessed persistence diagram D^W(τ).
    
    This is the main entry point for Step 4-5 of the pipeline.
    
    Parameters
    ----------
    point_cloud : PointCloudData
        Output from text_to_point_cloud().
    ph_result : PHResult
        Output from compute_witnessed_ph().
    config : Config, optional
        Configuration. Uses defaults if not provided.
    
    Returns
    -------
    WitnessedDiagram with all witnessed bars.
    
    Notes
    -----
    From Chapter 4: "This yields the witnessed persistence diagram
    D^W(τ) = {(0, b_i, d_i, ρ_i) : i ∈ I_τ}, with each witness ρ_i
    carrying:
    - a multiset of surface forms (tokens)
    - a finite set of utterance IDs where these tokens occur"
    """
    if config is None:
        config = default_config()
    
    bars = ph_result["bars"]
    generators = ph_result["generators"]
    
    # Build witnessed bars
    witnessed_bars: List[WitnessedBar] = []
    
    for bar in bars:
        bar_id = bar["id"]
        
        if bar_id not in generators:
            # No generator available - skip
            continue
        
        gamma = generators[bar_id]
        
        # Build the witnessed bar
        witnessed_bar = build_witnessed_bar(bar, gamma, point_cloud)
        
        # Validate
        warnings = validate_witness(witnessed_bar["witness"])
        if "Empty token witness set" in warnings:
            # Skip bars with no witnesses
            continue
        
        witnessed_bars.append(witnessed_bar)
    
    # Filter by witness size
    min_tokens = config.get("min_witness_tokens", 2)
    witnessed_bars = filter_by_witness_size(witnessed_bars, min_tokens)
    
    # Sort by persistence (descending)
    witnessed_bars.sort(key=lambda b: -b["persistence"])
    
    # Reassign IDs after filtering
    for i, bar in enumerate(witnessed_bars):
        bar["id"] = f"bar_{i}"
    
    return {
        "tau": "single_slice",
        "config": config,
        "num_tokens": len(point_cloud["token_ids"]),
        "num_utterances": len(point_cloud["utterances"]),
        "bars": witnessed_bars
    }


# =============================================================================
# Utility Functions for Analysis
# =============================================================================

def get_bar_by_tokens(
    diagram: WitnessedDiagram,
    token_texts: List[str],
    match_mode: str = "any"
) -> List[WitnessedBar]:
    """
    Find bars whose witnesses contain specific tokens.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The witnessed diagram.
    token_texts : List[str]
        Token surface forms or lemmas to search for.
    match_mode : str
        "any" - match if any token is present
        "all" - match only if all tokens are present
    
    Returns
    -------
    List of matching bars.
    """
    token_set = set(t.lower() for t in token_texts)
    matches = []
    
    for bar in diagram["bars"]:
        witness_surfaces = set(t.lower() for t in bar["witness"]["tokens"]["surface"])
        witness_lemmas = set(t.lower() for t in bar["witness"]["tokens"]["lemmas"])
        witness_all = witness_surfaces | witness_lemmas
        
        if match_mode == "any":
            if token_set & witness_all:
                matches.append(bar)
        elif match_mode == "all":
            if token_set <= witness_all:
                matches.append(bar)
    
    return matches


def get_bar_by_utterance(
    diagram: WitnessedDiagram,
    utterance_id: str
) -> List[WitnessedBar]:
    """
    Find bars instantiated in a specific utterance.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The witnessed diagram.
    utterance_id : str
        The utterance ID to search for.
    
    Returns
    -------
    List of bars whose witnesses include this utterance.
    """
    return [
        bar for bar in diagram["bars"]
        if utterance_id in bar["witness"]["utterances"]["ids"]
    ]


def compute_witness_overlap(
    bar1: WitnessedBar,
    bar2: WitnessedBar,
    level: str = "utterance"
) -> float:
    """
    Compute Jaccard similarity of witness sets.
    
    Parameters
    ----------
    bar1, bar2 : WitnessedBar
        Bars to compare.
    level : str
        "utterance" or "token" level comparison.
    
    Returns
    -------
    Jaccard similarity in [0, 1].
    
    Notes
    -----
    From Chapter 4: "overlap(ρ, ρ') ⟺ W_ρ ∩ W_ρ' ≠ ∅"
    """
    if level == "utterance":
        set1 = set(bar1["witness"]["utterances"]["ids"])
        set2 = set(bar2["witness"]["utterances"]["ids"])
    else:
        set1 = set(bar1["witness"]["tokens"]["ids"])
        set2 = set(bar2["witness"]["tokens"]["ids"])
    
    if not set1 and not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0
