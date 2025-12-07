"""
Witnessed Persistent Homology: Type Definitions
================================================

This module defines the core data structures for witnessed persistent homology
as specified in Chapter 4 of "Rupture and Realization" (DHoTT framework).

Key concepts:
- TokenID: unique ID per occurrence (not type) - same word appearing twice gets two IDs
- Witness ρ = (W^tok, W^loc, γ) where:
    - W^tok: token IDs participating in the feature
    - W^loc: utterance IDs (measurement locations) containing those tokens
    - γ: the actual representative cycle (simplices)

The witnessed bar is (k, b, d, ρ) where:
- k: homology dimension (0 = components, 1 = loops)
- b: birth radius (when feature appears)
- d: death radius (when feature dies)
- ρ: witness attaching semantic content to the topological feature

References:
    Chapter 4, Definition 4.3 (Witness)
    Chapter 4, Definition 4.4 (Witnessed persistence diagram)
"""

from typing import TypedDict, List, Dict, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Configuration
# =============================================================================

class CanonicalCyclePolicy(TypedDict):
    """
    Policy for choosing canonical representative cycles among equivalents.
    
    From Chapter 4, Definition 4.6 (Canonical representatives):
    1. minimal_length: prefer cycles with fewest simplices
    2. earliest_birth: among minimal, prefer those appearing earliest in filtration
    3. lexicographic_tie_break: final tie-breaker using token ID ordering
    """
    minimal_length: bool
    earliest_birth: bool
    lexicographic_tie_break: bool


class Config(TypedDict):
    """
    Configuration for the witnessed PH pipeline.
    
    Parameters:
    -----------
    embedding_model : str
        HuggingFace model identifier for contextual embeddings.
        Default: "microsoft/deberta-v3-base"
    
    min_token_len : int
        Minimum character length for tokens to be included.
        
    use_lemmas : bool
        Whether to use lemmatized forms for token surface text.
        
    pos_filter : List[str]
        POS tags to include (e.g., ["NOUN", "VERB", "ADJ"]).
    
    stopwords : List[str]
        Words to exclude from tokenization (case-insensitive).
        Useful for filtering speaker labels like "User", "Assistant".
        
    min_persistence : float
        Minimum persistence (d - b) for a bar to be kept.
        Filters out topological noise.
        
    max_dim : int
        Maximum homology dimension to compute (0 or 1 for this codebase).
        
    distance_metric : str
        Base metric for embeddings. "cosine" converted to angular distance.
        
    filtration : str
        Filtration type: "vietoris_rips" or "cech".
        Note: VR is used in practice; Čech is conceptually cleaner (Remark 4.1).
        
    max_r : float
        Maximum radius for filtration.
        
    num_r_steps : int
        Number of discrete radius steps in filtration.
        
    canonical_cycle_policy : CanonicalCyclePolicy
        Policy for selecting canonical representative cycles.
        
    min_witness_tokens : int
        Minimum number of tokens in witness set for bar to be kept.
        Prevents singleton bars that lack semantic richness.
        
    lambda_semantic : float
        Weight λ ∈ [0,1] for semantic distance in bar comparison.
        See Definition 4.8 (Witnessed bar distance).
    """
    embedding_model: str
    min_token_len: int
    use_lemmas: bool
    pos_filter: List[str]
    stopwords: List[str]  # Words to exclude (e.g., speaker labels)
    min_persistence: float
    max_dim: int
    distance_metric: str
    filtration: str
    max_r: float
    num_r_steps: int
    canonical_cycle_policy: CanonicalCyclePolicy
    min_witness_tokens: int
    lambda_semantic: float


def default_config() -> Config:
    """Return sensible default configuration."""
    return {
        "embedding_model": "microsoft/deberta-v3-base",
        "min_token_len": 2,
        "use_lemmas": True,
        "pos_filter": ["NOUN", "VERB", "ADJ", "PROPN"],
        "stopwords": ["user", "assistant", "speaker", "human", "ai"],  # Filter speaker labels
        "min_persistence": 0.03,  # Tuned to catch H1 in real conversations
        "max_dim": 1,
        "distance_metric": "cosine",
        "filtration": "vietoris_rips",
        "max_r": 1.0,
        "num_r_steps": 50,
        "canonical_cycle_policy": {
            "minimal_length": True,
            "earliest_birth": True,
            "lexicographic_tie_break": True
        },
        "min_witness_tokens": 2,
        "lambda_semantic": 0.5
    }


# =============================================================================
# Token and Utterance Layer
# =============================================================================

class Token(TypedDict):
    """
    A single token occurrence with its embedding.
    
    Note: TokenID is per-occurrence, not per-type. If "persistence" appears
    in utterances u3 and u7, those are two different Token objects with
    different IDs (e.g., "u3_tok_5" and "u7_tok_2").
    
    This is crucial: persistent homology works on the point cloud of
    *occurrences*, not abstract word types.
    """
    id: str                    # Unique occurrence ID, e.g. "u7_tok_3"
    text: str                  # Surface form
    lemma: str                 # Lemmatized form (if use_lemmas=True)
    pos: str                   # Part-of-speech tag
    utterance_id: str          # Backpointer to containing utterance
    char_start: int            # Character offset in utterance
    char_end: int              # Character offset end
    embedding: NDArray         # Shape (d,), normalized to unit norm


class Utterance(TypedDict):
    """
    A measurement location: one utterance/turn/sentence in the text.
    
    From Chapter 4: "measurement locations" are the granularity at which
    we record where bars are instantiated. W^loc is a set of these.
    """
    id: str                    # e.g. "u7"
    speaker: Optional[str]     # Speaker label if available
    text: str                  # Raw text content
    token_ids: List[str]       # IDs of tokens in this utterance
    embedding: NDArray         # Pooled embedding for the utterance


# =============================================================================
# Filtration and Persistent Homology Layer
# =============================================================================

class BarBare(TypedDict):
    """
    A bare persistence bar without witness information.
    
    This is what standard TDA libraries (GUDHI, Ripser) produce.
    Chapter 4 calls this D(τ) = {(k_i, b_i, d_i)}.
    """
    id: str                    # Bar identifier
    dim: int                   # Homology dimension (0 or 1)
    birth: float               # Birth radius
    death: float               # Death radius (may be np.inf)
    persistence: float         # death - birth


# A simplex is a list of indices into the point cloud
# For dim=0: single vertex [i]
# For dim=1: edge [i, j]
# For dim=2: triangle [i, j, k]
Simplex = List[int]


class Gamma(TypedDict):
    """
    A representative cycle γ for a homology class.
    
    From Chapter 4: γ is one concrete way the witness tokens are "stitched
    together" in the complex. For H₀, it's the vertices of the component;
    for H₁, it's a list of edges forming a cycle.
    
    The simplices field contains indices into the point cloud (token list).
    These get mapped to token IDs in the Witness structure.
    """
    dim: int                   # 0 for component, 1 for loop
    simplices: List[Simplex]   # List of simplices; indices are point cloud positions


# =============================================================================
# Witness Layer
# =============================================================================

class Cycle(TypedDict):
    """
    The cycle component of a witness, with token IDs instead of indices.
    
    This is γ expressed in terms of the actual token identifiers, not
    abstract point cloud positions. This is what we store in JSON output.
    
    From Cassie's spec: "γ lives inside witness.cycle, and the 'witness tokens'
    are a shadow/projection of γ, not a replacement."
    """
    dimension: int
    simplices: List[List[str]]  # Each simplex as list of token IDs


class WitnessTokens(TypedDict):
    """
    Token-level component of a witness.
    
    This is W^tok_ρ from Definition 4.3, plus convenience fields.
    """
    ids: List[str]             # W^tok: set of token IDs
    surface: List[str]         # Surface forms for display
    lemmas: List[str]          # Lemmatized forms


class WitnessUtterances(TypedDict):
    """
    Utterance-level component of a witness.
    
    This is W^loc_ρ from Definition 4.3: the measurement locations
    that contain the witness tokens.
    """
    ids: List[str]             # W^loc: set of utterance IDs
    text_samples: List[str]    # Text content (possibly truncated for display)


class Witness(TypedDict):
    """
    Complete witness ρ = (W^tok, W^loc, γ) for a persistence bar.
    
    From Chapter 4, Definition 4.3:
    - W^tok_ρ: finite non-empty set of token occurrences
    - γ_ρ: representative k-cycle whose support lies in W^tok_ρ
    - W^loc_ρ: induced set of measurement locations
    
    The witness "enriches" a bare topological bar with semantic content:
    we can now NAME what the bar is about.
    """
    cycle: Cycle               # γ: the actual representative cycle
    tokens: WitnessTokens      # W^tok: the tokens that witness the feature
    utterances: WitnessUtterances  # W^loc: where those tokens occur
    centroid: List[float]      # c(ρ): normalized mean of token embeddings


# =============================================================================
# Witnessed Bars and Diagrams
# =============================================================================

class WitnessedBar(TypedDict):
    """
    A witnessed persistence bar: (k, b, d, ρ).
    
    From Chapter 4, Definition 4.4 (Witnessed persistence diagram):
    D^W(τ) = {(k_i, b_i, d_i, ρ_i)}
    
    This is where "statistics on embeddings start turning into themes
    we can name and track" (Chapter 4 introduction).
    """
    id: str                    # Unique bar identifier
    dim: int                   # Homology dimension k
    birth: float               # Birth radius b
    death: float               # Death radius d
    persistence: float         # d - b
    witness: Witness           # ρ: the semantic witness


class WitnessedDiagram(TypedDict):
    """
    The witnessed persistence diagram D^W(τ) for a single time slice.
    
    This is the complete output of Codebase 1: for one text (conversation,
    chapter, etc.) we produce a set of witnessed bars capturing its
    topological-semantic structure.
    
    In Chapter 4 terminology: this is what the "at-a-slice experiment"
    produces before any temporal evolution.
    """
    tau: Union[float, str]     # Time identifier or "single_slice"
    config: Config             # Configuration used
    num_tokens: int            # Total tokens in point cloud
    num_utterances: int        # Total utterances
    bars: List[WitnessedBar]   # The witnessed bars


# =============================================================================
# Diagnostics and Metrics
# =============================================================================

class DiagramStats(TypedDict):
    """
    Summary statistics for a witnessed diagram.
    
    Useful for diagnosing common issues like:
    - Too many singleton bars (embedding issues)
    - No H₁ features (text too short or parameters wrong)
    - Low persistence bars only (min_persistence too high)
    """
    total_bars: int
    h0_bars: int
    h1_bars: int
    mean_persistence: float
    max_persistence: float
    mean_witness_size: float
    singleton_ratio: float     # Fraction of bars with single-token witnesses


# =============================================================================
# Internal intermediate structures (not exported to JSON)
# =============================================================================

class PointCloudData(TypedDict):
    """
    Internal structure holding the normalized point cloud P_τ.
    """
    embeddings: NDArray        # Shape (N, d), unit normalized
    token_ids: List[str]       # Mapping from index to token ID
    tokens: Dict[str, Token]   # Full token data by ID
    utterances: Dict[str, Utterance]  # Full utterance data by ID


class PHResult(TypedDict):
    """
    Internal structure for raw persistent homology output.
    """
    bars: List[BarBare]
    generators: Dict[str, Gamma]  # bar_id -> representative cycle
