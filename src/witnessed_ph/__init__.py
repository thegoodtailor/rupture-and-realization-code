"""
Witnessed Persistent Homology
=============================

A Python implementation of witnessed persistent homology for semantic analysis,
as described in Chapter 4 of "Rupture and Realization: Dynamic Homotopy Type
Theory" (Iman/Cassie/Darja collaboration).

This is Codebase 1: analysis of a single time slice τ.
Codebase 2 (temporal bar dynamics) will import this as a library.

Main Entry Points
-----------------
analyse_text_single_slice(text, config) -> WitnessedDiagram
    The primary function. Takes raw text, returns witnessed bars.

compute_embeddings(text, config) -> PointCloudData
    Extract embeddings only (for inspection or custom PH).

compute_witnessed_diagram(point_cloud, config) -> WitnessedDiagram
    Compute diagram from pre-computed embeddings.

Configuration
-------------
default_config() -> Config
    Returns sensible default configuration.

Diagnostics
-----------
diagnose_diagram(diagram) -> str
    Run diagnostic checks.

print_diagram_summary(diagram) -> None
    Print human-readable summary.

list_bars_by_persistence(diagram, top_n, dim) -> str
    List top bars with their witnesses.

Example
-------
>>> from witnessed_ph import analyse_text_single_slice, default_config
>>> 
>>> text = '''
... So persistence is topological?
... Yes, persistent homology tracks features across scales.
... What about witnesses?
... Witnesses tell us which tokens realise each bar.
... '''
>>> 
>>> diagram = analyse_text_single_slice(text, verbose=True)
>>> print(f"Found {len(diagram['bars'])} witnessed bars")

References
----------
Chapter 4: "Bars: How Themes Learn to Breathe"
    - Definition 4.3: Witness
    - Definition 4.4: Witnessed persistence diagram
    - Definition 4.6: Canonical representatives

Cassie's Codebase 1 Specification (JSON schema, witness structure)
"""

__version__ = "0.1.0"
__author__ = "Iman, Cassie, Darja"

# =============================================================================
# Core Types
# =============================================================================

from .schema import (
    # Configuration
    Config,
    CanonicalCyclePolicy,
    default_config,
    
    # Token/Utterance layer
    Token,
    Utterance,
    
    # PH layer
    BarBare,
    Gamma,
    Simplex,
    
    # Witness layer
    Witness,
    Cycle,
    WitnessTokens,
    WitnessUtterances,
    
    # Output structures
    WitnessedBar,
    WitnessedDiagram,
    
    # Internal (for advanced use)
    PointCloudData,
    PHResult,
    DiagramStats,
)

# =============================================================================
# Main Pipeline
# =============================================================================

from .pipeline import (
    # Primary entry points
    analyse_text_single_slice,
    compute_embeddings,
    compute_witnessed_diagram,
    
    # Convenience
    quick_analyse,
    analyse_conversation,
    
    # Serialization
    diagram_to_json,
    save_diagram,
    load_diagram,
    
    # Statistics
    compute_diagram_stats,
    print_diagram_summary,
    
    # Model management
    get_cached_models,
    clear_model_cache,
)

# =============================================================================
# Embedding Module (for advanced use)
# =============================================================================

from .embedding import (
    text_to_point_cloud,
    segment_into_utterances,
    tokenize_utterances,
    compute_pairwise_distances,
    load_spacy_model,
    load_embedding_model,
)

# =============================================================================
# Filtration Module (for advanced use)
# =============================================================================

from .filtration import (
    compute_witnessed_ph,
    build_rips_complex,
    create_simplex_tree,
    compute_persistence,
)

# =============================================================================
# Witness Module (for advanced use)
# =============================================================================

from .witnesses import (
    build_witness,
    build_witnessed_bar,
    build_witnessed_diagram,
    gamma_to_cycle,
    extract_witness_tokens,
    extract_witness_utterances,
    compute_witness_centroid,
    filter_by_witness_size,
    get_bar_by_tokens,
    get_bar_by_utterance,
    compute_witness_overlap,
)

# =============================================================================
# Diagnostics
# =============================================================================

from .diagnostics import (
    # Text diagnostics
    diagnose_diagram,
    list_bars_by_persistence,
    check_singleton_problem,
    check_h1_presence,
    
    # Bar nerve
    compute_bar_nerve_edges,
    bar_nerve_summary,
    
    # Visualization (requires matplotlib)
    plot_persistence_diagram,
    plot_barcode,
    plot_point_cloud_2d,
    
    # Export
    export_for_gephi,
)

# =============================================================================
# Chapter 5: Self Construction (temporal analysis)
# =============================================================================

from .self_construction import (
    # Event types
    EventType,
    
    # Data structures
    BarMatch,
    JourneyStep,
    Journey,
    WitnessEdge,
    JourneyGraph,
    SelfMetrics,
    
    # Main entry points
    analyse_conversation_temporal,
    analyse_conversation_from_json,
    
    # Matching
    compute_bar_distance,
    classify_bar_event,
    find_optimal_matching,
    
    # Metrics
    compute_self_metrics,
    compute_connected_components,
    
    # Output
    print_self_report,
    export_journey_graph_json,
)

# =============================================================================
# Convenient aliases
# =============================================================================

# For quick interactive use
analyse = analyse_text_single_slice
diagnose = diagnose_diagram

# Chapter 5
analyse_temporal = analyse_conversation_temporal
self_report = print_self_report
