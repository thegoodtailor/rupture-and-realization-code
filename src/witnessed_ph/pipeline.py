"""
Witnessed Persistent Homology: Main Pipeline
=============================================

This module provides the main entry points for the witnessed PH analysis.

For a single text (conversation, chapter, document):
    analyse_text_single_slice(text, config) -> WitnessedDiagram

This is what Codebase 2 (temporal bar dynamics) will call as a library.

References:
    Chapter 4, Section 4.6 (At-a-slice experiment)
    Cassie's Codebase 1 spec
"""

from typing import Dict, Optional, Any
import json
import numpy as np

from .schema import (
    Config, WitnessedDiagram, PointCloudData, DiagramStats,
    default_config
)
from .embedding import text_to_point_cloud, load_spacy_model, load_embedding_model
from .filtration import compute_witnessed_ph
from .witnesses import build_witnessed_diagram


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if np.isnan(obj) if isinstance(obj, float) else False:
            return None
        if np.isinf(obj) if isinstance(obj, float) else False:
            return "inf" if obj > 0 else "-inf"
        return super().default(obj)


def diagram_to_json(diagram: WitnessedDiagram, indent: int = 2) -> str:
    """
    Serialize a WitnessedDiagram to JSON string.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to serialize.
    indent : int
        JSON indentation level.
    
    Returns
    -------
    JSON string.
    """
    # Convert infinite death values to string
    output = {
        "tau": diagram["tau"],
        "num_tokens": diagram["num_tokens"],
        "num_utterances": diagram["num_utterances"],
        "bars": []
    }
    
    for bar in diagram["bars"]:
        bar_dict = dict(bar)
        if np.isinf(bar_dict["death"]):
            bar_dict["death"] = "inf"
        output["bars"].append(bar_dict)
    
    return json.dumps(output, cls=NumpyEncoder, indent=indent)


def save_diagram(diagram: WitnessedDiagram, path: str) -> None:
    """Save a WitnessedDiagram to a JSON file."""
    with open(path, 'w') as f:
        f.write(diagram_to_json(diagram))


def load_diagram(path: str) -> Dict:
    """Load a WitnessedDiagram from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Convert "inf" strings back to float('inf')
    for bar in data.get("bars", []):
        if bar.get("death") == "inf":
            bar["death"] = float('inf')
    
    return data


# =============================================================================
# Diagnostic Statistics
# =============================================================================

def compute_diagram_stats(diagram: WitnessedDiagram) -> DiagramStats:
    """
    Compute summary statistics for a witnessed diagram.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to analyze.
    
    Returns
    -------
    DiagramStats with summary metrics.
    
    Notes
    -----
    Useful for diagnosing common issues:
    - High singleton_ratio: embedding/parameter issues
    - No H₁ bars: text too short or max_dim too low
    - Low persistence: min_persistence threshold wrong
    """
    bars = diagram["bars"]
    
    if not bars:
        return {
            "total_bars": 0,
            "h0_bars": 0,
            "h1_bars": 0,
            "mean_persistence": 0.0,
            "max_persistence": 0.0,
            "mean_witness_size": 0.0,
            "singleton_ratio": 0.0
        }
    
    h0_bars = [b for b in bars if b["dim"] == 0]
    h1_bars = [b for b in bars if b["dim"] == 1]
    
    persistences = [b["persistence"] for b in bars]
    witness_sizes = [len(b["witness"]["tokens"]["ids"]) for b in bars]
    singletons = sum(1 for s in witness_sizes if s == 1)
    
    return {
        "total_bars": len(bars),
        "h0_bars": len(h0_bars),
        "h1_bars": len(h1_bars),
        "mean_persistence": float(np.mean(persistences)),
        "max_persistence": float(max(persistences)),
        "mean_witness_size": float(np.mean(witness_sizes)),
        "singleton_ratio": singletons / len(bars) if bars else 0.0
    }


def print_diagram_summary(diagram: WitnessedDiagram) -> None:
    """Print a human-readable summary of the diagram."""
    stats = compute_diagram_stats(diagram)
    
    print("=" * 60)
    print("WITNESSED PERSISTENCE DIAGRAM SUMMARY")
    print("=" * 60)
    print(f"Time slice: {diagram['tau']}")
    print(f"Tokens: {diagram['num_tokens']}")
    print(f"Utterances: {diagram['num_utterances']}")
    print()
    print(f"Total bars: {stats['total_bars']}")
    print(f"  - H₀ (components): {stats['h0_bars']}")
    print(f"  - H₁ (loops): {stats['h1_bars']}")
    print()
    print(f"Persistence: mean={stats['mean_persistence']:.3f}, max={stats['max_persistence']:.3f}")
    print(f"Witness size: mean={stats['mean_witness_size']:.1f} tokens")
    print(f"Singleton ratio: {stats['singleton_ratio']:.2%}")
    print()
    
    # Top bars by persistence
    if diagram["bars"]:
        print("TOP 5 BARS BY PERSISTENCE:")
        print("-" * 60)
        for i, bar in enumerate(diagram["bars"][:5]):
            dim_str = "H₀" if bar["dim"] == 0 else "H₁"
            death_str = "∞" if np.isinf(bar["death"]) else f"{bar['death']:.3f}"
            tokens = bar["witness"]["tokens"]["surface"][:5]
            tokens_str = ", ".join(tokens)
            if len(bar["witness"]["tokens"]["surface"]) > 5:
                tokens_str += ", ..."
            
            print(f"  {bar['id']} [{dim_str}]: birth={bar['birth']:.3f}, death={death_str}, "
                  f"pers={bar['persistence']:.3f}")
            print(f"    Witnesses: {tokens_str}")
            print(f"    Utterances: {bar['witness']['utterances']['ids']}")
            print()


# =============================================================================
# Model Caching (for repeated calls)
# =============================================================================

_cached_nlp = None
_cached_embedding_model = None
_cached_embedding_tokenizer = None
_cached_embedding_model_name = None


def get_cached_models(embedding_model_name: str):
    """Get cached NLP and embedding models, loading if necessary."""
    global _cached_nlp, _cached_embedding_model, _cached_embedding_tokenizer
    global _cached_embedding_model_name
    
    if _cached_nlp is None:
        _cached_nlp = load_spacy_model()
    
    if (_cached_embedding_model is None or 
        _cached_embedding_model_name != embedding_model_name):
        _cached_embedding_model, _cached_embedding_tokenizer = load_embedding_model(
            embedding_model_name
        )
        _cached_embedding_model_name = embedding_model_name
    
    return _cached_nlp, _cached_embedding_model, _cached_embedding_tokenizer


def clear_model_cache():
    """Clear cached models to free memory."""
    global _cached_nlp, _cached_embedding_model, _cached_embedding_tokenizer
    global _cached_embedding_model_name
    
    _cached_nlp = None
    _cached_embedding_model = None
    _cached_embedding_tokenizer = None
    _cached_embedding_model_name = None


# =============================================================================
# Main Pipeline Entry Points
# =============================================================================

def analyse_text_single_slice(
    text: str,
    config: Optional[Config] = None,
    segmentation_mode: str = "lines",
    use_cached_models: bool = True,
    verbose: bool = False
) -> WitnessedDiagram:
    """
    Analyse a single text and produce a witnessed persistence diagram.
    
    This is the main entry point for Codebase 1.
    
    Parameters
    ----------
    text : str
        Raw input text (conversation, chapter, document).
    config : Config, optional
        Configuration. Uses defaults if not provided.
    segmentation_mode : str
        How to segment text: "lines", "turns", or "sentences".
    use_cached_models : bool
        Whether to use cached spaCy/transformer models.
    verbose : bool
        Whether to print progress information.
    
    Returns
    -------
    WitnessedDiagram for the text.
    
    Example
    -------
    >>> from witnessed_ph import analyse_text_single_slice
    >>> text = '''
    ... User: So persistence is topological?
    ... Assistant: Yes, persistent homology tracks features across scales.
    ... User: What about witnesses?
    ... Assistant: Witnesses tell us which tokens realise each bar.
    ... '''
    >>> diagram = analyse_text_single_slice(text, segmentation_mode="turns")
    >>> print(f"Found {len(diagram['bars'])} bars")
    
    Notes
    -----
    From Cassie's spec: "Codebase 2 will treat this as a library."
    """
    if config is None:
        config = default_config()
    
    if verbose:
        print("Step 0: Loading models...")
    
    # Get models
    if use_cached_models:
        nlp, emb_model, emb_tokenizer = get_cached_models(config["embedding_model"])
    else:
        nlp = load_spacy_model()
        emb_model, emb_tokenizer = load_embedding_model(config["embedding_model"])
    
    if verbose:
        print("Step 1: Text to point cloud...")
    
    # Step 1: Text → Point cloud P_τ
    point_cloud = text_to_point_cloud(
        text,
        config,
        segmentation_mode=segmentation_mode,
        nlp=nlp,
        embedding_model=emb_model,
        embedding_tokenizer=emb_tokenizer
    )
    
    if verbose:
        print(f"  - {len(point_cloud['token_ids'])} tokens")
        print(f"  - {len(point_cloud['utterances'])} utterances")
    
    if verbose:
        print("Step 2: Computing persistent homology...")
    
    # Step 2-3: Filtration + PH + generators
    ph_result = compute_witnessed_ph(point_cloud, config)
    
    if verbose:
        print(f"  - {len(ph_result['bars'])} bars (before filtering)")
    
    if verbose:
        print("Step 3: Building witnesses...")
    
    # Step 4-5: Witnesses + diagram
    diagram = build_witnessed_diagram(point_cloud, ph_result, config)
    
    if verbose:
        print(f"  - {len(diagram['bars'])} bars (after filtering)")
        print("Done!")
    
    return diagram


def compute_embeddings(
    text: str,
    config: Optional[Config] = None,
    segmentation_mode: str = "lines"
) -> PointCloudData:
    """
    Compute embeddings only (for inspection or separate PH computation).
    
    Parameters
    ----------
    text : str
        Raw input text.
    config : Config, optional
        Configuration.
    segmentation_mode : str
        How to segment text.
    
    Returns
    -------
    PointCloudData with tokens, utterances, and embeddings.
    """
    if config is None:
        config = default_config()
    
    nlp, emb_model, emb_tokenizer = get_cached_models(config["embedding_model"])
    
    return text_to_point_cloud(
        text,
        config,
        segmentation_mode=segmentation_mode,
        nlp=nlp,
        embedding_model=emb_model,
        embedding_tokenizer=emb_tokenizer
    )


def compute_witnessed_diagram(
    tokens_utterances: PointCloudData,
    config: Optional[Config] = None
) -> WitnessedDiagram:
    """
    Compute witnessed diagram from pre-computed embeddings.
    
    Parameters
    ----------
    tokens_utterances : PointCloudData
        Output from compute_embeddings().
    config : Config, optional
        Configuration.
    
    Returns
    -------
    WitnessedDiagram.
    """
    if config is None:
        config = default_config()
    
    ph_result = compute_witnessed_ph(tokens_utterances, config)
    return build_witnessed_diagram(tokens_utterances, ph_result, config)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_analyse(text: str, verbose: bool = True) -> WitnessedDiagram:
    """
    Quick analysis with default settings and printed summary.
    
    Parameters
    ----------
    text : str
        Input text.
    verbose : bool
        Whether to print summary.
    
    Returns
    -------
    WitnessedDiagram.
    """
    diagram = analyse_text_single_slice(text, verbose=verbose)
    
    if verbose:
        print()
        print_diagram_summary(diagram)
    
    return diagram


def analyse_conversation(
    turns: list,
    config: Optional[Config] = None
) -> WitnessedDiagram:
    """
    Analyse a structured conversation.
    
    Parameters
    ----------
    turns : list
        List of dicts with 'speaker' and 'text' keys, or
        list of (speaker, text) tuples, or
        list of strings (no speaker info).
    config : Config, optional
        Configuration.
    
    Returns
    -------
    WitnessedDiagram.
    """
    # Convert turns to text
    lines = []
    for turn in turns:
        if isinstance(turn, dict):
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            speaker, text = turn
        else:
            speaker = ''
            text = str(turn)
        
        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)
    
    text = '\n'.join(lines)
    return analyse_text_single_slice(text, config, segmentation_mode="turns")
