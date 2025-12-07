"""
Witnessed Persistent Homology: Diagnostics and Visualization
=============================================================

This module provides tools for:
1. Visualizing persistence diagrams
2. Plotting H₁ cycles on UMAP/t-SNE projections
3. Inspecting witness quality
4. Debugging common issues

References:
    Chapter 4, Section 4.6 (Diagnostic observations)
    Cassie's spec: "Check for multi-token bars; warn if >80% are singletons"
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .schema import (
    WitnessedDiagram, WitnessedBar, PointCloudData, DiagramStats
)


# =============================================================================
# Text-based Diagnostics (no plotting dependencies)
# =============================================================================

def list_bars_by_persistence(
    diagram: WitnessedDiagram,
    top_n: int = 10,
    dim: Optional[int] = None
) -> str:
    """
    Generate a text report of top bars by persistence.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to report on.
    top_n : int
        Number of bars to show.
    dim : int, optional
        Filter to specific dimension (0 or 1).
    
    Returns
    -------
    Formatted string report.
    """
    bars = diagram["bars"]
    if dim is not None:
        bars = [b for b in bars if b["dim"] == dim]
    
    # Already sorted by persistence
    bars = bars[:top_n]
    
    lines = []
    lines.append("=" * 70)
    dim_str = f"H{dim}" if dim is not None else "ALL"
    lines.append(f"TOP {len(bars)} BARS BY PERSISTENCE ({dim_str})")
    lines.append("=" * 70)
    
    for bar in bars:
        dim_label = "H₀" if bar["dim"] == 0 else "H₁"
        death_str = "∞" if np.isinf(bar["death"]) else f"{bar['death']:.4f}"
        
        lines.append(f"\n{bar['id']} [{dim_label}]")
        lines.append(f"  Birth: {bar['birth']:.4f}")
        lines.append(f"  Death: {death_str}")
        lines.append(f"  Persistence: {bar['persistence']:.4f}")
        lines.append(f"  Witness tokens ({len(bar['witness']['tokens']['ids'])}):")
        
        # Show tokens with their surface forms
        tokens = bar["witness"]["tokens"]
        for i, (tid, surface, lemma) in enumerate(zip(
            tokens["ids"][:10], 
            tokens["surface"][:10], 
            tokens["lemmas"][:10]
        )):
            lines.append(f"    {surface} ({lemma}) [{tid}]")
        if len(tokens["ids"]) > 10:
            lines.append(f"    ... and {len(tokens['ids']) - 10} more")
        
        lines.append(f"  Utterances: {bar['witness']['utterances']['ids']}")
        
        # Show cycle structure for H₁
        if bar["dim"] == 1:
            cycle = bar["witness"]["cycle"]
            lines.append(f"  Cycle edges ({len(cycle['simplices'])}):")
            for simp in cycle["simplices"][:5]:
                lines.append(f"    {simp}")
            if len(cycle["simplices"]) > 5:
                lines.append(f"    ... and {len(cycle['simplices']) - 5} more edges")
    
    return "\n".join(lines)


def check_singleton_problem(
    diagram: WitnessedDiagram,
    warn_threshold: float = 0.8
) -> Tuple[bool, str]:
    """
    Check if the diagram has too many singleton witnesses.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to check.
    warn_threshold : float
        Fraction above which to warn (default 0.8 = 80%).
    
    Returns
    -------
    (is_problem, message) tuple.
    
    Notes
    -----
    From Cassie's spec: "Check for multi-token bars; warn if >80% are singletons"
    
    High singleton ratio usually indicates:
    - POS filter too restrictive
    - min_token_len too high
    - Embeddings not grouping semantically similar tokens
    - Text too short
    """
    bars = diagram["bars"]
    if not bars:
        return False, "No bars in diagram"
    
    singleton_count = sum(
        1 for b in bars 
        if len(b["witness"]["tokens"]["ids"]) == 1
    )
    ratio = singleton_count / len(bars)
    
    if ratio > warn_threshold:
        msg = (
            f"WARNING: {ratio:.1%} of bars are singletons "
            f"({singleton_count}/{len(bars)})\n"
            "This suggests issues with:\n"
            "  - POS filter may be too restrictive\n"
            "  - min_token_len may be too high\n"
            "  - Embedding model may not be grouping tokens well\n"
            "  - Text may be too short for meaningful clusters"
        )
        return True, msg
    else:
        msg = f"Singleton ratio: {ratio:.1%} ({singleton_count}/{len(bars)}) - OK"
        return False, msg


def check_h1_presence(diagram: WitnessedDiagram) -> Tuple[bool, str]:
    """
    Check if H₁ bars are present and report.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to check.
    
    Returns
    -------
    (has_h1, message) tuple.
    """
    h1_bars = [b for b in diagram["bars"] if b["dim"] == 1]
    
    if not h1_bars:
        msg = (
            "No H₁ bars found. This may indicate:\n"
            "  - Text is too short for loops to form\n"
            "  - max_dim in config is set to 0\n"
            "  - min_persistence threshold is filtering out H₁ bars\n"
            "  - No genuine loop structure in the token cloud"
        )
        return False, msg
    else:
        msg = f"Found {len(h1_bars)} H₁ bars (loops)"
        return True, msg


def diagnose_diagram(diagram: WitnessedDiagram) -> str:
    """
    Run all diagnostic checks and return a report.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to diagnose.
    
    Returns
    -------
    Diagnostic report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    
    # Basic stats
    lines.append(f"\nTokens: {diagram['num_tokens']}")
    lines.append(f"Utterances: {diagram['num_utterances']}")
    lines.append(f"Total bars: {len(diagram['bars'])}")
    
    h0 = len([b for b in diagram["bars"] if b["dim"] == 0])
    h1 = len([b for b in diagram["bars"] if b["dim"] == 1])
    lines.append(f"  H₀ bars: {h0}")
    lines.append(f"  H₁ bars: {h1}")
    
    # Singleton check
    lines.append("\n--- Singleton Check ---")
    is_problem, msg = check_singleton_problem(diagram)
    lines.append(msg)
    
    # H₁ check
    lines.append("\n--- H₁ Check ---")
    has_h1, msg = check_h1_presence(diagram)
    lines.append(msg)
    
    # Persistence distribution
    if diagram["bars"]:
        lines.append("\n--- Persistence Distribution ---")
        persistences = [b["persistence"] for b in diagram["bars"]]
        lines.append(f"Min: {min(persistences):.4f}")
        lines.append(f"Max: {max(persistences):.4f}")
        lines.append(f"Mean: {np.mean(persistences):.4f}")
        lines.append(f"Median: {np.median(persistences):.4f}")
    
    # Witness size distribution
    if diagram["bars"]:
        lines.append("\n--- Witness Size Distribution ---")
        sizes = [len(b["witness"]["tokens"]["ids"]) for b in diagram["bars"]]
        lines.append(f"Min: {min(sizes)}")
        lines.append(f"Max: {max(sizes)}")
        lines.append(f"Mean: {np.mean(sizes):.1f}")
        
        # Size histogram
        size_counts = {}
        for s in sizes:
            size_counts[s] = size_counts.get(s, 0) + 1
        lines.append("Distribution:")
        for size in sorted(size_counts.keys())[:10]:
            count = size_counts[size]
            bar = "█" * min(count, 50)
            lines.append(f"  {size:3d} tokens: {bar} ({count})")
    
    return "\n".join(lines)


# =============================================================================
# Bar Nerve Construction (for future Codebase 2)
# =============================================================================

def compute_bar_nerve_edges(
    diagram: WitnessedDiagram,
    level: str = "utterance"
) -> List[Tuple[str, str]]:
    """
    Compute edges of the bar nerve (bars that share witnesses).
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram.
    level : str
        "utterance" or "token" level overlap.
    
    Returns
    -------
    List of (bar_id1, bar_id2) tuples for overlapping bars.
    
    Notes
    -----
    From Chapter 4: "An edge (B, B') exists precisely when
    W_B^utt ∩ W_B'^utt ≠ ∅"
    """
    bars = diagram["bars"]
    edges = []
    
    for i, bar1 in enumerate(bars):
        for bar2 in bars[i+1:]:
            if level == "utterance":
                set1 = set(bar1["witness"]["utterances"]["ids"])
                set2 = set(bar2["witness"]["utterances"]["ids"])
            else:
                set1 = set(bar1["witness"]["tokens"]["ids"])
                set2 = set(bar2["witness"]["tokens"]["ids"])
            
            if set1 & set2:  # Non-empty intersection
                edges.append((bar1["id"], bar2["id"]))
    
    return edges


def bar_nerve_summary(diagram: WitnessedDiagram) -> str:
    """
    Generate a text summary of the bar nerve structure.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram.
    
    Returns
    -------
    Summary string.
    """
    edges = compute_bar_nerve_edges(diagram, level="utterance")
    
    lines = []
    lines.append("=" * 70)
    lines.append("BAR NERVE STRUCTURE (Utterance-level)")
    lines.append("=" * 70)
    lines.append(f"Vertices (bars): {len(diagram['bars'])}")
    lines.append(f"Edges (overlaps): {len(edges)}")
    
    if edges:
        # Compute degree distribution
        degrees = {}
        for bar in diagram["bars"]:
            degrees[bar["id"]] = 0
        for b1, b2 in edges:
            degrees[b1] = degrees.get(b1, 0) + 1
            degrees[b2] = degrees.get(b2, 0) + 1
        
        max_deg = max(degrees.values()) if degrees else 0
        lines.append(f"Max degree: {max_deg}")
        
        # Highest-degree bars (most connected themes)
        sorted_bars = sorted(degrees.items(), key=lambda x: -x[1])
        lines.append("\nMost connected bars:")
        for bar_id, deg in sorted_bars[:5]:
            bar = next(b for b in diagram["bars"] if b["id"] == bar_id)
            tokens = ", ".join(bar["witness"]["tokens"]["surface"][:3])
            lines.append(f"  {bar_id} (deg={deg}): {tokens}...")
    
    return "\n".join(lines)


# =============================================================================
# Visualization (requires matplotlib)
# =============================================================================

def plot_persistence_diagram(
    diagram: WitnessedDiagram,
    ax=None,
    title: str = "Persistence Diagram",
    show_labels: bool = True
):
    """
    Plot the persistence diagram.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to plot.
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.
    title : str
        Plot title.
    show_labels : bool
        Whether to show bar IDs as labels.
    
    Returns
    -------
    The axes object.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    bars = diagram["bars"]
    
    # Separate by dimension
    h0_bars = [b for b in bars if b["dim"] == 0]
    h1_bars = [b for b in bars if b["dim"] == 1]
    
    # Plot diagonal
    max_val = max(
        max((b["death"] for b in bars if not np.isinf(b["death"])), default=1),
        max((b["birth"] for b in bars), default=1)
    )
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3)
    
    # Plot H₀ bars
    if h0_bars:
        births = [b["birth"] for b in h0_bars]
        deaths = [b["death"] if not np.isinf(b["death"]) else max_val * 1.1 
                  for b in h0_bars]
        ax.scatter(births, deaths, c='blue', s=50, label='H₀', alpha=0.7)
        
        if show_labels:
            for b, birth, death in zip(h0_bars, births, deaths):
                ax.annotate(b["id"], (birth, death), fontsize=8, alpha=0.7)
    
    # Plot H₁ bars
    if h1_bars:
        births = [b["birth"] for b in h1_bars]
        deaths = [b["death"] if not np.isinf(b["death"]) else max_val * 1.1 
                  for b in h1_bars]
        ax.scatter(births, deaths, c='red', s=50, marker='^', label='H₁', alpha=0.7)
        
        if show_labels:
            for b, birth, death in zip(h1_bars, births, deaths):
                ax.annotate(b["id"], (birth, death), fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    
    return ax


def plot_barcode(
    diagram: WitnessedDiagram,
    ax=None,
    title: str = "Persistence Barcode"
):
    """
    Plot the persistence barcode.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram to plot.
    ax : matplotlib axes, optional
        Axes to plot on.
    title : str
        Plot title.
    
    Returns
    -------
    The axes object.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = diagram["bars"]
    
    # Find max finite death for infinite bars
    finite_deaths = [b["death"] for b in bars if not np.isinf(b["death"])]
    max_death = max(finite_deaths) if finite_deaths else 1.0
    
    y_pos = 0
    colors = {'H0': 'blue', 'H1': 'red'}
    
    for bar in bars:
        color = colors.get(f"H{bar['dim']}", 'gray')
        birth = bar["birth"]
        death = bar["death"] if not np.isinf(bar["death"]) else max_death * 1.2
        
        ax.hlines(y_pos, birth, death, colors=color, linewidth=2)
        
        # Add marker for infinite bars
        if np.isinf(bar["death"]):
            ax.scatter([death], [y_pos], marker='>', c=color, s=30)
        
        y_pos += 1
    
    ax.set_xlabel("Scale (r)")
    ax.set_ylabel("Bar index")
    ax.set_title(title)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='H₀'),
        Line2D([0], [0], color='red', linewidth=2, label='H₁')
    ]
    ax.legend(handles=legend_elements)
    
    return ax


def plot_point_cloud_2d(
    point_cloud: PointCloudData,
    diagram: Optional[WitnessedDiagram] = None,
    method: str = "umap",
    highlight_bar: Optional[str] = None,
    ax=None,
    title: str = "Token Embedding Point Cloud"
):
    """
    Plot 2D projection of the point cloud with optional bar highlighting.
    
    Parameters
    ----------
    point_cloud : PointCloudData
        The point cloud data.
    diagram : WitnessedDiagram, optional
        If provided, can highlight specific bars.
    method : str
        Dimensionality reduction method: "umap" or "tsne".
    highlight_bar : str, optional
        Bar ID to highlight.
    ax : matplotlib axes, optional
        Axes to plot on.
    title : str
        Plot title.
    
    Returns
    -------
    The axes object.
    """
    import matplotlib.pyplot as plt
    
    embeddings = point_cloud["embeddings"]
    token_ids = point_cloud["token_ids"]
    tokens = point_cloud["tokens"]
    
    # Reduce to 2D
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            method = "tsne"
    
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        coords = reducer.fit_transform(embeddings)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Default plot all points
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=20, alpha=0.5)
    
    # Highlight specific bar if requested
    if highlight_bar and diagram:
        bar = next((b for b in diagram["bars"] if b["id"] == highlight_bar), None)
        if bar:
            witness_token_ids = set(bar["witness"]["tokens"]["ids"])
            highlight_indices = [
                i for i, tid in enumerate(token_ids)
                if tid in witness_token_ids
            ]
            
            color = 'blue' if bar["dim"] == 0 else 'red'
            ax.scatter(
                coords[highlight_indices, 0],
                coords[highlight_indices, 1],
                c=color, s=100, alpha=0.8, edgecolors='black'
            )
            
            # Add labels
            for idx in highlight_indices:
                tid = token_ids[idx]
                text = tokens[tid]["text"]
                ax.annotate(text, (coords[idx, 0], coords[idx, 1]), fontsize=8)
            
            # Draw cycle edges for H₁
            if bar["dim"] == 1:
                cycle = bar["witness"]["cycle"]
                tid_to_idx = {tid: i for i, tid in enumerate(token_ids)}
                
                for simplex in cycle["simplices"]:
                    if len(simplex) == 2:
                        idx1 = tid_to_idx.get(simplex[0])
                        idx2 = tid_to_idx.get(simplex[1])
                        if idx1 is not None and idx2 is not None:
                            ax.plot(
                                [coords[idx1, 0], coords[idx2, 0]],
                                [coords[idx1, 1], coords[idx2, 1]],
                                'r-', linewidth=2, alpha=0.7
                            )
    
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title)
    
    return ax


# =============================================================================
# Export for Visualization Tools
# =============================================================================

def export_for_gephi(
    diagram: WitnessedDiagram,
    nodes_path: str,
    edges_path: str
) -> None:
    """
    Export bar nerve as Gephi-compatible CSV files.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        The diagram.
    nodes_path : str
        Path for nodes CSV.
    edges_path : str
        Path for edges CSV.
    """
    import csv
    
    bars = diagram["bars"]
    edges = compute_bar_nerve_edges(diagram)
    
    # Write nodes
    with open(nodes_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label', 'Dim', 'Persistence', 'WitnessSize'])
        for bar in bars:
            label = ", ".join(bar["witness"]["tokens"]["surface"][:3])
            writer.writerow([
                bar["id"],
                label,
                bar["dim"],
                bar["persistence"],
                len(bar["witness"]["tokens"]["ids"])
            ])
    
    # Write edges
    with open(edges_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Type'])
        for b1, b2 in edges:
            writer.writerow([b1, b2, 'Undirected'])
    
    print(f"Exported {len(bars)} nodes to {nodes_path}")
    print(f"Exported {len(edges)} edges to {edges_path}")
