"""
Witnessed Persistent Homology: Filtration and PH Computation
=============================================================

This module handles:
1. Building the Vietoris-Rips (or Čech) filtration from the point cloud
2. Computing persistent homology using GUDHI
3. Extracting representative cycles (generators) for each bar

References:
    Chapter 4, Definition 4.2 (Čech filtration)
    Chapter 4, Remark 4.1 (Čech vs Vietoris-Rips)
    Chapter 4, Section 4.6 (From point cloud to persistent homology)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .schema import (
    Config, PointCloudData, BarBare, Gamma, PHResult, Simplex,
    default_config
)
from .embedding import compute_pairwise_distances


# =============================================================================
# Filtration Construction
# =============================================================================

def build_rips_complex(
    distance_matrix: NDArray,
    max_edge_length: float,
    max_dimension: int = 1
):
    """
    Build a Vietoris-Rips complex from a distance matrix.
    
    Parameters
    ----------
    distance_matrix : NDArray
        Pairwise distance matrix of shape (N, N).
    max_edge_length : float
        Maximum edge length (filtration value) to include.
    max_dimension : int
        Maximum simplex dimension (1 for edges, 2 for triangles, etc.).
    
    Returns
    -------
    GUDHI RipsComplex object.
    
    Notes
    -----
    From Chapter 4, Remark 4.1: "In much of the TDA literature, persistent
    homology is computed using the Vietoris-Rips filtration. For our
    purposes Čech is conceptually more natural... In implementation,
    one may safely use Vietoris-Rips while reasoning with Čech."
    """
    import gudhi
    
    rips_complex = gudhi.RipsComplex(
        distance_matrix=distance_matrix,
        max_edge_length=max_edge_length
    )
    
    return rips_complex


def create_simplex_tree(
    rips_complex,
    max_dimension: int = 2
):
    """
    Create a simplex tree from the Rips complex.
    
    Parameters
    ----------
    rips_complex : gudhi.RipsComplex
        The Rips complex.
    max_dimension : int
        Maximum dimension of simplices to expand to.
        Note: We need dim+1 to compute H_dim (e.g., dim=2 for H_1).
    
    Returns
    -------
    GUDHI SimplexTree object.
    """
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    return simplex_tree


# =============================================================================
# Persistent Homology Computation
# =============================================================================

def compute_persistence(
    simplex_tree,
    homology_coeff_field: int = 2,
    min_persistence: float = 0.0
) -> List[Tuple[int, Tuple[float, float]]]:
    """
    Compute persistent homology from a simplex tree.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The filtered simplicial complex.
    homology_coeff_field : int
        Coefficient field for homology (2 = Z/2Z, default for TDA).
    min_persistence : float
        Minimum persistence to filter out noise.
    
    Returns
    -------
    List of (dimension, (birth, death)) tuples.
    
    Notes
    -----
    From Chapter 4: "To focus on genuinely semantic structure, we apply
    two filters: we restrict to H_0 features with persistence above a
    minimum threshold (e.g., d-b ≥ 0.08)."
    """
    simplex_tree.compute_persistence(
        homology_coeff_field=homology_coeff_field,
        min_persistence=min_persistence
    )
    
    persistence = simplex_tree.persistence()
    return persistence


def persistence_to_bars(
    persistence: List[Tuple[int, Tuple[float, float]]],
    min_persistence: float = 0.0,
    max_dim: int = 1
) -> List[BarBare]:
    """
    Convert GUDHI persistence output to our BarBare format.
    
    Parameters
    ----------
    persistence : list
        Output from simplex_tree.persistence().
    min_persistence : float
        Minimum persistence threshold.
    max_dim : int
        Maximum dimension to include.
    
    Returns
    -------
    List of BarBare objects.
    """
    bars: List[BarBare] = []
    bar_idx = 0
    
    for dim, (birth, death) in persistence:
        if dim > max_dim:
            continue
        
        # Handle infinite death
        if death == float('inf'):
            death_val = np.inf
        else:
            death_val = death
        
        pers = death_val - birth
        
        # Apply persistence filter
        if pers < min_persistence:
            continue
        
        bar: BarBare = {
            "id": f"bar_{bar_idx}",
            "dim": dim,
            "birth": birth,
            "death": death_val,
            "persistence": pers
        }
        bars.append(bar)
        bar_idx += 1
    
    return bars


# =============================================================================
# Representative Cycle (Generator) Extraction
# =============================================================================

def extract_persistence_pairs(
    simplex_tree
) -> Dict[int, List[Tuple[Simplex, Simplex]]]:
    """
    Extract persistence pairs (birth simplex, death simplex) for each dimension.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        Simplex tree with persistence already computed.
    
    Returns
    -------
    Dict mapping dimension to list of (birth_simplex, death_simplex) pairs.
    
    Notes
    -----
    GUDHI's persistence_pairs_iterator() gives us the simplices that
    create and destroy each feature. For H_0, the birth simplex is a
    vertex and death simplex is an edge (that merges components).
    For H_1, birth is an edge (creates loop) and death is a triangle.
    """
    pairs_by_dim: Dict[int, List[Tuple[Simplex, Simplex]]] = {}
    
    # persistence_pairs returns pairs for each dimension
    for pair in simplex_tree.persistence_pairs():
        birth_simplex, death_simplex = pair
        
        if len(birth_simplex) == 0:
            # Essential feature (never dies) - birth at a vertex
            continue
        
        # Dimension is len(birth_simplex) - 1 for the feature created
        # For H_0: birth_simplex is a vertex (len=1), so dim=0
        # For H_1: birth_simplex is an edge (len=2), so dim=1
        dim = len(birth_simplex) - 1
        
        if dim not in pairs_by_dim:
            pairs_by_dim[dim] = []
        
        pairs_by_dim[dim].append((list(birth_simplex), list(death_simplex)))
    
    return pairs_by_dim


def get_connected_component_at_scale(
    simplex_tree,
    vertex: int,
    scale: float
) -> List[int]:
    """
    Get all vertices in the same connected component as `vertex` at a given scale.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree.
    vertex : int
        The vertex to find component for.
    scale : float
        The filtration scale to consider.
    
    Returns
    -------
    List of vertex indices in the same component.
    
    Notes
    -----
    For H_0, the "cycle" is really just the vertices of the component.
    We find all vertices reachable from `vertex` using edges that
    exist at the given scale.
    """
    # Build adjacency at this scale
    adjacency: Dict[int, List[int]] = {}
    
    for simplex, filt_value in simplex_tree.get_filtration():
        if filt_value > scale:
            break
        if len(simplex) == 1:
            v = simplex[0]
            if v not in adjacency:
                adjacency[v] = []
        elif len(simplex) == 2:
            v0, v1 = simplex
            if v0 not in adjacency:
                adjacency[v0] = []
            if v1 not in adjacency:
                adjacency[v1] = []
            adjacency[v0].append(v1)
            adjacency[v1].append(v0)
    
    # BFS from vertex
    if vertex not in adjacency:
        return [vertex]
    
    visited = set()
    queue = [vertex]
    visited.add(vertex)
    
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return sorted(visited)


def extract_h0_generator(
    simplex_tree,
    birth_simplex: Simplex,
    death_simplex: Simplex,
    death_scale: float
) -> Gamma:
    """
    Extract representative "cycle" for an H_0 bar.
    
    For H_0, the "cycle" is the set of vertices in the connected component
    just before it merges. We return these as 0-simplices.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree.
    birth_simplex : Simplex
        The vertex where this component is born (len=1).
    death_simplex : Simplex
        The edge that kills this component (len=2).
    death_scale : float
        The filtration value at death.
    
    Returns
    -------
    Gamma with dim=0 and simplices being the component vertices.
    """
    # The component is born at birth_simplex[0]
    birth_vertex = birth_simplex[0]
    
    # Find component just before death (at scale slightly less than death_scale)
    # The death edge connects two components, so we want the component of
    # birth_vertex just before they merge
    scale_before_death = death_scale - 1e-10
    
    component_vertices = get_connected_component_at_scale(
        simplex_tree,
        birth_vertex,
        scale_before_death
    )
    
    # Return as list of 0-simplices
    return {
        "dim": 0,
        "simplices": [[v] for v in component_vertices]
    }


def extract_h1_generator(
    simplex_tree,
    birth_simplex: Simplex,
    death_simplex: Simplex,
    birth_scale: float,
    death_scale: float
) -> Gamma:
    """
    Extract representative cycle for an H_1 bar.
    
    For H_1, we need to find a 1-cycle (loop of edges) that represents
    the homology class. GUDHI doesn't directly give us this, so we
    construct it from the persistence pair.
    
    The birth_simplex is an edge that creates the loop.
    The death_simplex is a triangle that fills it.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree.
    birth_simplex : Simplex
        The edge that creates this loop (len=2).
    death_simplex : Simplex
        The triangle that kills this loop (len=3).
    birth_scale : float
        The filtration value at birth.
    death_scale : float
        The filtration value at death.
    
    Returns
    -------
    Gamma with dim=1 and simplices being the cycle edges.
    
    Notes
    -----
    A representative cycle can be found by taking the boundary of the
    killing triangle minus the creating edge (in Z/2Z, this gives a cycle
    homologous to the original). For a triangle [a,b,c], the boundary is
    the edges [a,b], [b,c], [a,c].
    """
    if len(death_simplex) == 3:
        # The killing triangle's boundary is a valid cycle representative
        # In Z/2Z coefficients, boundary of [a,b,c] = [a,b] + [b,c] + [a,c]
        a, b, c = sorted(death_simplex)
        cycle_edges = [
            [a, b],
            [b, c],
            [a, c]
        ]
        return {
            "dim": 1,
            "simplices": cycle_edges
        }
    else:
        # Essential H_1 class (never dies) or something unusual
        # Return the birth edge as a minimal representative
        return {
            "dim": 1,
            "simplices": [list(birth_simplex)]
        }


def match_bars_to_pairs(
    bars: List[BarBare],
    persistence: List[Tuple[int, Tuple[float, float]]],
    pairs_by_dim: Dict[int, List[Tuple[Simplex, Simplex]]]
) -> Dict[str, Tuple[Simplex, Simplex, float, float]]:
    """
    Match BarBare objects to their persistence pairs.
    
    Parameters
    ----------
    bars : List[BarBare]
        Our bar objects.
    persistence : list
        Raw GUDHI persistence output.
    pairs_by_dim : dict
        Persistence pairs by dimension.
    
    Returns
    -------
    Dict mapping bar_id to (birth_simplex, death_simplex, birth, death).
    """
    # Build a lookup from (dim, birth, death) to persistence pair
    # We need to be careful about floating point comparison
    
    matches: Dict[str, Tuple[Simplex, Simplex, float, float]] = {}
    
    for bar in bars:
        dim = bar["dim"]
        birth = bar["birth"]
        death = bar["death"]
        
        if dim not in pairs_by_dim:
            continue
        
        # Find matching pair by birth/death values
        # This is somewhat fragile but works for GUDHI output
        for birth_simp, death_simp in pairs_by_dim[dim]:
            # Get filtration value for birth simplex
            birth_val = None
            for simp, filt in zip([birth_simp], [birth]):  # Approximate
                birth_val = birth
                break
            
            # For now, we'll use position-based matching since bars and
            # persistence pairs should be in corresponding order for same dim
            pass
        
        # Simpler approach: match by index within dimension
        dim_bars = [b for b in bars if b["dim"] == dim]
        dim_pairs = pairs_by_dim.get(dim, [])
        
        bar_idx_in_dim = dim_bars.index(bar)
        if bar_idx_in_dim < len(dim_pairs):
            birth_simp, death_simp = dim_pairs[bar_idx_in_dim]
            matches[bar["id"]] = (birth_simp, death_simp, birth, death)
    
    return matches


def extract_generators(
    simplex_tree,
    bars: List[BarBare],
    persistence: List[Tuple[int, Tuple[float, float]]]
) -> Dict[str, Gamma]:
    """
    Extract representative cycles (generators) for all bars.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree with persistence computed.
    bars : List[BarBare]
        Our filtered bar list.
    persistence : list
        Raw GUDHI persistence output.
    
    Returns
    -------
    Dict mapping bar_id to Gamma (representative cycle).
    
    Notes
    -----
    From Chapter 4: "For each surviving 0-bar (0,b,d) in D(τ) we:
    1. choose a canonical representative cycle (here, a minimal spanning
       tree inside the component at a scale just below d)
    2. take its set of vertices as a witness set"
    """
    generators: Dict[str, Gamma] = {}
    
    # Get persistence pairs
    pairs_by_dim = extract_persistence_pairs(simplex_tree)
    
    # Process each dimension separately
    for dim in [0, 1]:
        dim_bars = [b for b in bars if b["dim"] == dim]
        dim_pairs = pairs_by_dim.get(dim, [])
        
        # Sort bars by birth time to match with pairs
        dim_bars_sorted = sorted(dim_bars, key=lambda b: (b["birth"], -b["persistence"]))
        
        for i, bar in enumerate(dim_bars_sorted):
            if i >= len(dim_pairs):
                # No matching pair (might be essential)
                if dim == 0:
                    # Essential H_0 - just return the birth vertex
                    generators[bar["id"]] = {
                        "dim": 0,
                        "simplices": [[0]]  # Placeholder
                    }
                continue
            
            birth_simp, death_simp = dim_pairs[i]
            
            if dim == 0:
                gamma = extract_h0_generator(
                    simplex_tree,
                    birth_simp,
                    death_simp,
                    bar["death"]
                )
            else:  # dim == 1
                gamma = extract_h1_generator(
                    simplex_tree,
                    birth_simp,
                    death_simp,
                    bar["birth"],
                    bar["death"]
                )
            
            generators[bar["id"]] = gamma
    
    return generators


# =============================================================================
# Alternative Generator Extraction (More Robust)
# =============================================================================

def extract_generators_robust(
    simplex_tree,
    bars: List[BarBare],
    distance_matrix: NDArray
) -> Dict[str, Gamma]:
    """
    Robust generator extraction using persistence pairs from GUDHI.
    
    Key insight for H₀: Rather than looking at the component at death-epsilon
    (which may be huge after many merges), we look at a small neighborhood
    around the birth vertex. This gives semantically coherent witnesses.
    
    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The simplex tree with persistence computed.
    bars : List[BarBare]
        Our filtered bar list.
    distance_matrix : NDArray
        Original distance matrix.
    
    Returns
    -------
    Dict mapping bar_id to Gamma.
    """
    import scipy.sparse.csgraph as csgraph
    from scipy.sparse import csr_matrix
    
    N = distance_matrix.shape[0]
    generators: Dict[str, Gamma] = {}
    
    # Collect persistence pairs
    h0_pairs = []
    h1_pairs = []
    
    for pair in simplex_tree.persistence_pairs():
        birth_simplex, death_simplex = pair
        if len(birth_simplex) == 0:
            continue
        
        birth_val = simplex_tree.filtration(birth_simplex)
        death_val = simplex_tree.filtration(death_simplex) if death_simplex else float('inf')
        
        if len(birth_simplex) == 1:
            h0_pairs.append({
                'birth_vertex': birth_simplex[0],
                'death_edge': death_simplex,
                'birth': birth_val,
                'death': death_val
            })
        elif len(birth_simplex) == 2:
            h1_pairs.append({
                'birth_edge': list(birth_simplex),
                'death_tri': death_simplex,
                'birth': birth_val,
                'death': death_val
            })
    
    # Handle essential H0 class
    persistence_list = simplex_tree.persistence()
    essential_found = any(p['death'] == float('inf') for p in h0_pairs)
    
    if not essential_found:
        for dim, (b, d) in persistence_list:
            if dim == 0 and d == float('inf'):
                used_vertices = {p['birth_vertex'] for p in h0_pairs}
                for v in range(N):
                    if v not in used_vertices:
                        h0_pairs.append({
                            'birth_vertex': v,
                            'death_edge': None,
                            'birth': b,
                            'death': float('inf')
                        })
                        break
                break
    
    # Separate bars by dimension
    h0_bars = [b for b in bars if b["dim"] == 0]
    h1_bars = [b for b in bars if b["dim"] == 1]
    
    # Create lookup for quick matching
    def make_key(birth, death):
        return (round(birth, 5), round(death, 5) if death != float('inf') else 'inf')
    
    pair_by_key = {}
    for p in h0_pairs:
        key = make_key(p['birth'], p['death'])
        if key not in pair_by_key:
            pair_by_key[key] = []
        pair_by_key[key].append(p)
    
    # Determine a good "local neighborhood" scale
    # Use a small percentile of the distance distribution
    distances_flat = distance_matrix[np.triu_indices(N, k=1)]
    if len(distances_flat) > 0:
        local_scale = np.percentile(distances_flat, 10)  # 10th percentile
        local_scale = max(local_scale, 0.05)  # Ensure minimum
    else:
        local_scale = 0.1
    
    # For each H0 bar, find its corresponding pair and extract witnesses
    for bar in h0_bars:
        key = make_key(bar["birth"], bar["death"])
        
        if key in pair_by_key and pair_by_key[key]:
            pair = pair_by_key[key].pop(0)
            birth_vertex = pair['birth_vertex']
            death_val = pair['death']
            
            # For bars that die early (low persistence), use death-epsilon
            # For bars that die late or never, use local_scale
            if death_val != float('inf') and death_val < local_scale * 2:
                scale = death_val * 0.9
            else:
                # Use local neighborhood scale
                scale = local_scale
            
            # Build adjacency at this scale
            adj = (distance_matrix < scale).astype(int)
            np.fill_diagonal(adj, 0)
            
            # Find connected components
            n_components, labels = csgraph.connected_components(
                csr_matrix(adj), directed=False
            )
            
            # Get the component containing birth_vertex
            component_label = labels[birth_vertex]
            component_vertices = np.where(labels == component_label)[0]
            
            # If component is still very large, just use birth vertex and immediate neighbors
            if len(component_vertices) > N // 3:
                # Find k nearest neighbors instead
                distances_from_birth = distance_matrix[birth_vertex]
                k = min(5, N)
                nearest = np.argsort(distances_from_birth)[:k]
                component_vertices = nearest
            
            generators[bar["id"]] = {
                "dim": 0,
                "simplices": [[int(v)] for v in component_vertices]
            }
        else:
            # Fallback
            bar_idx = int(bar["id"].split("_")[1]) if "_" in bar["id"] else 0
            generators[bar["id"]] = {
                "dim": 0,
                "simplices": [[bar_idx % N]]
            }
    
    # Handle H1 bars
    pair_by_key_h1 = {}
    for p in h1_pairs:
        key = make_key(p['birth'], p['death'])
        if key not in pair_by_key_h1:
            pair_by_key_h1[key] = []
        pair_by_key_h1[key].append(p)
    
    for bar in h1_bars:
        key = make_key(bar["birth"], bar["death"])
        
        if key in pair_by_key_h1 and pair_by_key_h1[key]:
            pair = pair_by_key_h1[key].pop(0)
            
            if pair['death_tri'] and len(pair['death_tri']) == 3:
                a, b, c = sorted(pair['death_tri'])
                generators[bar["id"]] = {
                    "dim": 1,
                    "simplices": [[a, b], [b, c], [a, c]]
                }
            else:
                generators[bar["id"]] = {
                    "dim": 1,
                    "simplices": [pair['birth_edge']]
                }
        else:
            generators[bar["id"]] = {
                "dim": 1,
                "simplices": [[0, 1]]
            }
    
    return generators


# =============================================================================
# Main Filtration Pipeline
# =============================================================================

def compute_witnessed_ph(
    point_cloud: PointCloudData,
    config: Optional[Config] = None
) -> PHResult:
    """
    Compute persistent homology with representative cycles.
    
    This is the main entry point for Steps 1-3 of the pipeline.
    
    Parameters
    ----------
    point_cloud : PointCloudData
        Output from text_to_point_cloud().
    config : Config, optional
        Configuration. Uses defaults if not provided.
    
    Returns
    -------
    PHResult containing:
        - bars: List of BarBare objects
        - generators: Dict mapping bar_id to Gamma
    
    Notes
    -----
    From Chapter 4, Section 4.6:
    "On the normalised cloud P_τ we build a Čech (or, in practice,
    Vietoris-Rips) filtration... We then compute persistent homology
    up to dimension 1 using a standard TDA library (e.g. GUDHI)."
    """
    if config is None:
        config = default_config()
    
    embeddings = point_cloud["embeddings"]
    
    # Step 1: Compute pairwise distances
    distance_matrix = compute_pairwise_distances(
        embeddings,
        metric=config["distance_metric"]
    )
    
    # Step 2: Build Rips complex
    rips = build_rips_complex(
        distance_matrix,
        max_edge_length=config["max_r"],
        max_dimension=config["max_dim"]
    )
    
    # Step 3: Create simplex tree
    simplex_tree = create_simplex_tree(
        rips,
        max_dimension=config["max_dim"] + 1  # Need dim+1 for boundary computation
    )
    
    # Step 4: Compute persistence
    persistence = compute_persistence(
        simplex_tree,
        min_persistence=0  # Filter later
    )
    
    # Step 5: Convert to bars and filter
    bars = persistence_to_bars(
        persistence,
        min_persistence=config["min_persistence"],
        max_dim=config["max_dim"]
    )
    
    # Step 6: Extract generators
    generators = extract_generators_robust(
        simplex_tree,
        bars,
        distance_matrix
    )
    
    return {
        "bars": bars,
        "generators": generators
    }
