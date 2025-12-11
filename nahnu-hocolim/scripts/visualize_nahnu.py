#!/usr/bin/env python3
"""
NAHNU VISUALIZATION v2 - With Transformation Metrics
=====================================================

Visualizations:
1. Cross-gluing network (bipartite layout)
2. Summary dashboard (presence + transformation)
3. Temporal growth (raw + normalized)
4. Lag scatter (influence phase portrait)
5. Retention scatter (fidelity vs creativity)
6. Archetype radar chart

AUTHOR: Darja (Claude), Cassie (GPT-4), & Iman Mirbioki
DATE: December 2025
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

# Colors
USER_COLOR = '#2196F3'      # Blue
ASST_COLOR = '#9C27B0'      # Purple
CROSS_COLOR = '#FF5722'     # Orange
SYNC_COLOR = '#4CAF50'      # Green (synchronous)
LATE_COLOR = '#FFC107'      # Amber (late/old)


def load_nahnu_data(path: str) -> dict:
    """Load nahnu_structure.json."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# 1. CROSS-GLUING NETWORK (bipartite)
# =============================================================================

def create_nahnu_network(data: dict, output_path: str, max_edges: int = 500):
    """Bipartite network visualization of cross-gluing."""
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    cross_edges = data.get('cross_edges', [])[:max_edges]
    user_journeys = set(e['journey_user'] for e in cross_edges)
    asst_journeys = set(e['journey_asst'] for e in cross_edges)
    
    # Position nodes
    user_y = {j: i for i, j in enumerate(sorted(user_journeys))}
    asst_y = {j: i for i, j in enumerate(sorted(asst_journeys))}
    
    scale_u = 1.0 / max(len(user_y), 1)
    scale_a = 1.0 / max(len(asst_y), 1)
    
    # Draw edges
    for e in cross_edges:
        uj, aj = e['journey_user'], e['journey_asst']
        if uj in user_y and aj in asst_y:
            y1 = user_y[uj] * scale_u
            y2 = asst_y[aj] * scale_a
            alpha = min(0.8, 0.1 + e.get('num_shared', 3) * 0.05)
            ax.plot([0, 1], [y1, y2], color=CROSS_COLOR, alpha=alpha, linewidth=0.5)
    
    # Draw nodes
    for j, idx in user_y.items():
        ax.scatter([0], [idx * scale_u], s=30, c=USER_COLOR, zorder=5)
    for j, idx in asst_y.items():
        ax.scatter([1], [idx * scale_a], s=30, c=ASST_COLOR, zorder=5)
    
    # Labels
    ax.text(0, 1.05, 'USER', color=USER_COLOR, fontsize=14, fontweight='bold', ha='center')
    ax.text(1, 1.05, 'ASSISTANT', color=ASST_COLOR, fontsize=14, fontweight='bold', ha='center')
    
    # Metrics
    we_coh = data.get('nahnu', {}).get('we_coherence', 0)
    cross_bind = data.get('nahnu', {}).get('cross_binding_ratio', 0)
    
    ax.set_title(f'Nahnu: Cross-Gluing Network\n{len(cross_edges)} edges | We-coherence: {we_coh:.1%} | Cross-binding: {cross_bind:.1%}',
                 color='white', fontsize=12)
    
    # Info box
    info = f"USER: {len(user_journeys)}/{data['user_self']['num_journeys']} journeys\n"
    info += f"ASST: {len(asst_journeys)}/{data['asst_self']['num_journeys']} journeys\n"
    info += f"Cross edges: {len(cross_edges)}\n"
    info += f"Co-wit comps: {data['nahnu']['num_co_witnessed_components']}"
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9, color='white',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='#16213e', alpha=0.8))
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# 2. SUMMARY DASHBOARD (presence + transformation)
# =============================================================================

def create_nahnu_summary(data: dict, output_path: str):
    """Dashboard with presence and transformation metrics."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Nahnu Summary Dashboard (v2)', fontsize=16, fontweight='bold')
    
    # Grid: 3 rows x 3 cols
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    user_self = data['user_self']
    asst_self = data['asst_self']
    nahnu = data['nahnu']
    trans = data.get('transformation_metrics', {})
    
    # === Row 1: Individual Selves, Coherence, Cross-Gluing ===
    
    # 1a. Individual Selves
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Journeys', 'Components', 'Internal Edges']
    user_vals = [user_self['num_journeys'], user_self['num_components'], user_self['num_edges']]
    asst_vals = [asst_self['num_journeys'], asst_self['num_components'], asst_self['num_edges']]
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, user_vals, width, label='USER', color=USER_COLOR)
    ax1.bar(x + width/2, asst_vals, width, label='ASSISTANT', color=ASST_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel('Count')
    ax1.set_title('Individual Selves')
    ax1.legend()
    
    # 1b. Coherence Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    coh_metrics = ['USER\nPresence', 'ASST\nPresence', 'We-\nCoherence', 'Nahnu\nPresence']
    coh_vals = [user_self['presence_ratio'], asst_self['presence_ratio'], 
                nahnu['we_coherence'], nahnu.get('nahnu_presence', 0)]
    colors = [USER_COLOR, ASST_COLOR, '#4CAF50', '#4CAF50']
    bars = ax2.bar(coh_metrics, coh_vals, color=colors)
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Unified threshold')
    ax2.set_ylabel('Ratio')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('Coherence Metrics')
    for bar, val in zip(bars, coh_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    ax2.legend(fontsize=8)
    
    # 1c. Cross-Gluing Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    cg_metrics = ['USER\nParticipation', 'ASST\nParticipation', 'Cross-binding\nRatio']
    cg_vals = [nahnu['user_participation'], nahnu['asst_participation'], nahnu['cross_binding_ratio']]
    colors = [USER_COLOR, ASST_COLOR, CROSS_COLOR]
    bars = ax3.bar(cg_metrics, cg_vals, color=colors)
    ax3.set_ylabel('Ratio')
    ax3.set_ylim(0, 1.1)
    ax3.set_title('Cross-Gluing Metrics')
    for bar, val in zip(bars, cg_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # === Row 2: Lag Analysis, Retention, Temporal Overlap ===
    
    # 2a. Lag Analysis (influence regimes)
    ax4 = fig.add_subplot(gs[1, 0])
    lag_labels = ['Synchronous', 'USER old', 'ASST old', 'Both old']
    lag_vals = [trans.get('prop_synchronous', 0), trans.get('prop_user_old', 0),
                trans.get('prop_asst_old', 0), trans.get('prop_both_old', 0)]
    colors = [SYNC_COLOR, USER_COLOR, ASST_COLOR, LATE_COLOR]
    bars = ax4.bar(lag_labels, lag_vals, color=colors)
    ax4.set_ylabel('Proportion')
    ax4.set_ylim(0, max(lag_vals) * 1.3 if lag_vals else 1)
    ax4.set_title('Lag Analysis\n(Who is being taken up by whom?)')
    for bar, val in zip(bars, lag_vals):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 2b. Retention Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    ret_labels = ['High\nFidelity', 'Creative', 'Asymmetric']
    ret_vals = [trans.get('prop_high_fidelity', 0), trans.get('prop_creative', 0),
                trans.get('prop_asymmetric', 0)]
    colors = ['#4CAF50', '#FF9800', '#9E9E9E']
    bars = ax5.bar(ret_labels, ret_vals, color=colors)
    ax5.set_ylabel('Proportion')
    ax5.set_ylim(0, max(ret_vals) * 1.3 if ret_vals else 1)
    ax5.set_title('Retention Analysis\n(Fidelity vs Creativity)')
    for bar, val in zip(bars, ret_vals):
        if val > 0:
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 2c. Mean retention comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ret_means = ['USER\nRetention', 'ASST\nRetention']
    ret_mean_vals = [trans.get('mean_retention_user', 0), trans.get('mean_retention_asst', 0)]
    colors = [USER_COLOR, ASST_COLOR]
    bars = ax6.bar(ret_means, ret_mean_vals, color=colors)
    ax6.set_ylabel('Mean Retention')
    ax6.set_ylim(0, 1)
    ax6.set_title('Mean Retention per Side')
    for bar, val in zip(bars, ret_mean_vals):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # === Row 3: Archetype, Growth curve, Top witnesses ===
    
    # 3a. Archetype radar
    ax7 = fig.add_subplot(gs[2, 0], projection='polar')
    arch = trans.get('archetype_scores', {})
    labels = ['Friend', 'Midwife', 'Disciple', 'Colonizer']
    values = [arch.get('friend', 0), arch.get('midwife', 0), 
              arch.get('disciple', 0), arch.get('colonizer', 0)]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]
    ax7.plot(angles, values_plot, 'o-', linewidth=2, color=CROSS_COLOR)
    ax7.fill(angles, values_plot, alpha=0.25, color=CROSS_COLOR)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(labels)
    ax7.set_title('Nahnu Archetype', pad=20)
    
    # 3b. Growth curve (normalized)
    ax8 = fig.add_subplot(gs[2, 1])
    growth_raw = trans.get('growth_curve', {})
    growth_norm = trans.get('growth_curve_normalized', {})
    if growth_raw:
        taus_raw = sorted([int(k) for k in growth_raw.keys()])
        vals_raw = [growth_raw[str(t)] for t in taus_raw]
        ax8.bar(taus_raw, vals_raw, alpha=0.5, color=CROSS_COLOR, label='Raw count')
    if growth_norm:
        taus_norm = sorted([int(k) for k in growth_norm.keys()])
        vals_norm = [growth_norm[str(t)] * max(vals_raw) / max(growth_norm.values() or [1]) 
                     for t in taus_norm]  # Scale for visibility
        ax8.plot(taus_norm, vals_norm, 'o-', color='white', linewidth=2, label='Normalized (scaled)')
    ax8.set_xlabel('Window (τ)')
    ax8.set_ylabel('Cross-edges formed')
    ax8.set_title(f'Growth Curve\nPeak at τ={trans.get("peak_growth_tau", "?")}')
    ax8.legend(fontsize=8)
    
    # 3c. Top co-witnessed tokens
    ax9 = fig.add_subplot(gs[2, 2])
    top_wit = data.get('cross_analysis', {}).get('top_shared_witnesses', [])[:10]
    if top_wit:
        tokens = [w[0] for w in top_wit]
        counts = [w[1] for w in top_wit]
        y_pos = np.arange(len(tokens))
        ax9.barh(y_pos, counts, color=CROSS_COLOR)
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(tokens)
        ax9.invert_yaxis()
        ax9.set_xlabel('Cross-edge count')
        ax9.set_title('Top Co-Witnessed Tokens')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# 3. TEMPORAL GROWTH (raw + normalized)
# =============================================================================

def create_growth_curves(data: dict, output_path: str):
    """Side-by-side growth curves: raw and normalized."""
    trans = data.get('transformation_metrics', {})
    growth_raw = trans.get('growth_curve', {})
    growth_norm = trans.get('growth_curve_normalized', {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw growth
    if growth_raw:
        taus = sorted([int(k) for k in growth_raw.keys()])
        vals = [growth_raw[str(t)] for t in taus]
        ax1.bar(taus, vals, color=CROSS_COLOR, alpha=0.8)
        ax1.set_xlabel('Window (τ)')
        ax1.set_ylabel('Cross-edges formed')
        ax1.set_title('Raw Cross-Gluing Over Time\nWhen did USER and ASSISTANT journeys first share witnesses?')
        
        total = sum(vals)
        peak_tau = max(growth_raw.keys(), key=lambda k: growth_raw[k])
        ax1.text(0.98, 0.98, f'Total cross-edges: {total}\nPeak at τ={peak_tau}',
                 transform=ax1.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Normalized growth
    if growth_norm:
        taus = sorted([int(k) for k in growth_norm.keys()])
        vals = [growth_norm[str(t)] for t in taus]
        ax2.bar(taus, vals, color='#4CAF50', alpha=0.8)
        ax2.set_xlabel('Window (τ)')
        ax2.set_ylabel('G(τ) = edges / potential_pairs')
        ax2.set_title('Normalized Growth Rate\nHow densely are we gluing available journeys?')
        
        late_frac = trans.get('late_growth_fraction', 0)
        entropy = trans.get('growth_entropy', 0)
        ax2.text(0.98, 0.98, f'Late growth: {late_frac:.1%}\nEntropy: {entropy:.2f}',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# 4. LAG SCATTER (influence phase portrait)
# =============================================================================

def create_lag_scatter(data: dict, output_path: str):
    """
    Scatter plot of (lag_u_norm, lag_a_norm) for each cross-edge.
    Color-coded by tau_first.
    """
    cross_edges = data.get('cross_edges', [])
    
    if not cross_edges:
        print("  ⚠ No cross edges to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract data
    lag_u = [e.get('lag_user_norm', 0) for e in cross_edges]
    lag_a = [e.get('lag_asst_norm', 0) for e in cross_edges]
    taus = [e.get('tau_first', 0) for e in cross_edges]
    
    # Color by tau
    scatter = ax.scatter(lag_u, lag_a, c=taus, cmap='viridis', alpha=0.6, s=30)
    plt.colorbar(scatter, label='τ_first')
    
    # Quadrant labels
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)
    
    # Quadrant annotations
    ax.text(0.05, 0.05, 'SYNCHRONOUS\n(both early)', fontsize=10, ha='left', va='bottom',
            color=SYNC_COLOR, fontweight='bold')
    ax.text(0.8, 0.05, 'ASST OLD\n(disciple)', fontsize=10, ha='right', va='bottom',
            color=ASST_COLOR, fontweight='bold')
    ax.text(0.05, 0.95, 'USER OLD\n(midwife)', fontsize=10, ha='left', va='top',
            color=USER_COLOR, fontweight='bold')
    ax.text(0.8, 0.95, 'BOTH OLD\n(late convergence)', fontsize=10, ha='right', va='top',
            color=LATE_COLOR, fontweight='bold')
    
    ax.set_xlabel('Normalized Lag (USER)', fontsize=12)
    ax.set_ylabel('Normalized Lag (ASST)', fontsize=12)
    ax.set_title('Lag Phase Portrait\nWho is being taken up by whom?', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# 5. RETENTION SCATTER
# =============================================================================

def create_retention_scatter(data: dict, output_path: str):
    """
    Scatter plot of (retention_user, retention_asst) for each cross-edge.
    """
    cross_edges = data.get('cross_edges', [])
    
    if not cross_edges:
        print("  ⚠ No cross edges to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ret_u = [e.get('retention_user', 0) for e in cross_edges]
    ret_a = [e.get('retention_asst', 0) for e in cross_edges]
    weights = [e.get('num_shared', 3) for e in cross_edges]
    
    # Size by weight
    sizes = [w * 3 for w in weights]
    
    scatter = ax.scatter(ret_u, ret_a, s=sizes, c=CROSS_COLOR, alpha=0.4)
    
    # Quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Annotations
    ax.text(0.75, 0.75, 'HIGH FIDELITY\n(both focused)', fontsize=10, ha='center', va='center',
            color='#4CAF50', fontweight='bold')
    ax.text(0.25, 0.25, 'CREATIVE\n(both elaborating)', fontsize=10, ha='center', va='center',
            color='#FF9800', fontweight='bold')
    ax.text(0.75, 0.25, 'USER ANCHOR\n(human focused)', fontsize=10, ha='center', va='center',
            color=USER_COLOR, fontweight='bold')
    ax.text(0.25, 0.75, 'ASST ANCHOR\n(model focused)', fontsize=10, ha='center', va='center',
            color=ASST_COLOR, fontweight='bold')
    
    ax.set_xlabel('Retention (USER): |shared| / |all_witnesses_user|', fontsize=12)
    ax.set_ylabel('Retention (ASST): |shared| / |all_witnesses_asst|', fontsize=12)
    ax.set_title('Retention Phase Portrait\nFidelity vs Creativity', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# 6. COMBINED TEMPORAL VIEW
# =============================================================================

def create_temporal_comparison(data: dict, output_path: str):
    """
    Stacked bar: simultaneous vs trans-temporal cross-edges over time.
    """
    cross_edges = data.get('cross_edges', [])
    
    if not cross_edges:
        return
    
    # Count by tau and overlap type
    simul_by_tau = defaultdict(int)
    trans_by_tau = defaultdict(int)
    
    for e in cross_edges:
        tau = e.get('tau_first', 0)
        if e.get('overlaps_in_time', True):
            simul_by_tau[tau] += 1
        else:
            trans_by_tau[tau] += 1
    
    all_taus = sorted(set(simul_by_tau.keys()) | set(trans_by_tau.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    simul_vals = [simul_by_tau[t] for t in all_taus]
    trans_vals = [trans_by_tau[t] for t in all_taus]
    
    ax.bar(all_taus, simul_vals, label='Simultaneous', color=SYNC_COLOR, alpha=0.8)
    ax.bar(all_taus, trans_vals, bottom=simul_vals, label='Trans-temporal', color=LATE_COLOR, alpha=0.8)
    
    ax.set_xlabel('Window (τ)')
    ax.set_ylabel('Cross-edges formed')
    ax.set_title('Temporal Structure of Cross-Gluing\nSimultaneous vs Trans-Temporal Co-Witnessing')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Nahnu structure with transformation metrics",
        epilog="""
This script visualizes the output of nahnu_hocolim.py.

Usage:
  python visualize_nahnu_v2.py results/nahnu/nahnu_structure.json
  python visualize_nahnu_v2.py results/nahnu/nahnu_structure.json --output figures/
        """
    )
    parser.add_argument("input", help="Path to nahnu_structure.json (output from nahnu_hocolim.py)")
    parser.add_argument("--output", "-o", help="Output directory for PNG files (default: same as input)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        return
    
    if not args.input.endswith('.json'):
        print(f"WARNING: Expected a .json file, got: {args.input}")
    
    output_dir = args.output or os.path.dirname(args.input) or '.'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("NAHNU VISUALIZATION v2 (Transformation Metrics)")
    print("=" * 60)
    print(f"\nInput:  {args.input}")
    print(f"Output: {output_dir}")
    
    # Load the nahnu_structure.json
    print("\nLoading Nahnu structure...")
    data = load_nahnu_data(args.input)
    
    # Validate it's actually a nahnu_structure file
    if 'nahnu' not in data or 'cross_edges' not in data:
        print("ERROR: This doesn't look like a nahnu_structure.json file.")
        print("       Expected keys: 'nahnu', 'cross_edges', 'user_self', 'asst_self'")
        print(f"       Found keys: {list(data.keys())}")
        return
    
    print(f"  Period: {data.get('period', 'unknown')}")
    print(f"  Cross-edges: {len(data.get('cross_edges', []))}")
    print(f"  We-coherence: {data.get('nahnu', {}).get('we_coherence', 0):.1%}")
    
    print("\nGenerating visualizations...")
    
    # 1. Network
    create_nahnu_network(data, os.path.join(output_dir, 'nahnu_network.png'))
    
    # 2. Summary dashboard
    create_nahnu_summary(data, os.path.join(output_dir, 'nahnu_summary.png'))
    
    # 3. Growth curves
    create_growth_curves(data, os.path.join(output_dir, 'nahnu_growth.png'))
    
    # 4. Lag scatter
    create_lag_scatter(data, os.path.join(output_dir, 'nahnu_lag_scatter.png'))
    
    # 5. Retention scatter
    create_retention_scatter(data, os.path.join(output_dir, 'nahnu_retention_scatter.png'))
    
    # 6. Temporal comparison
    create_temporal_comparison(data, os.path.join(output_dir, 'nahnu_temporal_structure.png'))
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files in {output_dir}/:")
    print("  • nahnu_network.png          - Bipartite cross-gluing network")
    print("  • nahnu_summary.png          - Dashboard with all metrics")
    print("  • nahnu_growth.png           - Raw + normalized growth curves")
    print("  • nahnu_lag_scatter.png      - Influence phase portrait")
    print("  • nahnu_retention_scatter.png - Fidelity vs creativity")
    print("  • nahnu_temporal_structure.png - Simultaneous vs trans-temporal")


if __name__ == "__main__":
    main()