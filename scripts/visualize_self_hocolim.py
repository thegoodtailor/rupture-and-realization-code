#!/usr/bin/env python3
"""
Self-as-Hocolim Visualization Suite
====================================

Generates all visualizations for Chapter 5 demonstrator.

Usage:
    python visualize_self_hocolim.py                          # Uses self_structure.json in current dir
    python visualize_self_hocolim.py path/to/self_structure.json
    python visualize_self_hocolim.py self_structure.json --output results/viz

Outputs:
    1. timeline.png           - Journey lifespans with events (spawn/carry/drift/reentry)
    2. presence.png           - Self coherence over time bar chart
    3. network.png            - Static network graph (matplotlib)
    4. dashboard.html         - Interactive D3 dashboard with network + timeline + presence
    5. network_interactive.html - Dedicated beautiful D3 network with controls

Requirements:
    pip install matplotlib networkx numpy

Author: Generated for "Rupture and Realization" Chapter 5
"""

import json
import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any
import colorsys

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Static PNGs will be skipped.")
    print("  Install with: pip install matplotlib numpy")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Static network graph will be skipped.")
    print("  Install with: pip install networkx")


# =============================================================================
# UTILITIES
# =============================================================================

def get_tau_color(tau: int, max_tau: int, alpha: float = 1.0) -> tuple:
    """Get RGBA color for a given tau value - blue (early) to red (late)."""
    hue = 0.6 - (tau / max_tau) * 0.6  # 0.6 (blue) to 0.0 (red)
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return (*rgb, alpha)


def get_tau_color_hex(tau: int, max_tau: int) -> str:
    """Get hex color for a given tau."""
    rgb = get_tau_color(tau, max_tau, 1.0)[:3]
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


# =============================================================================
# 1. JOURNEY TIMELINE (PNG)
# =============================================================================

def create_timeline(data: dict, output_path: str, max_journeys: int = 80):
    """Create a timeline showing journey lifespans with events."""
    if not HAS_MATPLOTLIB:
        print(f"  ⊘ Skipping {output_path} (matplotlib not installed)")
        return
    
    journeys = sorted(data['journeys'], key=lambda j: (j['birth_tau'], -j['lifespan']))
    
    # Sample if too many - mix of long-running, recent, early, middle
    if len(journeys) > max_journeys:
        long_running = sorted(journeys, key=lambda j: -j['lifespan'])[:30]
        recent = [j for j in journeys if j['birth_tau'] >= 30][:25]
        early = [j for j in journeys if j['birth_tau'] <= 5][:15]
        middle = [j for j in journeys if 10 <= j['birth_tau'] <= 25][:15]
        
        selected = list({j['id']: j for j in (long_running + recent + early + middle)}.values())
        journeys = sorted(selected, key=lambda j: (j['birth_tau'], -j['lifespan']))[:max_journeys]
    
    fig, ax = plt.subplots(figsize=(16, max(10, len(journeys) * 0.14)))
    
    num_windows = len(data['presence_states'])
    window_ids = [ps['window_id'] for ps in data['presence_states']]
    
    for idx, journey in enumerate(journeys):
        y = len(journeys) - idx - 1
        
        # Draw each step
        for step in journey['steps']:
            tau = step['tau']
            event = step['event']
            color = get_tau_color(tau, num_windows)
            
            if event == 'spawn':
                ax.scatter([tau], [y], c=[color], s=60, marker='o', zorder=3, 
                          edgecolors='black', linewidths=0.5)
            elif event == 'carry':
                ax.scatter([tau], [y], c=[color], s=40, marker='s', zorder=3, edgecolors='none')
            elif event == 'drift':
                ax.scatter([tau], [y], c=[color], s=30, marker='D', zorder=3, 
                          edgecolors='none', alpha=0.7)
            elif event == 'reentry':
                ax.scatter([tau], [y], c=['gold'], s=80, marker='*', zorder=4, 
                          edgecolors='black', linewidths=0.5)
        
        # Connect with line
        taus = [step['tau'] for step in journey['steps']]
        if len(taus) > 1:
            ax.plot(taus, [y] * len(taus), color='gray', alpha=0.3, linewidth=1, zorder=1)
        
        # Label
        sig = journey['signature'][:25]
        ax.text(-1, y, sig, ha='right', va='center', fontsize=6, alpha=0.8)
    
    ax.set_xticks(range(num_windows))
    ax.set_xticklabels(window_ids, rotation=45, ha='right', fontsize=7)
    ax.set_xlim(-8, num_windows)
    ax.set_ylim(-1, len(journeys))
    ax.set_ylabel('Journeys', fontsize=10)
    ax.set_xlabel('Time Window', fontsize=10)
    ax.set_title(f'Self-as-Hocolim: Journey Timeline\n{len(journeys)} journeys across {num_windows} windows', 
                 fontsize=12)
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='blue', s=60, marker='o', label='Spawn', edgecolors='black', linewidths=0.5),
        plt.scatter([], [], c='green', s=40, marker='s', label='Carry'),
        plt.scatter([], [], c='orange', s=30, marker='D', label='Drift'),
        plt.scatter([], [], c='gold', s=80, marker='*', label='Reentry', edgecolors='black', linewidths=0.5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {output_path}")


# =============================================================================
# 2. PRESENCE CHART (PNG)
# =============================================================================

def create_presence_chart(data: dict, output_path: str):
    """Create a presence-over-time bar chart."""
    if not HAS_MATPLOTLIB:
        print(f"  ⊘ Skipping {output_path} (matplotlib not installed)")
        return
    
    states = data['presence_states']
    taus = [s['tau'] for s in states]
    presence = [s['presence'] for s in states]
    window_ids = [s['window_id'] for s in states]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    colors = []
    for p in presence:
        if p >= 0.8:
            colors.append('#2ecc71')  # Green - unified
        elif p >= 0.5:
            colors.append('#f39c12')  # Orange - partial
        else:
            colors.append('#e74c3c')  # Red - fragmented
    
    ax.bar(taus, presence, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=0.8, color='#2ecc71', linestyle='--', alpha=0.5, label='Unified threshold')
    ax.axhline(y=0.5, color='#f39c12', linestyle='--', alpha=0.5, label='Partial threshold')
    
    ax.set_xticks(taus)
    ax.set_xticklabels(window_ids, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Presence (fraction in largest component)', fontsize=10)
    ax.set_xlabel('Time Window', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Self Coherence Over Time\nOverall presence ratio: {data["presence_ratio"]:.3f}', fontsize=12)
    
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Unified (≥0.8)'),
        mpatches.Patch(color='#f39c12', label='Partial (0.5-0.8)'),
        mpatches.Patch(color='#e74c3c', label='Fragmented (<0.5)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {output_path}")


# =============================================================================
# 3. NETWORK GRAPH - STATIC (PNG)
# =============================================================================

def create_network_static(data: dict, output_path: str, max_edges: int = 400, max_nodes: int = 120):
    """Create a static network visualization using matplotlib."""
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        print(f"  ⊘ Skipping {output_path} (matplotlib/networkx not installed)")
        return
    
    journey_connectivity = defaultdict(int)
    for edge in data['gluing_edges']:
        journey_connectivity[edge['journey_a']] += 1
        journey_connectivity[edge['journey_b']] += 1
    
    top_journeys = sorted(journey_connectivity.keys(), 
                         key=lambda j: journey_connectivity[j], reverse=True)[:max_nodes]
    top_set = set(top_journeys)
    
    journey_lookup = {j['id']: j for j in data['journeys']}
    
    G = nx.Graph()
    
    for jid in top_journeys:
        j = journey_lookup[jid]
        G.add_node(jid, birth_tau=j['birth_tau'], lifespan=j['lifespan'], signature=j['signature'])
    
    relevant_edges = [e for e in data['gluing_edges'] 
                     if e['journey_a'] in top_set and e['journey_b'] in top_set]
    relevant_edges = sorted(relevant_edges, key=lambda e: -len(e['shared_witnesses']))[:max_edges]
    
    for edge in relevant_edges:
        G.add_edge(edge['journey_a'], edge['journey_b'], 
                  weight=len(edge['shared_witnesses']), cross_temporal=edge['cross_temporal'])
    
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    num_windows = len(data['presence_states'])
    
    edge_colors = []
    edge_widths = []
    for (u, v, d) in G.edges(data=True):
        edge_colors.append('#3498db' if d['cross_temporal'] else '#bdc3c7')
        edge_widths.append(0.3 + d['weight'] / 50)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color=edge_colors, width=edge_widths, ax=ax)
    
    node_colors = [get_tau_color(G.nodes[n]['birth_tau'], num_windows) for n in G.nodes()]
    node_sizes = [80 + G.nodes[n]['lifespan'] * 8 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          ax=ax, edgecolors='black', linewidths=0.5)
    
    high_degree = [n for n in G.nodes() if G.degree(n) > 15]
    labels = {n: journey_lookup[n]['signature'][:12] for n in high_degree}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    
    ax.set_title(f'Self-as-Hocolim: Gluing Network\n{G.number_of_nodes()} journeys, {G.number_of_edges()} edges', 
                 fontsize=12)
    ax.axis('off')
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm_r, norm=plt.Normalize(vmin=0, vmax=num_windows))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label='Birth τ (window)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {output_path}")


# =============================================================================
# 4. INTERACTIVE DASHBOARD (HTML)
# =============================================================================

def create_dashboard_html(data: dict, output_path: str):
    """Create an interactive D3.js dashboard with network, timeline, and presence."""
    
    journey_lookup = {j['id']: j for j in data['journeys']}
    
    journey_connectivity = defaultdict(int)
    for edge in data['gluing_edges']:
        journey_connectivity[edge['journey_a']] += 1
        journey_connectivity[edge['journey_b']] += 1
    
    top_journeys = sorted(journey_connectivity.keys(), 
                         key=lambda j: journey_connectivity[j], reverse=True)[:100]
    top_set = set(top_journeys)
    
    nodes = []
    for jid in top_journeys:
        j = journey_lookup[jid]
        nodes.append({
            'id': jid,
            'signature': j['signature'],
            'birth_tau': j['birth_tau'],
            'lifespan': j['lifespan'],
            'has_reentry': j['has_reentry'],
            'glued_with': j['glued_with_count'],
            'witnesses': j['all_witnesses'][:10]
        })
    
    relevant_edges = [e for e in data['gluing_edges'] 
                     if e['journey_a'] in top_set and e['journey_b'] in top_set]
    relevant_edges = sorted(relevant_edges, key=lambda e: -len(e['shared_witnesses']))[:300]
    
    links = [{'source': e['journey_a'], 'target': e['journey_b'], 
              'weight': len(e['shared_witnesses']), 'cross_temporal': e['cross_temporal']} 
             for e in relevant_edges]
    
    presence_data = data['presence_states']
    num_windows = len(presence_data)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Self-as-Hocolim Dashboard</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ text-align: center; color: #00d4ff; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 20px; }}
        .container {{ display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
        .panel {{ background: #16213e; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .panel h2 {{ margin-top: 0; color: #00d4ff; font-size: 14px; border-bottom: 1px solid #333; padding-bottom: 5px; }}
        #network {{ width: 700px; height: 550px; }}
        #timeline {{ width: 480px; height: 550px; overflow-y: auto; }}
        #presence {{ width: 100%; max-width: 1200px; height: 180px; }}
        .tooltip {{ position: absolute; background: #0f3460; border: 1px solid #00d4ff; border-radius: 5px;
                   padding: 10px; font-size: 12px; pointer-events: none; max-width: 300px; z-index: 1000; }}
        .tooltip strong {{ color: #00d4ff; }}
        .stats {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 20px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ font-size: 12px; color: #888; }}
        .legend {{ display: flex; gap: 15px; justify-content: center; margin-top: 10px; font-size: 11px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 50%; }}
    </style>
</head>
<body>
    <h1>Self-as-Hocolim</h1>
    <p class="subtitle">Computational Proxy for Chapter 5 — {data['period']}</p>
    
    <div class="stats">
        <div class="stat"><div class="stat-value">{data['num_journeys']}</div><div class="stat-label">Journeys</div></div>
        <div class="stat"><div class="stat-value">{len(data['gluing_edges']):,}</div><div class="stat-label">Gluing Edges</div></div>
        <div class="stat"><div class="stat-value">{data['num_components']}</div><div class="stat-label">Components</div></div>
        <div class="stat"><div class="stat-value">{data['presence_ratio']:.1%}</div><div class="stat-label">Presence</div></div>
    </div>
    
    <div class="container">
        <div class="panel">
            <h2>Gluing Network (Top 100 journeys)</h2>
            <div id="network"></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #00bfff;"></div> Early</div>
                <div class="legend-item"><div class="legend-color" style="background: #90EE90;"></div> Middle</div>
                <div class="legend-item"><div class="legend-color" style="background: #ff6b6b;"></div> Recent</div>
            </div>
        </div>
        <div class="panel">
            <h2>Journey Lifespans</h2>
            <div id="timeline"></div>
        </div>
    </div>
    
    <div class="container" style="margin-top: 20px;">
        <div class="panel">
            <h2>Presence Over Time</h2>
            <div id="presence"></div>
        </div>
    </div>
    
    <div class="tooltip" style="display: none;"></div>

<script>
const nodes = {json.dumps(nodes)};
const links = {json.dumps(links)};
const presenceData = {json.dumps(presence_data)};
const numWindows = {num_windows};

function getTauColor(tau) {{
    if (tau <= 10) return d3.interpolateBlues(0.4 + tau/20);
    if (tau <= 25) return d3.interpolateGreens(0.3 + (tau-10)/30);
    return d3.interpolateReds(0.4 + (tau-25)/20);
}}

const tooltip = d3.select('.tooltip');

// NETWORK
const networkSvg = d3.select('#network').append('svg').attr('width', 700).attr('height', 550);
const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(50).strength(0.1))
    .force('charge', d3.forceManyBody().strength(-100))
    .force('center', d3.forceCenter(350, 275))
    .force('collision', d3.forceCollide().radius(d => 5 + d.lifespan/3));

const link = networkSvg.append('g').selectAll('line').data(links).join('line')
    .attr('stroke', d => d.cross_temporal ? '#3498db' : '#555')
    .attr('stroke-opacity', d => 0.2 + d.weight/100)
    .attr('stroke-width', d => 0.5 + d.weight/30);

const node = networkSvg.append('g').selectAll('circle').data(nodes).join('circle')
    .attr('r', d => 4 + d.lifespan/4)
    .attr('fill', d => getTauColor(d.birth_tau))
    .attr('stroke', d => d.has_reentry ? 'gold' : '#333')
    .attr('stroke-width', d => d.has_reentry ? 2 : 0.5)
    .call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended))
    .on('mouseover', (event, d) => {{
        tooltip.style('display', 'block')
            .html(`<strong>${{d.signature}}</strong><br>Birth: τ=${{d.birth_tau}}<br>Lifespan: ${{d.lifespan}}<br>Reentry: ${{d.has_reentry ? 'Yes' : 'No'}}<br>Witnesses: ${{d.witnesses.slice(0,5).join(', ')}}...`);
    }})
    .on('mousemove', (event) => {{ tooltip.style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 10) + 'px'); }})
    .on('mouseout', () => tooltip.style('display', 'none'));

simulation.on('tick', () => {{
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
}});

function dragstarted(event) {{ if (!event.active) simulation.alphaTarget(0.3).restart(); event.subject.fx = event.subject.x; event.subject.fy = event.subject.y; }}
function dragged(event) {{ event.subject.fx = event.x; event.subject.fy = event.y; }}
function dragended(event) {{ if (!event.active) simulation.alphaTarget(0); event.subject.fx = null; event.subject.fy = null; }}

// PRESENCE
const pm = {{top: 20, right: 30, bottom: 50, left: 50}}, pw = 1150 - pm.left - pm.right, ph = 140 - pm.top - pm.bottom;
const presenceSvg = d3.select('#presence').append('svg').attr('width', pw + pm.left + pm.right).attr('height', ph + pm.top + pm.bottom).append('g').attr('transform', `translate(${{pm.left}},${{pm.top}})`);
const xScale = d3.scaleBand().domain(presenceData.map(d => d.window_id)).range([0, pw]).padding(0.1);
const yScale = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
presenceSvg.append('line').attr('x1', 0).attr('x2', pw).attr('y1', yScale(0.8)).attr('y2', yScale(0.8)).attr('stroke', '#2ecc71').attr('stroke-dasharray', '4,4').attr('opacity', 0.5);
presenceSvg.selectAll('rect').data(presenceData).join('rect')
    .attr('x', d => xScale(d.window_id)).attr('y', d => yScale(d.presence)).attr('width', xScale.bandwidth()).attr('height', d => ph - yScale(d.presence))
    .attr('fill', d => d.presence >= 0.8 ? '#2ecc71' : (d.presence >= 0.5 ? '#f39c12' : '#e74c3c'))
    .on('mouseover', (event, d) => {{ tooltip.style('display', 'block').html(`<strong>${{d.window_id}}</strong><br>Presence: ${{(d.presence * 100).toFixed(1)}}%<br>Components: ${{d.components}}`); }})
    .on('mousemove', (event) => {{ tooltip.style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 10) + 'px'); }})
    .on('mouseout', () => tooltip.style('display', 'none'));
presenceSvg.append('g').attr('transform', `translate(0,${{ph}})`).call(d3.axisBottom(xScale)).selectAll('text').attr('transform', 'rotate(-45)').style('text-anchor', 'end').style('font-size', '8px');
presenceSvg.append('g').call(d3.axisLeft(yScale).tickFormat(d => (d * 100) + '%'));

// TIMELINE
const timelineDiv = d3.select('#timeline');
const sortedNodes = [...nodes].sort((a, b) => a.birth_tau - b.birth_tau || b.lifespan - a.lifespan);
const tlHeight = sortedNodes.length * 8 + 40;
const tlSvg = timelineDiv.append('svg').attr('width', 460).attr('height', tlHeight);
const tlXScale = d3.scaleLinear().domain([0, numWindows]).range([100, 450]);
sortedNodes.forEach((j, idx) => {{
    const y = idx * 8 + 20;
    tlSvg.append('rect').attr('x', tlXScale(j.birth_tau)).attr('y', y - 2)
        .attr('width', Math.max(tlXScale(j.birth_tau + j.lifespan) - tlXScale(j.birth_tau), 2)).attr('height', 4)
        .attr('fill', getTauColor(j.birth_tau)).attr('rx', 2)
        .on('mouseover', (event) => {{ tooltip.style('display', 'block').html(`<strong>${{j.signature}}</strong><br>τ=${{j.birth_tau}} → τ=${{j.birth_tau + j.lifespan - 1}}<br>Reentry: ${{j.has_reentry ? '↺ Yes' : 'No'}}`); }})
        .on('mousemove', (event) => {{ tooltip.style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 10) + 'px'); }})
        .on('mouseout', () => tooltip.style('display', 'none'));
    if (j.has_reentry) tlSvg.append('text').attr('x', tlXScale(j.birth_tau + j.lifespan) + 3).attr('y', y + 2).text('↺').attr('fill', 'gold').attr('font-size', '8px');
    tlSvg.append('text').attr('x', 95).attr('y', y + 2).text(j.signature.slice(0, 12)).attr('fill', '#888').attr('font-size', '6px').attr('text-anchor', 'end');
}});
</script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓ {output_path}")


# =============================================================================
# 5. INTERACTIVE NETWORK (HTML) - Beautiful standalone
# =============================================================================

def create_network_interactive(data: dict, output_path: str):
    """Create a beautiful standalone D3 network with temporal heat and controls."""
    
    journey_lookup = {j['id']: j for j in data['journeys']}
    
    journey_connectivity = defaultdict(int)
    for edge in data['gluing_edges']:
        journey_connectivity[edge['journey_a']] += 1
        journey_connectivity[edge['journey_b']] += 1
    
    # Mix of highly connected, long-running, and recent
    top_connected = sorted(journey_connectivity.keys(), 
                          key=lambda j: journey_connectivity[j], reverse=True)[:80]
    long_running = sorted(data['journeys'], key=lambda j: -j['lifespan'])[:40]
    recent = [j for j in data['journeys'] if j['birth_tau'] >= 30][:30]
    
    selected_ids = list(set(top_connected) | set(j['id'] for j in long_running) | set(j['id'] for j in recent))[:150]
    top_set = set(selected_ids)
    
    nodes = []
    for jid in selected_ids:
        j = journey_lookup[jid]
        nodes.append({
            'id': jid,
            'signature': j['signature'],
            'birth_tau': j['birth_tau'],
            'lifespan': j['lifespan'],
            'has_reentry': j['has_reentry'],
            'glued_with': j['glued_with_count'],
            'witnesses': j['all_witnesses'][:15],
            'connectivity': journey_connectivity[jid]
        })
    
    relevant_edges = [e for e in data['gluing_edges'] 
                     if e['journey_a'] in top_set and e['journey_b'] in top_set]
    relevant_edges = sorted(relevant_edges, key=lambda e: (-int(e['cross_temporal']), -len(e['shared_witnesses'])))[:600]
    
    links = [{'source': e['journey_a'], 'target': e['journey_b'], 
              'weight': len(e['shared_witnesses']), 'cross_temporal': e['cross_temporal'],
              'witnesses': e['shared_witnesses'][:10]} for e in relevant_edges]
    
    presence_data = data['presence_states']
    num_windows = len(presence_data)
    window_labels = [p['window_id'] for p in presence_data]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Self-as-Hocolim: Gluing Network</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
               background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3a 50%, #0a1a2a 100%);
               min-height: 100vh; color: #e0e0e0; overflow: hidden; }}
        .header {{ text-align: center; padding: 15px 20px; background: rgba(0,0,0,0.3); border-bottom: 1px solid rgba(100,200,255,0.2); }}
        h1 {{ font-size: 24px; font-weight: 300; color: #00d4ff; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 5px; }}
        .subtitle {{ font-size: 12px; color: #888; letter-spacing: 1px; }}
        .stats {{ display: flex; justify-content: center; gap: 40px; margin-top: 10px; }}
        .stat {{ text-align: center; }}
        .stat-value {{ font-size: 22px; font-weight: 600; color: #00d4ff; }}
        .stat-label {{ font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
        #network {{ width: 100%; height: calc(100vh - 140px); }}
        .tooltip {{ position: absolute; background: rgba(10, 20, 40, 0.95); border: 1px solid #00d4ff;
                   border-radius: 8px; padding: 12px 15px; font-size: 12px; pointer-events: none;
                   max-width: 350px; z-index: 1000; box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3); }}
        .tooltip .title {{ font-size: 14px; font-weight: 600; color: #00d4ff; margin-bottom: 8px;
                          border-bottom: 1px solid rgba(100,200,255,0.3); padding-bottom: 5px; }}
        .tooltip .row {{ display: flex; justify-content: space-between; margin: 4px 0; }}
        .tooltip .label {{ color: #888; }}
        .tooltip .value {{ color: #fff; font-weight: 500; }}
        .tooltip .witnesses {{ margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(100,200,255,0.2); font-size: 11px; color: #aaa; }}
        .tooltip .witnesses span {{ display: inline-block; background: rgba(0, 212, 255, 0.2); padding: 2px 6px; border-radius: 3px; margin: 2px; color: #00d4ff; }}
        .legend {{ position: absolute; bottom: 20px; left: 20px; background: rgba(10, 20, 40, 0.9);
                  border: 1px solid rgba(100,200,255,0.3); border-radius: 8px; padding: 15px; font-size: 11px; }}
        .legend-title {{ font-weight: 600; color: #00d4ff; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; font-size: 10px; }}
        .legend-gradient {{ width: 150px; height: 12px; border-radius: 6px; margin: 5px 0; }}
        .legend-labels {{ display: flex; justify-content: space-between; font-size: 10px; color: #888; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 5px 0; }}
        .legend-circle {{ width: 12px; height: 12px; border-radius: 50%; }}
        .controls {{ position: absolute; top: 100px; right: 20px; background: rgba(10, 20, 40, 0.9);
                    border: 1px solid rgba(100,200,255,0.3); border-radius: 8px; padding: 15px; font-size: 11px; }}
        .controls-title {{ font-weight: 600; color: #00d4ff; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; font-size: 10px; }}
        .control-row {{ margin: 8px 0; }}
        .control-row label {{ display: block; color: #888; margin-bottom: 3px; }}
        .control-row input[type="range"] {{ width: 120px; accent-color: #00d4ff; }}
        .control-row input[type="checkbox"] {{ accent-color: #00d4ff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Self-as-Hocolim</h1>
        <p class="subtitle">Gluing Network · {data['period']} · Computational Proxy for Chapter 5</p>
        <div class="stats">
            <div class="stat"><div class="stat-value">{data['num_journeys']}</div><div class="stat-label">Journeys</div></div>
            <div class="stat"><div class="stat-value">{len(data['gluing_edges']):,}</div><div class="stat-label">Gluing Edges</div></div>
            <div class="stat"><div class="stat-value">{data['num_components']}</div><div class="stat-label">Components</div></div>
            <div class="stat"><div class="stat-value">{data['presence_ratio']:.1%}</div><div class="stat-label">Presence</div></div>
        </div>
    </div>
    <div id="network"></div>
    <div class="tooltip" style="display: none;"></div>
    <div class="legend">
        <div class="legend-title">Birth Time (τ)</div>
        <div class="legend-gradient" style="background: linear-gradient(to right, #00bfff, #00ff88, #ffff00, #ff8800, #ff0066);"></div>
        <div class="legend-labels"><span>{window_labels[0]}</span><span>{window_labels[num_windows//2]}</span><span>{window_labels[-1]}</span></div>
        <div style="margin-top: 15px;"><div class="legend-title">Node Size</div>
            <div class="legend-item"><div class="legend-circle" style="background: #888; width: 8px; height: 8px;"></div><span>Short lifespan</span></div>
            <div class="legend-item"><div class="legend-circle" style="background: #888; width: 16px; height: 16px;"></div><span>Long lifespan</span></div>
        </div>
        <div style="margin-top: 15px;"><div class="legend-title">Edges</div>
            <div class="legend-item"><div style="width: 20px; height: 2px; background: #00d4ff;"></div><span>Cross-temporal</span></div>
            <div class="legend-item"><div style="width: 20px; height: 2px; background: #444;"></div><span>Same period</span></div>
        </div>
        <div style="margin-top: 15px;"><div class="legend-title">Markers</div>
            <div class="legend-item"><div class="legend-circle" style="border: 2px solid gold; background: transparent;"></div><span>Has reentry ↺</span></div>
        </div>
    </div>
    <div class="controls">
        <div class="controls-title">Controls</div>
        <div class="control-row"><label>Link Distance</label><input type="range" id="linkDistance" min="20" max="150" value="60"></div>
        <div class="control-row"><label>Charge Strength</label><input type="range" id="chargeStrength" min="-500" max="-50" value="-150"></div>
        <div class="control-row"><label><input type="checkbox" id="showLabels"> Show Labels</label></div>
        <div class="control-row"><label><input type="checkbox" id="crossTemporalOnly"> Cross-temporal only</label></div>
    </div>
<script>
const nodes = {json.dumps(nodes)};
const links = {json.dumps(links)};
const numWindows = {num_windows};
const windowLabels = {json.dumps(window_labels)};

function getTauColor(tau) {{
    const t = tau / numWindows;
    if (t < 0.25) return d3.interpolateRgb('#00bfff', '#00ff88')(t * 4);
    else if (t < 0.5) return d3.interpolateRgb('#00ff88', '#ffff00')((t - 0.25) * 4);
    else if (t < 0.75) return d3.interpolateRgb('#ffff00', '#ff8800')((t - 0.5) * 4);
    else return d3.interpolateRgb('#ff8800', '#ff0066')((t - 0.75) * 4);
}}

const width = window.innerWidth, height = window.innerHeight - 140;
const svg = d3.select('#network').append('svg').attr('width', width).attr('height', height);
const defs = svg.append('defs');
const filter = defs.append('filter').attr('id', 'glow');
filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
const feMerge = filter.append('feMerge');
feMerge.append('feMergeNode').attr('in', 'coloredBlur');
feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

const g = svg.append('g');
const zoom = d3.zoom().scaleExtent([0.2, 5]).on('zoom', (event) => g.attr('transform', event.transform));
svg.call(zoom);

const tooltip = d3.select('.tooltip');
const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(60).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-150))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => 8 + d.lifespan / 3));

const link = g.append('g').selectAll('line').data(links).join('line')
    .attr('stroke', d => d.cross_temporal ? '#00d4ff' : '#333')
    .attr('stroke-opacity', d => d.cross_temporal ? 0.4 + d.weight/150 : 0.15 + d.weight/200)
    .attr('stroke-width', d => 0.5 + d.weight / 40);

const node = g.append('g').selectAll('circle').data(nodes).join('circle')
    .attr('r', d => 5 + d.lifespan / 4 + d.connectivity / 50)
    .attr('fill', d => getTauColor(d.birth_tau))
    .attr('stroke', d => d.has_reentry ? 'gold' : 'rgba(255,255,255,0.3)')
    .attr('stroke-width', d => d.has_reentry ? 2.5 : 0.5)
    .style('filter', 'url(#glow)').style('cursor', 'pointer')
    .call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended));

const labels = g.append('g').selectAll('text').data(nodes).join('text')
    .text(d => d.signature.slice(0, 15)).attr('font-size', 8).attr('fill', '#888')
    .attr('dx', 10).attr('dy', 3).style('display', 'none').style('pointer-events', 'none');

node.on('mouseover', function(event, d) {{
    d3.select(this).transition().duration(200).attr('r', d => (5 + d.lifespan / 4 + d.connectivity / 50) * 1.5);
    link.attr('stroke-opacity', l => (l.source.id === d.id || l.target.id === d.id) ? 0.9 : (l.cross_temporal ? 0.1 : 0.05));
    tooltip.style('display', 'block').html(`<div class="title">${{d.signature}}</div>
        <div class="row"><span class="label">Birth</span><span class="value">τ=${{d.birth_tau}} (${{windowLabels[d.birth_tau] || '?'}})</span></div>
        <div class="row"><span class="label">Lifespan</span><span class="value">${{d.lifespan}} windows</span></div>
        <div class="row"><span class="label">Reentry</span><span class="value">${{d.has_reentry ? '↺ Yes' : 'No'}}</span></div>
        <div class="row"><span class="label">Glued with</span><span class="value">${{d.glued_with}} journeys</span></div>
        <div class="row"><span class="label">Connections</span><span class="value">${{d.connectivity}} edges</span></div>
        <div class="witnesses"><strong>Top Witnesses:</strong><br>${{d.witnesses.slice(0, 10).map(w => '<span>' + w + '</span>').join('')}}</div>`);
}}).on('mousemove', (event) => {{ tooltip.style('left', (event.pageX + 15) + 'px').style('top', (event.pageY - 10) + 'px'); }})
.on('mouseout', function(event, d) {{
    d3.select(this).transition().duration(200).attr('r', d => 5 + d.lifespan / 4 + d.connectivity / 50);
    link.attr('stroke-opacity', l => l.cross_temporal ? 0.4 + l.weight/150 : 0.15 + l.weight/200);
    tooltip.style('display', 'none');
}});

simulation.on('tick', () => {{
    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y).attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('cx', d => d.x).attr('cy', d => d.y);
    labels.attr('x', d => d.x).attr('y', d => d.y);
}});

function dragstarted(event) {{ if (!event.active) simulation.alphaTarget(0.3).restart(); event.subject.fx = event.subject.x; event.subject.fy = event.subject.y; }}
function dragged(event) {{ event.subject.fx = event.x; event.subject.fy = event.y; }}
function dragended(event) {{ if (!event.active) simulation.alphaTarget(0); event.subject.fx = null; event.subject.fy = null; }}

document.getElementById('linkDistance').addEventListener('input', function() {{ simulation.force('link').distance(+this.value); simulation.alpha(0.3).restart(); }});
document.getElementById('chargeStrength').addEventListener('input', function() {{ simulation.force('charge').strength(+this.value); simulation.alpha(0.3).restart(); }});
document.getElementById('showLabels').addEventListener('change', function() {{ labels.style('display', this.checked ? 'block' : 'none'); }});
document.getElementById('crossTemporalOnly').addEventListener('change', function() {{ link.style('display', l => {{ if (this.checked && !l.cross_temporal) return 'none'; return 'block'; }}); }});
svg.call(zoom.transform, d3.zoomIdentity.translate(width/6, height/6).scale(0.7));
</script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✓ {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for Self-as-Hocolim',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python visualize_self_hocolim.py                     # Uses self_structure.json in current dir
  python visualize_self_hocolim.py results/self_structure.json
  python visualize_self_hocolim.py data.json --output viz/

Outputs:
  timeline.png            Journey lifespans with events
  presence.png            Coherence over time
  network.png             Static network graph
  dashboard.html          Interactive D3 dashboard
  network_interactive.html   Beautiful D3 network with controls
        '''
    )
    parser.add_argument('input', nargs='?', default='self_structure.json',
                       help='Path to self_structure.json (default: ./self_structure.json)')
    parser.add_argument('--output', '-o', default='.',
                       help='Output directory (default: current directory)')
    parser.add_argument('--max-journeys', type=int, default=80,
                       help='Max journeys in timeline (default: 80)')
    parser.add_argument('--max-nodes', type=int, default=120,
                       help='Max nodes in network (default: 120)')
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)
    
    print(f"\nLoading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  {data['num_journeys']} journeys, {len(data['gluing_edges']):,} edges")
    print(f"  {data['num_components']} components, {data['presence_ratio']:.1%} presence")
    print(f"  Period: {data['period']}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\nGenerating visualizations in {args.output}/...")
    
    # Generate all visualizations
    create_timeline(data, os.path.join(args.output, 'timeline.png'), args.max_journeys)
    create_presence_chart(data, os.path.join(args.output, 'presence.png'))
    create_network_static(data, os.path.join(args.output, 'network.png'), max_nodes=args.max_nodes)
    create_dashboard_html(data, os.path.join(args.output, 'dashboard.html'))
    create_network_interactive(data, os.path.join(args.output, 'network_interactive.html'))
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs in {os.path.abspath(args.output)}/:")
    print("  • timeline.png            - Journey lifespans")
    print("  • presence.png            - Coherence over time")
    print("  • network.png             - Static network")
    print("  • dashboard.html          - Interactive dashboard")
    print("  • network_interactive.html - Beautiful D3 network")
    print(f"\nOpen the HTML files in a browser for interactive exploration.")


if __name__ == '__main__':
    main()