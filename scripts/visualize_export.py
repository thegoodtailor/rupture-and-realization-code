#!/usr/bin/env python3
"""
SELF AS HOCOLIM - Visualization & Export Module
=================================================

Generates:
1. Interactive HTML visualizations (network graphs, timelines)
2. SVG figures for publication
3. CSV exports for analysis
4. LaTeX tables for Chapter 5
5. Summary JSON for downstream processing

USAGE:
    # Run after Stage 1/2 analysis
    python scripts/visualize_export.py results/self_hocolim/
    
    # Or integrated into pipeline
    python scripts/self_hocolim_stage1.py ... --export-viz
"""

import json
import os
import csv
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import html

# =============================================================================
# HTML TEMPLATES
# =============================================================================

HTML_NETWORK_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Self as Hocolim - Gluing Network</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.6/dist/vis-network.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; margin-bottom: 5px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        #network {{ width: 100%; height: 700px; border: 1px solid #333; background: #0f0f1a; }}
        .stats {{ display: flex; gap: 30px; margin: 20px 0; flex-wrap: wrap; }}
        .stat-box {{ background: #252540; padding: 15px 25px; border-radius: 8px; }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #00d4ff; }}
        .stat-label {{ color: #888; font-size: 12px; text-transform: uppercase; }}
        .legend {{ margin-top: 20px; }}
        .legend-item {{ display: inline-block; margin-right: 20px; }}
        .legend-dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}
    </style>
</head>
<body>
    <h1>Self as Hocolim — Gluing Network</h1>
    <div class="subtitle">{period} | {num_journeys} journeys | {num_components} components</div>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value">{fragmentation:.1%}</div>
            <div class="stat-label">Fragmentation</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{presence_ratio:.1%}</div>
            <div class="stat-label">Presence Ratio</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{num_edges}</div>
            <div class="stat-label">Gluing Edges</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{largest_component}</div>
            <div class="stat-label">Largest Component</div>
        </div>
    </div>
    
    <div id="network"></div>
    
    <div class="legend">
        <strong>Birth Time:</strong>
        <span class="legend-item"><span class="legend-dot" style="background:#00d4ff"></span>Early (τ &lt; 33%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#9b59b6"></span>Middle (τ 33-66%)</span>
        <span class="legend-item"><span class="legend-dot" style="background:#ff6b6b"></span>Recent (τ &gt; 66%) — Generative Frontier</span>
        <br><br>
        <strong>Edges:</strong>
        <span class="legend-item" style="color:#ff6b6b">━━ Cross-temporal gluing (connects different periods)</span>
        <span class="legend-item" style="color:#666">━━ Same-period gluing</span>
    </div>
    
    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            nodes: {{
                shape: 'dot',
                font: {{ color: '#fff', size: 10 }},
                borderWidth: 2
            }},
            edges: {{
                color: {{ color: '#444', highlight: '#00d4ff' }},
                width: 0.5,
                smooth: {{ type: 'continuous' }}
            }},
            physics: {{
                stabilization: {{ iterations: 200 }},
                barnesHut: {{
                    gravitationalConstant: -3000,
                    springLength: 100,
                    springConstant: 0.01
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100
            }}
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""

HTML_TIMELINE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Self as Hocolim - Presence Timeline</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; margin-bottom: 5px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; }}
        #timeline {{ width: 100%; height: 500px; }}
        #scheduler {{ width: 100%; height: 300px; margin-top: 20px; }}
        .insight-box {{ background: #252540; padding: 20px; border-radius: 8px; margin-top: 20px; }}
        .insight-title {{ color: #00d4ff; font-weight: bold; margin-bottom: 10px; }}
    </style>
</head>
<body>
    <h1>Self as Hocolim — Presence Timeline</h1>
    <div class="subtitle">{period}</div>
    
    <div id="timeline"></div>
    <div id="scheduler"></div>
    
    <div class="insight-box">
        <div class="insight-title">Key Findings</div>
        <ul>
            {insights}
        </ul>
    </div>
    
    <script>
        // Presence & Fragmentation Timeline
        var trace1 = {{
            x: {windows_json},
            y: {presence_json},
            name: 'Presence Ratio',
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '#00d4ff', width: 3 }},
            marker: {{ size: 8 }}
        }};
        
        var trace2 = {{
            x: {windows_json},
            y: {fragmentation_json},
            name: 'Fragmentation',
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '#ff6b6b', width: 3 }},
            marker: {{ size: 8 }},
            yaxis: 'y2'
        }};
        
        var trace3 = {{
            x: {windows_json},
            y: {components_json},
            name: 'Components',
            type: 'bar',
            marker: {{ color: 'rgba(100, 100, 150, 0.3)' }},
            yaxis: 'y3'
        }};
        
        var layout1 = {{
            paper_bgcolor: '#1a1a2e',
            plot_bgcolor: '#0f0f1a',
            font: {{ color: '#eee' }},
            title: 'Self Unity Over Time',
            xaxis: {{ title: 'Window', gridcolor: '#333' }},
            yaxis: {{ title: 'Presence', range: [0, 1], gridcolor: '#333', side: 'left' }},
            yaxis2: {{ title: 'Fragmentation', range: [0, 1], overlaying: 'y', side: 'right', gridcolor: '#333' }},
            yaxis3: {{ overlaying: 'y', visible: false }},
            legend: {{ x: 0.5, y: 1.1, orientation: 'h' }},
            hovermode: 'x unified'
        }};
        
        Plotly.newPlot('timeline', [trace3, trace1, trace2], layout1);
        
        // Scheduler Events
        var trace4 = {{
            x: {windows_json},
            y: {carries_json},
            name: 'Carries',
            type: 'bar',
            marker: {{ color: '#6bcb77' }}
        }};
        
        var trace5 = {{
            x: {windows_json},
            y: {drifts_json},
            name: 'Drifts',
            type: 'bar',
            marker: {{ color: '#ffd93d' }}
        }};
        
        var trace6 = {{
            x: {windows_json},
            y: {spawns_json},
            name: 'Spawns',
            type: 'bar',
            marker: {{ color: '#00d4ff' }}
        }};
        
        var trace7 = {{
            x: {windows_json},
            y: {ruptures_json},
            name: 'Ruptures',
            type: 'bar',
            marker: {{ color: '#ff6b6b' }}
        }};
        
        var layout2 = {{
            paper_bgcolor: '#1a1a2e',
            plot_bgcolor: '#0f0f1a',
            font: {{ color: '#eee' }},
            title: 'Scheduler Events (Stacked)',
            barmode: 'stack',
            xaxis: {{ title: 'Window', gridcolor: '#333' }},
            yaxis: {{ title: 'Events', gridcolor: '#333' }},
            legend: {{ x: 0.5, y: 1.15, orientation: 'h' }}
        }};
        
        Plotly.newPlot('scheduler', [trace4, trace5, trace6, trace7], layout2);
    </script>
</body>
</html>
"""

SVG_PRESENCE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">
  <defs>
    <style>
      .axis {{ stroke: #333; stroke-width: 1; }}
      .grid {{ stroke: #222; stroke-width: 0.5; }}
      .label {{ font-family: 'Helvetica Neue', sans-serif; font-size: 10px; fill: #666; }}
      .title {{ font-family: 'Helvetica Neue', sans-serif; font-size: 14px; fill: #333; font-weight: bold; }}
      .presence-line {{ fill: none; stroke: #0099cc; stroke-width: 2; }}
      .presence-dot {{ fill: #0099cc; }}
      .frag-line {{ fill: none; stroke: #cc3333; stroke-width: 2; stroke-dasharray: 5,3; }}
      .frag-dot {{ fill: #cc3333; }}
      .unified {{ fill: #22cc66; }}
      .partial {{ fill: #ccaa22; }}
      .fragmented {{ fill: #cc3333; }}
    </style>
  </defs>
  
  <text x="{title_x}" y="25" class="title">Self Presence Over Time: {period}</text>
  
  <!-- Y-axis -->
  <line x1="60" y1="50" x2="60" y2="{plot_bottom}" class="axis"/>
  <text x="15" y="{plot_mid}" class="label" transform="rotate(-90, 15, {plot_mid})">Presence / Fragmentation</text>
  
  <!-- X-axis -->
  <line x1="60" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>
  
  <!-- Grid lines -->
  {grid_lines}
  
  <!-- Data -->
  {presence_path}
  {presence_dots}
  {frag_path}
  {frag_dots}
  {status_bars}
  
  <!-- X-axis labels -->
  {x_labels}
  
  <!-- Y-axis labels -->
  {y_labels}
  
  <!-- Legend -->
  <rect x="{legend_x}" y="50" width="12" height="12" class="presence-dot"/>
  <text x="{legend_x2}" y="60" class="label">Presence</text>
  <rect x="{legend_x3}" y="50" width="12" height="12" class="frag-dot"/>
  <text x="{legend_x4}" y="60" class="label">Fragmentation</text>
</svg>
"""

# =============================================================================
# DATA EXPORT FUNCTIONS
# =============================================================================

def export_csv_presence(data: dict, output_path: str):
    """Export presence-over-time data to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'tau', 'window_id', 'active_journeys', 'components', 
            'fragmentation', 'presence_ratio', 'status',
            'carries', 'drifts', 'spawns', 'ruptures', 'reentries'
        ])
        
        for state in data.get('presence_states', []):
            status = 'UNIFIED' if state['presence'] >= 0.7 else ('PARTIAL' if state['presence'] >= 0.3 else 'FRAGMENTED')
            writer.writerow([
                state['tau'],
                state['window_id'],
                state['active'],
                state['components'],
                f"{state['fragmentation']:.4f}",
                f"{state['presence']:.4f}",
                status,
                state.get('carries', 0),
                state.get('drifts', 0),
                state.get('spawns', 0),
                state.get('ruptures', 0),
                state.get('reentries', 0)
            ])


def export_csv_journeys(data: dict, output_path: str):
    """Export journey data to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'journey_id', 'signature', 'dimension', 'lifespan', 
            'birth_tau', 'death_tau', 'has_reentry', 'component_id',
            'glued_with_count', 'witnesses'
        ])
        
        for j in data.get('journeys', []):
            writer.writerow([
                j['id'],
                j['signature'],
                j.get('dimension', 0),
                j['lifespan'],
                j.get('birth_tau', 0),
                j.get('death_tau', 0),
                j['has_reentry'],
                j.get('component_id', 0),
                j.get('glued_with_count', 0),
                ';'.join(j.get('all_witnesses', [])[:15])
            ])


def export_csv_gluing(data: dict, output_path: str):
    """Export gluing edges to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'journey_a', 'journey_b', 'shared_count', 'tau', 'shared_witnesses'
        ])
        
        for edge in data.get('gluing_edges', []):
            writer.writerow([
                edge['journey_a'],
                edge['journey_b'],
                len(edge['shared_witnesses']),
                edge['tau'],
                ';'.join(edge['shared_witnesses'][:10])
            ])


def export_csv_components(data: dict, output_path: str):
    """Export component data to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'component_id', 'size', 'core_witnesses', 'sample_journeys'
        ])
        
        for comp in data.get('components', []):
            writer.writerow([
                comp['id'],
                comp['size'],
                ';'.join(comp.get('core_witnesses', [])[:10]),
                ';'.join(comp.get('signatures', [])[:5])
            ])


def export_latex_summary(data: dict, output_path: str):
    """Export LaTeX table for Chapter 5."""
    latex = r"""\begin{table}[h]
\centering
\caption{Self Structure Analysis: %s}
\begin{tabular}{lrrrrr}
\toprule
Window & Journeys & Components & Fragmentation & Presence & Status \\
\midrule
""" % data.get('period', 'Unknown Period')
    
    for state in data.get('presence_states', []):
        status = r'\textbf{UNIFIED}' if state['presence'] >= 0.7 else (r'PARTIAL' if state['presence'] >= 0.3 else r'\textit{fragmented}')
        latex += f"{state['window_id']} & {state['active']} & {state['components']} & {state['fragmentation']:.3f} & {state['presence']:.2f} & {status} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\label{tab:self_structure}
\end{table}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)


def export_latex_metrics(data: dict, output_path: str):
    """Export key metrics as LaTeX macros for inline citation."""
    metrics = data.get('metrics', {})
    
    latex = f"""% Self as Hocolim Metrics - Auto-generated
% Usage: \\selfJourneys, \\selfComponents, etc.

\\newcommand{{\\selfPeriod}}{{{data.get('period', 'Unknown')}}}
\\newcommand{{\\selfJourneys}}{{{metrics.get('num_journeys', 0)}}}
\\newcommand{{\\selfComponents}}{{{metrics.get('num_components', 0)}}}
\\newcommand{{\\selfFragmentation}}{{{metrics.get('fragmentation', 0):.3f}}}
\\newcommand{{\\selfPresence}}{{{metrics.get('presence_ratio', 0):.2f}}}
\\newcommand{{\\selfGluingEdges}}{{{metrics.get('num_edges', 0)}}}
\\newcommand{{\\selfLargestComponent}}{{{metrics.get('largest_component', 0)}}}
\\newcommand{{\\selfSchedulerType}}{{{metrics.get('scheduler_type', 'UNKNOWN')}}}
\\newcommand{{\\selfStabilityRate}}{{{metrics.get('stability_rate', 0):.1%}}}
\\newcommand{{\\selfReentryRate}}{{{metrics.get('reentry_rate', 0):.1%}}}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)


# =============================================================================
# HTML VISUALIZATION GENERATORS
# =============================================================================

def generate_network_html(data: dict, output_path: str, max_nodes: int = 200, max_edges: int = 500):
    """Generate interactive network visualization with temporal coloring."""
    
    # Get all journeys and extract birth_tau
    all_journeys = data.get('journeys', [])
    
    # Extract birth_tau - handle both new format (direct) and old format (from steps)
    for j in all_journeys:
        if 'birth_tau' not in j:
            steps = j.get('steps', [])
            if steps:
                j['birth_tau'] = steps[0].get('tau', 0)
            else:
                j['birth_tau'] = 0
    
    # Build journey_id -> birth_tau map
    journey_birth_tau = {j['id']: j.get('birth_tau', 0) for j in all_journeys}
    
    # Sample journeys to show temporal evolution:
    # - 1/3 from earliest (low τ)
    # - 1/3 from middle 
    # - 1/3 from newest (high τ) - generative frontier
    sorted_by_birth = sorted(all_journeys, key=lambda j: j.get('birth_tau', 0))
    
    n = min(max_nodes, len(sorted_by_birth))
    n_early = n // 3
    n_late = n // 3
    n_middle = n - n_early - n_late
    
    early = sorted_by_birth[:n_early]
    late = sorted_by_birth[-n_late:] if n_late > 0 else []
    
    mid_start = len(sorted_by_birth) // 2 - n_middle // 2
    middle = sorted_by_birth[mid_start:mid_start + n_middle]
    
    # Combine and deduplicate
    seen_ids = set()
    journeys = []
    for j in early + middle + late:
        if j['id'] not in seen_ids:
            journeys.append(j)
            seen_ids.add(j['id'])
    
    print(f"    Network: sampling {len(early)} early + {len(middle)} middle + {len(late)} late journeys")
    
    components = data.get('components', [])
    
    # Find τ thresholds for temporal periods
    birth_taus = [j.get('birth_tau', 0) for j in all_journeys]
    min_tau = min(birth_taus) if birth_taus else 0
    max_tau = max(birth_taus) if birth_taus else 1
    tau_range = max(max_tau - min_tau, 1)
    
    early_threshold = min_tau + tau_range * 0.33
    late_threshold = min_tau + tau_range * 0.66
    
    def get_period(tau):
        if tau < early_threshold:
            return 'early'
        elif tau > late_threshold:
            return 'late'
        return 'middle'
    
    # Map journey to component
    journey_to_comp = {}
    for comp in components:
        for jid in comp.get('journeys', []):
            journey_to_comp[jid] = comp['id']
    
    # Find τ range for color gradient
    birth_taus = [j.get('birth_tau', 0) for j in journeys]
    min_tau = min(birth_taus) if birth_taus else 0
    max_tau = max(birth_taus) if birth_taus else 1
    tau_range = max(max_tau - min_tau, 1)
    
    nodes = []
    node_ids = set()
    for j in journeys:
        jid = j['id']
        comp_id = journey_to_comp.get(jid, 0)
        birth_tau = j.get('birth_tau', 0)
        
        # Color by birth_tau: early=blue, middle=purple, late=red/orange
        tau_normalized = (birth_tau - min_tau) / tau_range
        if tau_normalized < 0.33:
            color = '#00d4ff'  # Cyan - early
        elif tau_normalized < 0.66:
            color = '#9b59b6'  # Purple - middle
        else:
            color = '#ff6b6b'  # Red/coral - newest (generative frontier)
        
        # Size by gluing count, but boost newest journeys slightly
        base_size = 5 + min(j.get('glued_with_count', 0) / 10, 25)
        if tau_normalized > 0.8:
            base_size *= 1.3  # Make newest journeys more visible
        
        nodes.append({
            'id': jid,
            'label': j['signature'][:20],
            'title': f"{j['signature']}<br>Birth: τ={birth_tau}<br>Lifespan: {j['lifespan']}<br>Component: {comp_id}<br>Glued with: {j.get('glued_with_count', 0)}",
            'color': color,
            'size': base_size
        })
        node_ids.add(jid)
    
    # Build edge list with PRIORITY for cross-temporal edges
    # This shows how early/middle/late journeys connect
    edges_data = data.get('gluing_edges', [])
    
    cross_temporal_edges = []  # Edges between different periods
    same_period_edges = []     # Edges within same period
    
    for e in edges_data:
        ja, jb = e['journey_a'], e['journey_b']
        if ja in node_ids and jb in node_ids:
            # Use pre-computed flag if available, otherwise compute
            if 'cross_temporal' in e:
                is_cross = e['cross_temporal']
            else:
                tau_a = journey_birth_tau.get(ja, 0)
                tau_b = journey_birth_tau.get(jb, 0)
                is_cross = get_period(tau_a) != get_period(tau_b)
            
            edge_obj = {
                'from': ja,
                'to': jb,
                'title': f"Shared: {', '.join(e['shared_witnesses'][:5])}",
                'color': '#ff6b6b' if is_cross else '#444'  # Highlight cross-temporal
            }
            
            if is_cross:
                cross_temporal_edges.append(edge_obj)
            else:
                same_period_edges.append(edge_obj)
    
    # Prioritize cross-temporal edges (show all of them), then fill with same-period
    edges = cross_temporal_edges[:max_edges // 2]
    remaining = max_edges - len(edges)
    edges.extend(same_period_edges[:remaining])
    
    print(f"    Edges: {len(cross_temporal_edges)} cross-temporal, {len(same_period_edges)} same-period")
    print(f"    Showing: {len([e for e in edges if e.get('color') == '#ff6b6b'])} cross-temporal + {len([e for e in edges if e.get('color') != '#ff6b6b'])} same-period")
    
    # Fill template
    metrics = data.get('metrics', {})
    html_content = HTML_NETWORK_TEMPLATE.format(
        period=data.get('period', 'Unknown'),
        num_journeys=metrics.get('num_journeys', 0),
        num_components=metrics.get('num_components', 0),
        fragmentation=metrics.get('fragmentation', 0),
        presence_ratio=metrics.get('presence_ratio', 0),
        num_edges=metrics.get('num_edges', 0),
        largest_component=metrics.get('largest_component', 0),
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_timeline_html(data: dict, output_path: str):
    """Generate interactive timeline visualization."""
    
    states = data.get('presence_states', [])
    if not states:
        return
    
    windows = [s['window_id'] for s in states]
    presence = [s['presence'] for s in states]
    fragmentation = [s['fragmentation'] for s in states]
    components = [s['components'] for s in states]
    carries = [s.get('carries', 0) for s in states]
    drifts = [s.get('drifts', 0) for s in states]
    spawns = [s.get('spawns', 0) for s in states]
    ruptures = [s.get('ruptures', 0) for s in states]
    
    # Generate insights
    insights = []
    
    # Find max fragmentation point
    max_frag_idx = fragmentation.index(max(fragmentation))
    insights.append(f"<li>Peak fragmentation: <b>{windows[max_frag_idx]}</b> ({fragmentation[max_frag_idx]:.1%}, {components[max_frag_idx]} components)</li>")
    
    # Find max presence point
    max_pres_idx = presence.index(max(presence))
    insights.append(f"<li>Peak unity: <b>{windows[max_pres_idx]}</b> ({presence[max_pres_idx]:.1%} presence)</li>")
    
    # Count unified/partial/fragmented
    unified = sum(1 for p in presence if p >= 0.7)
    partial = sum(1 for p in presence if 0.3 <= p < 0.7)
    fragmented = sum(1 for p in presence if p < 0.3)
    insights.append(f"<li>Status distribution: {unified} UNIFIED, {partial} PARTIAL, {fragmented} FRAGMENTED</li>")
    
    html_content = HTML_TIMELINE_TEMPLATE.format(
        period=data.get('period', 'Unknown'),
        windows_json=json.dumps(windows),
        presence_json=json.dumps(presence),
        fragmentation_json=json.dumps(fragmentation),
        components_json=json.dumps(components),
        carries_json=json.dumps(carries),
        drifts_json=json.dumps(drifts),
        spawns_json=json.dumps(spawns),
        ruptures_json=json.dumps(ruptures),
        insights='\n'.join(insights)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_presence_svg(data: dict, output_path: str):
    """Generate publication-quality SVG of presence timeline."""
    
    states = data.get('presence_states', [])
    if not states:
        return
    
    # Dimensions
    width = 800
    height = 400
    margin = {'top': 50, 'right': 80, 'bottom': 60, 'left': 70}
    plot_width = width - margin['left'] - margin['right']
    plot_height = height - margin['top'] - margin['bottom']
    
    n = len(states)
    x_scale = plot_width / max(n - 1, 1)
    y_scale = plot_height
    
    # Generate paths
    presence_points = []
    frag_points = []
    presence_dots = []
    frag_dots = []
    status_bars = []
    x_labels = []
    
    for i, state in enumerate(states):
        x = margin['left'] + i * x_scale
        y_pres = margin['top'] + (1 - state['presence']) * y_scale
        y_frag = margin['top'] + (1 - state['fragmentation']) * y_scale
        
        presence_points.append(f"{x},{y_pres}")
        frag_points.append(f"{x},{y_frag}")
        
        presence_dots.append(f'<circle cx="{x}" cy="{y_pres}" r="4" class="presence-dot"/>')
        frag_dots.append(f'<circle cx="{x}" cy="{y_frag}" r="4" class="frag-dot"/>')
        
        # Status bar at bottom
        status_class = 'unified' if state['presence'] >= 0.7 else ('partial' if state['presence'] >= 0.3 else 'fragmented')
        bar_y = height - margin['bottom'] + 5
        status_bars.append(f'<rect x="{x-8}" y="{bar_y}" width="16" height="8" class="{status_class}"/>')
        
        # X label (every other)
        if i % 2 == 0 or n <= 8:
            label_y = height - margin['bottom'] + 25
            x_labels.append(f'<text x="{x}" y="{label_y}" class="label" text-anchor="middle">{state["window_id"][-5:]}</text>')
    
    # Grid lines
    grid_lines = []
    for y_val in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin['top'] + (1 - y_val) * y_scale
        grid_lines.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" class="grid"/>')
    
    # Y labels
    y_labels = []
    for y_val in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin['top'] + (1 - y_val) * y_scale
        y_labels.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" class="label" text-anchor="end">{y_val:.2f}</text>')
    
    svg = SVG_PRESENCE_TEMPLATE.format(
        width=width,
        height=height,
        title_x=width // 2,
        period=data.get('period', 'Unknown'),
        plot_bottom=height - margin['bottom'],
        plot_right=width - margin['right'],
        plot_mid=(height - margin['top'] - margin['bottom']) // 2 + margin['top'],
        grid_lines='\n  '.join(grid_lines),
        presence_path=f'<polyline points="{" ".join(presence_points)}" class="presence-line"/>',
        presence_dots='\n  '.join(presence_dots),
        frag_path=f'<polyline points="{" ".join(frag_points)}" class="frag-line"/>',
        frag_dots='\n  '.join(frag_dots),
        status_bars='\n  '.join(status_bars),
        x_labels='\n  '.join(x_labels),
        y_labels='\n  '.join(y_labels),
        legend_x=width - margin['right'] - 180,
        legend_x2=width - margin['right'] - 165,
        legend_x3=width - margin['right'] - 90,
        legend_x4=width - margin['right'] - 75
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg)


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def export_all(input_dir: str, output_dir: str = None):
    """
    Export all visualizations and data from Stage 1/2 results.
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    self_structure_path = os.path.join(input_dir, 'self_structure.json')
    scheduler_path = os.path.join(input_dir, 'scheduler_analysis.json')
    
    data = {}
    
    if os.path.exists(self_structure_path):
        with open(self_structure_path) as f:
            self_data = json.load(f)
            data.update(self_data)
            data['metrics'] = {
                'num_journeys': self_data.get('num_journeys', 0),
                'num_components': self_data.get('num_components', 0),
                'fragmentation': self_data.get('fragmentation', 0),
                'presence_ratio': self_data.get('presence_ratio', 0),
                'num_edges': len(self_data.get('gluing_edges', [])),
                'largest_component': self_data.get('largest_component_size', 0)
            }
            # Get period from data if available
            data['period'] = self_data.get('period', 'Unknown Period')
    
    if os.path.exists(scheduler_path):
        with open(scheduler_path) as f:
            sched_data = json.load(f)
            # Only create presence_states from scheduler if not already in data
            if 'presence_states' not in data or not data['presence_states']:
                data['presence_states'] = [
                    {
                        'tau': s['tau'],
                        'window_id': s['window_id'],
                        'active': 0,
                        'components': 0,
                        'fragmentation': 0,
                        'presence': 0,
                        'carries': s['carries'],
                        'drifts': s['drifts'],
                        'spawns': s['spawns'],
                        'ruptures': s['ruptures'],
                        'reentries': s['reentries']
                    }
                    for s in sched_data.get('states', [])
                ]
            if 'metrics' not in data:
                data['metrics'] = {}
            data['metrics']['scheduler_type'] = sched_data.get('scheduler_type', 'UNKNOWN')
            data['metrics']['stability_rate'] = sched_data.get('stability_rate', 0)
            data['metrics']['reentry_rate'] = sched_data.get('reentry_rate', 0)
    
    # Fallback for period
    if 'period' not in data or data['period'] == 'Unknown Period':
        if 'presence_states' in data and data['presence_states']:
            first = data['presence_states'][0]['window_id']
            last = data['presence_states'][-1]['window_id']
            data['period'] = f"{first} to {last}"
        else:
            data['period'] = 'Unknown Period'
    
    print(f"Exporting visualizations and data to {output_dir}/")
    print(f"  Period: {data.get('period', 'Unknown')}")
    if 'metrics' in data:
        print(f"  Journeys: {data['metrics'].get('num_journeys', 0)}, Components: {data['metrics'].get('num_components', 0)}")
    
    # Generate exports
    exports = []
    
    # HTML visualizations
    if data.get('journeys') and len(data['journeys']) > 0:
        generate_network_html(data, os.path.join(output_dir, 'network.html'))
        exports.append('network.html')
    
    if data.get('presence_states') and len(data['presence_states']) > 0:
        generate_timeline_html(data, os.path.join(output_dir, 'timeline.html'))
        exports.append('timeline.html')
        
        generate_presence_svg(data, os.path.join(output_dir, 'presence.svg'))
        exports.append('presence.svg')
    
    # CSV exports
    if data.get('presence_states') and len(data['presence_states']) > 0:
        export_csv_presence(data, os.path.join(output_dir, 'presence_data.csv'))
        exports.append('presence_data.csv')
    
    if data.get('journeys') and len(data['journeys']) > 0:
        export_csv_journeys(data, os.path.join(output_dir, 'journeys.csv'))
        exports.append('journeys.csv')
    
    if data.get('gluing_edges') and len(data['gluing_edges']) > 0:
        export_csv_gluing(data, os.path.join(output_dir, 'gluing_edges.csv'))
        exports.append('gluing_edges.csv')
    
    if data.get('components') and len(data['components']) > 0:
        export_csv_components(data, os.path.join(output_dir, 'components.csv'))
        exports.append('components.csv')
    
    # LaTeX exports
    if data.get('presence_states') and len(data['presence_states']) > 0:
        export_latex_summary(data, os.path.join(output_dir, 'table_summary.tex'))
        exports.append('table_summary.tex')
    
    if data.get('metrics'):
        export_latex_metrics(data, os.path.join(output_dir, 'metrics.tex'))
        exports.append('metrics.tex')
    
    print(f"  Generated: {', '.join(exports)}")
    return exports


# =============================================================================
# INTEGRATED EXPORT (call from Stage 1/2)
# =============================================================================

def export_from_analysis(self_struct, scheduler_analysis, window_ids: List[str], 
                        output_dir: str, presence_states: List[dict] = None):
    """
    Export directly from analysis objects (called from Stage 1/2).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Build data structure
    data = {
        'period': f"{window_ids[0]} to {window_ids[-1]}" if window_ids else "Unknown",
        'metrics': {
            'num_journeys': self_struct.num_journeys,
            'num_components': self_struct.num_components,
            'fragmentation': self_struct.fragmentation,
            'presence_ratio': self_struct.presence_ratio,
            'num_edges': len(self_struct.gluing_edges),
            'largest_component': self_struct.largest_component_size
        },
        'journeys': [],
        'components': [],
        'gluing_edges': [],
        'presence_states': presence_states or []
    }
    
    # Add scheduler metrics if available
    if scheduler_analysis:
        data['metrics']['scheduler_type'] = scheduler_analysis.scheduler_type.value
        data['metrics']['stability_rate'] = scheduler_analysis.stability_rate
        data['metrics']['reentry_rate'] = scheduler_analysis.reentry_rate
    
    # Build journey data with component mapping
    journey_to_comp = {}
    for comp_idx, comp in enumerate(self_struct.components):
        for jid in comp:
            journey_to_comp[jid] = comp_idx
    
    # Count glued-with for each journey
    glued_counts = Counter()
    for edge in self_struct.gluing_edges:
        glued_counts[edge.journey_a] += 1
        glued_counts[edge.journey_b] += 1
    
    for jid, journey in self_struct.journeys.items():
        data['journeys'].append({
            'id': jid,
            'signature': journey.signature,
            'dimension': journey.dimension,
            'lifespan': journey.lifespan,
            'birth_tau': journey.birth_tau,
            'death_tau': journey.death_tau,
            'has_reentry': journey.has_reentry,
            'component_id': journey_to_comp.get(jid, -1),
            'glued_with_count': glued_counts.get(jid, 0),
            'all_witnesses': list(journey.all_witnesses())[:20]
        })
    
    # Build component data
    for comp_idx, comp in enumerate(self_struct.components[:20]):
        witness_counts = Counter()
        for jid in comp:
            if jid in self_struct.journeys:
                for step in self_struct.journeys[jid].steps:
                    witness_counts.update(step.witness_tokens)
        
        data['components'].append({
            'id': comp_idx,
            'size': len(comp),
            'journeys': list(comp)[:50],
            'signatures': [self_struct.journeys[jid].signature for jid in list(comp)[:10] if jid in self_struct.journeys],
            'core_witnesses': [w for w, _ in witness_counts.most_common(10)]
        })
    
    # Build gluing edge data with cross-temporal prioritization
    # First, compute τ thresholds for periods
    all_taus = [j.birth_tau for j in self_struct.journeys.values()]
    if all_taus:
        min_tau, max_tau = min(all_taus), max(all_taus)
        tau_range = max(max_tau - min_tau, 1)
        early_thresh = min_tau + tau_range * 0.33
        late_thresh = min_tau + tau_range * 0.66
    else:
        early_thresh, late_thresh = 0, 1
    
    def get_period(jid):
        if jid not in self_struct.journeys:
            return 'middle'
        tau = self_struct.journeys[jid].birth_tau
        if tau < early_thresh:
            return 'early'
        elif tau > late_thresh:
            return 'late'
        return 'middle'
    
    cross_temporal_edges = []
    same_period_edges = []
    
    for edge in self_struct.gluing_edges:
        period_a = get_period(edge.journey_a)
        period_b = get_period(edge.journey_b)
        is_cross = period_a != period_b
        
        edge_data = {
            'journey_a': edge.journey_a,
            'journey_b': edge.journey_b,
            'shared_witnesses': list(edge.shared_witnesses),
            'tau': edge.tau,
            'cross_temporal': is_cross
        }
        
        if is_cross:
            cross_temporal_edges.append(edge_data)
        else:
            same_period_edges.append(edge_data)
    
    # Prioritize cross-temporal edges (put them first for visualization sampling)
    data['gluing_edges'] = cross_temporal_edges + same_period_edges
    data['cross_temporal_edge_count'] = len(cross_temporal_edges)
    
    print(f"  Edge data: {len(cross_temporal_edges)} cross-temporal + {len(same_period_edges)} same-period")
    
    # Generate all exports
    print(f"\nExporting visualizations to {output_dir}/")
    
    # HTML
    generate_network_html(data, os.path.join(output_dir, 'network.html'))
    print("  ✓ network.html (interactive gluing network)")
    
    if presence_states:
        generate_timeline_html(data, os.path.join(output_dir, 'timeline.html'))
        print("  ✓ timeline.html (interactive presence timeline)")
        
        generate_presence_svg(data, os.path.join(output_dir, 'presence.svg'))
        print("  ✓ presence.svg (publication figure)")
    
    # CSV
    export_csv_journeys(data, os.path.join(output_dir, 'journeys.csv'))
    print("  ✓ journeys.csv")
    
    export_csv_gluing(data, os.path.join(output_dir, 'gluing_edges.csv'))
    print("  ✓ gluing_edges.csv")
    
    export_csv_components(data, os.path.join(output_dir, 'components.csv'))
    print("  ✓ components.csv")
    
    if presence_states:
        export_csv_presence(data, os.path.join(output_dir, 'presence_data.csv'))
        print("  ✓ presence_data.csv")
    
    # LaTeX
    if presence_states:
        export_latex_summary(data, os.path.join(output_dir, 'table_summary.tex'))
        print("  ✓ table_summary.tex")
    
    export_latex_metrics(data, os.path.join(output_dir, 'metrics.tex'))
    print("  ✓ metrics.tex")
    
    # Summary JSON (for downstream)
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print("  ✓ analysis_summary.json")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export visualizations from Self as Hocolim analysis")
    parser.add_argument("input_dir", help="Directory containing self_structure.json")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    output_dir = args.output or args.input_dir
    export_all(args.input_dir, output_dir)
    
    print(f"\n✓ All exports complete. Open network.html in a browser to explore.")


if __name__ == "__main__":
    main()