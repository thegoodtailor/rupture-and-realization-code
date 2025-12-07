#!/usr/bin/env python3
"""
Chapter 4 Worked Example: Witnessed Bars for a Single Slice
============================================================

This demonstrates Codebase 1 output for the book's Chapter 4.
Shows H₀ (thematic components) and H₁ (semantic loops) with full witnesses.

The text is a conversation about climate and Rome - matching the later
temporal example in Chapter 4, but analysed here as a SINGLE SLICE
(all utterances pooled together, no time evolution).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# TEXT: Climate and Rome conversation (matching Chapter 4 example)
# =============================================================================

# NOTE: We strip "User:" and "Assistant:" labels to avoid them becoming
# spurious thematic clusters. The content words are what matter.

TEXT = """
i've been thinking about climate change a lot lately
climate change is certainly one of the defining challenges of our time what aspects concern you most
the economic impacts worry me how do we balance growth with sustainability
that tension between economic growth and environmental protection is real many economists now argue for sustainable development models
what about carbon pricing does it actually work
carbon pricing can be effective when well designed it creates market incentives to reduce emissions while allowing flexibility in how reductions are achieved
let's change topics tell me about ancient rome
ancient rome was a remarkable civilization that lasted over a thousand years from a small city state to a vast empire spanning many regions it shaped law architecture and governance for centuries
actually let's go back to climate what about renewable energy
renewable energy has seen remarkable growth solar and wind costs have plummeted making them competitive with fossil fuels in many regions
so there's hope for addressing climate change through technology
technology is certainly part of the solution renewables storage and efficiency improvements all contribute to reducing carbon emissions
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "min_persistence": 0.03,
    "min_witness_tokens": 2,
    "segmentation": "lines",
}

# =============================================================================
# ANALYSIS
# =============================================================================

def main():
    from witnessed_ph import analyse_text_single_slice, default_config
    
    print("=" * 72)
    print("CHAPTER 4 WORKED EXAMPLE: WITNESSED BARS (SINGLE SLICE)")
    print("=" * 72)
    print()
    print("This analysis treats the entire conversation as ONE time slice τ.")
    print("We extract the witnessed persistence diagram D^W(τ) showing:")
    print("  • H₀ bars: thematic components (clusters of related tokens)")
    print("  • H₁ bars: semantic loops (cycles of meaning)")
    print()
    
    config = default_config()
    config["min_persistence"] = CONFIG["min_persistence"]
    config["min_witness_tokens"] = CONFIG["min_witness_tokens"]
    
    print(f"Configuration:")
    print(f"  min_persistence = {config['min_persistence']}")
    print(f"  min_witness_tokens = {config['min_witness_tokens']}")
    print(f"  embedding_model = {config['embedding_model']}")
    print()
    
    # Run analysis
    diagram = analyse_text_single_slice(
        TEXT, 
        config=config,
        segmentation_mode=CONFIG["segmentation"],
        verbose=True
    )
    
    # Separate by dimension
    h0_bars = [b for b in diagram['bars'] if b['dim'] == 0]
    h1_bars = [b for b in diagram['bars'] if b['dim'] == 1]
    
    print()
    print("=" * 72)
    print(f"WITNESSED PERSISTENCE DIAGRAM D^W(τ)")
    print(f"  Total bars: {len(diagram['bars'])}")
    print(f"  H₀ bars (components): {len(h0_bars)}")
    print(f"  H₁ bars (loops): {len(h1_bars)}")
    print(f"  Tokens analysed: {diagram['num_tokens']}")
    print(f"  Utterances: {diagram['num_utterances']}")
    print("=" * 72)
    
    # =========================================================================
    # H₀ BARS: THEMATIC COMPONENTS
    # =========================================================================
    print()
    print("=" * 72)
    print("H₀ BARS: THEMATIC COMPONENTS")
    print("=" * 72)
    print()
    print("Each H₀ bar represents a connected component in the embedding space:")
    print("a cluster of tokens that remain close across the filtration scale.")
    print("The witness tells us WHAT the component is about.")
    print()
    
    # Categorise bars by likely theme
    def categorise_bar(bar):
        tokens_lower = [t.lower() for t in bar['witness']['tokens']['surface']]
        token_set = set(tokens_lower)
        
        climate_words = {'climate', 'carbon', 'emissions', 'warming', 'environmental', 
                        'sustainability', 'renewable', 'solar', 'wind', 'energy', 'fossil'}
        rome_words = {'rome', 'roman', 'civilization', 'empire', 'ancient', 'architecture',
                     'governance', 'law', 'centuries', 'city-state'}
        econ_words = {'economic', 'growth', 'pricing', 'market', 'incentives', 'costs',
                     'competitive', 'development', 'balance'}
        tech_words = {'technology', 'storage', 'efficiency', 'improvements', 'solution'}
        
        if token_set & climate_words:
            return "CLIMATE"
        elif token_set & rome_words:
            return "ROME"
        elif token_set & econ_words:
            return "ECONOMICS"
        elif token_set & tech_words:
            return "TECHNOLOGY"
        else:
            return "OTHER"
    
    # Group bars by theme
    by_theme = {}
    for bar in h0_bars:
        theme = categorise_bar(bar)
        if theme not in by_theme:
            by_theme[theme] = []
        by_theme[theme].append(bar)
    
    # Print by theme
    for theme in ["CLIMATE", "ECONOMICS", "ROME", "TECHNOLOGY", "OTHER"]:
        if theme not in by_theme:
            continue
        bars = by_theme[theme]
        print(f"\n--- {theme} THEME ({len(bars)} bars) ---")
        
        for bar in bars[:4]:  # Show top 4 per theme
            print(f"\n  {bar['id']} [H₀]")
            print(f"    Persistence: [{bar['birth']:.3f}, {bar['death']:.3f}) = {bar['persistence']:.3f}")
            
            tokens = bar['witness']['tokens']
            token_str = ', '.join(tokens['surface'][:6])
            if len(tokens['surface']) > 6:
                token_str += f", ... ({len(tokens['surface'])} total)"
            print(f"    Witness tokens: {{{token_str}}}")
            
            utts = bar['witness']['utterances']['ids']
            print(f"    Utterances: {utts}")
    
    # =========================================================================
    # H₁ BARS: SEMANTIC LOOPS
    # =========================================================================
    if h1_bars:
        print()
        print()
        print("=" * 72)
        print("H₁ BARS: SEMANTIC LOOPS (CYCLES)")
        print("=" * 72)
        print()
        print("Each H₁ bar represents a loop in the embedding space: a cycle of")
        print("tokens that are pairwise close but enclose a 'void' - meaning that")
        print("circulates without collapsing. The γ shows the edges forming the loop.")
        print()
        
        for bar in h1_bars:
            print(f"\n{'─' * 60}")
            print(f"{bar['id']} [H₁] ★ SEMANTIC LOOP")
            print(f"{'─' * 60}")
            print(f"  Persistence: [{bar['birth']:.3f}, {bar['death']:.3f}) = {bar['persistence']:.3f}")
            
            # Show the cycle structure (γ)
            cycle = bar['witness']['cycle']
            print(f"\n  γ (cycle representative):")
            print(f"    Dimension: {cycle['dimension']}")
            print(f"    Edges forming the loop:")
            
            # Build token lookup
            token_lookup = {}
            for tid, surface in zip(bar['witness']['tokens']['ids'], 
                                   bar['witness']['tokens']['surface']):
                token_lookup[tid] = surface
            
            for edge in cycle['simplices']:
                t1 = token_lookup.get(edge[0], edge[0])
                t2 = token_lookup.get(edge[1], edge[1])
                print(f"      {t1} ↔ {t2}")
            
            # Show witness tokens
            tokens = bar['witness']['tokens']
            print(f"\n  Witness W_ρ = {{{', '.join(tokens['surface'])}}}")
            
            # Show utterances
            utts = bar['witness']['utterances']
            print(f"\n  Measurement locations: {utts['ids']}")
            for uid, text in zip(utts['ids'], utts['text_samples']):
                preview = text[:70] + "..." if len(text) > 70 else text
                print(f"    [{uid}] \"{preview}\"")
    else:
        print()
        print("=" * 72)
        print("H₁ BARS: None found at this persistence threshold")
        print("=" * 72)
        print("(Try lowering min_persistence to 0.01)")
    
    # =========================================================================
    # INTERPRETATION FOR CHAPTER 4
    # =========================================================================
    print()
    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print()
    print("This single-slice analysis reveals the static shape of the conversation:")
    print()
    
    if "CLIMATE" in by_theme:
        print(f"• CLIMATE theme: {len(by_theme['CLIMATE'])} H₀ bars")
        print("  The climate discussion forms a coherent cluster in embedding space,")
        print("  witnessed by tokens like 'climate', 'carbon', 'emissions', 'renewable'.")
        print()
    
    if "ROME" in by_theme:
        print(f"• ROME theme: {len(by_theme['ROME'])} H₀ bars")
        print("  The Rome digression forms a SEPARATE component - it does not")
        print("  merge with climate at low scales, reflecting genuine topic change.")
        print()
    
    if "ECONOMICS" in by_theme:
        print(f"• ECONOMICS theme: {len(by_theme['ECONOMICS'])} H₀ bars")
        print("  Economic vocabulary ('growth', 'pricing', 'market') forms its own")
        print("  cluster, connecting to BOTH climate and general discussion.")
        print()
    
    if h1_bars:
        print(f"• SEMANTIC LOOPS: {len(h1_bars)} H₁ bars")
        print("  The H₁ bars capture circulation of meaning - places where concepts")
        print("  form a triangle of mutual proximity without collapsing to a point.")
        print("  These are the 'debt→credit→interest→loan→debt' structures that")
        print("  Chapter 4 describes as 'meaning that refuses to close'.")
    
    print()
    print("The witnessed bars give us not just THAT features exist, but")
    print("WHAT they are about. This is the key contribution: différance")
    print("with a face.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
