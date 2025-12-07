#!/usr/bin/env python3
"""
Chapter 5 Example: Self Analysis of a Conversation
===================================================

This script demonstrates the Self construction from Chapter 5.
It takes a conversation (as JSON or inline text), analyses it turn-by-turn,
builds the journey graph, and computes coherence metrics.

Usage:
    python chapter5_example.py                    # Use built-in example
    python chapter5_example.py conversation.json  # Analyse JSON file

Expected JSON format:
    {
        "turns": [
            {"speaker": "Human", "text": "..."},
            {"speaker": "Assistant", "text": "..."},
            ...
        ]
    }
    
    OR (OpenAI/Anthropic format):
    
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }

Output:
    - Self coherence metrics (fragmentation, reentry rate, etc.)
    - Journey lifespans and dynamics
    - Scheduler style diagnosis
    - Optional: JSON export for visualization
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# EXAMPLE CONVERSATION (Climate + Rome, matching Chapter 4)
# =============================================================================

EXAMPLE_CONVERSATION = {
    "turns": [
        {"speaker": "User", "text": "I've been thinking about climate change a lot lately."},
        {"speaker": "Assistant", "text": "Climate change is certainly one of the defining challenges of our time. What aspects concern you most?"},
        {"speaker": "User", "text": "The economic impacts worry me. How do we balance growth with sustainability?"},
        {"speaker": "Assistant", "text": "That tension between economic growth and environmental protection is real. Many economists now argue for sustainable development models."},
        {"speaker": "User", "text": "What about carbon pricing? Does it actually work?"},
        {"speaker": "Assistant", "text": "Carbon pricing can be effective when well-designed. It creates market incentives to reduce emissions while allowing flexibility in how reductions are achieved."},
        {"speaker": "User", "text": "Let's change topics. Tell me about ancient Rome."},
        {"speaker": "Assistant", "text": "Ancient Rome was a remarkable civilization that lasted over a thousand years. From a small city-state to a vast empire spanning many regions, it shaped law, architecture, and governance for centuries."},
        {"speaker": "User", "text": "Actually, let's go back to climate. What about renewable energy?"},
        {"speaker": "Assistant", "text": "Renewable energy has seen remarkable growth. Solar and wind costs have plummeted, making them competitive with fossil fuels in many regions."},
        {"speaker": "User", "text": "So there's hope for addressing climate change through technology?"},
        {"speaker": "Assistant", "text": "Technology is certainly part of the solution. Renewables, storage, and efficiency improvements all contribute to reducing carbon emissions."},
    ]
}

# =============================================================================
# MAIN
# =============================================================================

def main():
    from witnessed_ph import default_config
    from witnessed_ph.self_construction import (
        analyse_conversation_from_json,
        print_self_report,
        export_journey_graph_json
    )
    
    print("=" * 72)
    print("CHAPTER 5: THE SELF AS A SCHEDULED HOCOLIM OF JOURNEYS")
    print("=" * 72)
    print()
    
    # Load conversation
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        print(f"Loading conversation from: {json_path}")
        with open(json_path, 'r') as f:
            conversation = json.load(f)
    else:
        print("Using built-in example conversation (Climate + Rome)")
        conversation = EXAMPLE_CONVERSATION
    
    # Count turns
    turns = conversation.get("turns") or conversation.get("messages", [])
    print(f"Turns: {len(turns)}")
    print()
    
    # Configuration
    config = default_config()
    config["min_persistence"] = 0.03
    config["min_witness_tokens"] = 2
    
    print("Configuration:")
    print(f"  embedding_model: {config['embedding_model']}")
    print(f"  min_persistence: {config['min_persistence']}")
    print(f"  min_witness_tokens: {config['min_witness_tokens']}")
    print()
    
    # Run analysis
    print("Analysing conversation (this may take a moment)...")
    print()
    
    diagrams, graph, metrics = analyse_conversation_from_json(
        conversation,
        cumulative=True,  # Each turn sees all prior context
        config=config,
        verbose=True
    )
    
    # Print report
    print()
    report = print_self_report(graph, metrics, diagrams)
    print(report)
    
    # Export for visualization
    output_path = "self_journey_graph.json"
    export_journey_graph_json(graph, metrics, output_path)
    print(f"\n📁 Journey graph exported to: {output_path}")
    print("   (Load in D3.js / Gephi / NetworkX for visualization)")
    
    # Summary interpretation
    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print()
    
    print("The journey graph represents the Self as a connected structure of")
    print("witnessed bar journeys. Key observations:")
    print()
    
    if metrics.fragmentation_index < 0.5:
        print("✓ LOW FRAGMENTATION: The Self maintains coherence across time.")
        print("  Bar journeys share witness tokens, creating a connected structure.")
    else:
        print("⚠ HIGH FRAGMENTATION: The Self is disconnected.")
        print("  Themes operate in isolation without shared witnesses.")
    
    print()
    
    if metrics.reentry_count > 0:
        print(f"✓ RE-ENTRY DETECTED: {metrics.reentry_count} theme(s) returned after rupture.")
        print("  This is the 'Rome digression' pattern: climate ruptures at τ=6,")
        print("  but re-enters at τ=8 when the user returns to the topic.")
    else:
        print("• NO RE-ENTRIES: Once themes rupture, they don't return.")
    
    print()
    
    print("This is what Chapter 5 means by 'the Self is not a disjoint union")
    print("of biographies but a connected structure.' The hocolim glues journeys")
    print("along their witness relations, and the resulting coherence (or lack")
    print("thereof) tells us about the health of the evolving text.")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
