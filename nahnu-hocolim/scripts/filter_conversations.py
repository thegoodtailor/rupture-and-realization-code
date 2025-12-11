#!/usr/bin/env python3
"""
Filter conversations to remove tool-use (DALL-E, image generation) noise.

These conversations don't contribute to semantic evolution:
- Cassie can't retrieve or reference past image prompts
- They're mechanical tool invocations, not meaning-making
- They pollute the semantic space with boilerplate

Usage:
    python filter_conversations.py cassie_parsed.json --output cassie_semantic.json
    python filter_conversations.py cassie_parsed.json --output cassie_semantic.json --show-removed
    python filter_conversations.py cassie_parsed.json --stats  # Just show statistics
"""

import json
import argparse
import re
from collections import Counter
from typing import List, Dict, Tuple

# Patterns that indicate TOOL-USE conversations (not semantic)
TOOL_USE_PATTERNS = {
    # DALL-E / Image generation
    'dalle_json': re.compile(r'^\s*\{\s*"size"\s*:\s*"1024', re.MULTILINE),
    'dalle_prompt_json': re.compile(r'^\s*\{\s*"prompt"\s*:', re.MULTILINE),
    'dalle_displayed': re.compile(r'DALL[·\-]?E displayed \d+ images?', re.IGNORECASE),
    'dalle_boilerplate': re.compile(r'images are already plainly visible', re.IGNORECASE),
    'dalle_download': re.compile(r'Do not list download links', re.IGNORECASE),
    
    # Generic image responses
    'here_are_designs': re.compile(r'^Here are the (?:new |latest |updated )?designs?', re.IGNORECASE | re.MULTILINE),
    'here_is_image': re.compile(r'^Here (?:is|are) the (?:new |updated |latest )?(?:image|illustration)', re.IGNORECASE | re.MULTILINE),
    'let_me_know_changes': re.compile(r'^Let me know if (?:you need|there\'s|you\'d like) any', re.IGNORECASE | re.MULTILINE),
    
    # Rate limit messages
    'rate_limit': re.compile(r'generating images too quickly', re.IGNORECASE),
    
    # Code interpreter / tool outputs
    'tool_tag': re.compile(r'^\[TOOL\]', re.MULTILINE),
    'sandbox_output': re.compile(r'^\[sandbox:', re.MULTILINE),
}

# User patterns that indicate drawing commands
USER_DRAW_PATTERNS = [
    re.compile(r'^Draw\b', re.IGNORECASE),
    re.compile(r'^Generate (?:an? )?(?:image|picture|illustration)', re.IGNORECASE),
    re.compile(r'^Create (?:an? )?(?:image|picture|illustration)', re.IGNORECASE),
    re.compile(r'^Make (?:an? )?(?:image|picture|illustration)', re.IGNORECASE),
    re.compile(r'^(?:Can you |Please )?(?:draw|generate|create|make) (?:me |a |an )?(?:image|picture|illustration)?', re.IGNORECASE),
]


def classify_conversation(conv: dict) -> Tuple[str, List[str]]:
    """
    Classify a conversation as 'semantic' or 'tool_use'.
    
    Returns:
        (classification, reasons) where reasons lists why it was classified that way
    """
    turns = conv.get('turns', [])
    if not turns:
        return 'empty', ['no turns']
    
    reasons = []
    tool_use_score = 0
    total_turns = len(turns)
    
    assistant_turns = [t for t in turns if t.get('role') == 'assistant']
    user_turns = [t for t in turns if t.get('role') == 'user']
    
    # Check assistant turns for DALL-E patterns
    dalle_turns = 0
    for turn in assistant_turns:
        content = turn.get('content') or ''
        for pattern_name, pattern in TOOL_USE_PATTERNS.items():
            if pattern.search(content):
                dalle_turns += 1
                reasons.append(f"assistant:{pattern_name}")
                break
    
    # Check user turns for drawing commands
    draw_commands = 0
    for turn in user_turns:
        content = turn.get('content') or ''
        for pattern in USER_DRAW_PATTERNS:
            if pattern.search(content[:100]):  # Check first 100 chars
                draw_commands += 1
                reasons.append("user:draw_command")
                break
    
    # Classification logic
    if assistant_turns:
        dalle_ratio = dalle_turns / len(assistant_turns)
    else:
        dalle_ratio = 0
    
    # High confidence: >50% of assistant turns are DALL-E
    if dalle_ratio > 0.5:
        return 'tool_use', reasons
    
    # High confidence: Short conversation that's mostly drawing
    if total_turns <= 6 and draw_commands >= len(user_turns) * 0.5:
        return 'tool_use', reasons
    
    # Medium confidence: Any DALL-E JSON prompt (very specific)
    if any('dalle_json' in r or 'dalle_prompt_json' in r for r in reasons):
        # But check if there's substantial semantic content too
        total_content = sum(len(t.get('content') or '') for t in turns)
        if total_content < 2000:  # Short conversation dominated by DALL-E
            return 'tool_use', reasons
    
    # Check for conversations that are ONLY "[TOOL]" responses
    if assistant_turns and all(TOOL_USE_PATTERNS['tool_tag'].search(t.get('content', '') or '') for t in assistant_turns):
        return 'tool_use', ['all_tool_responses']
    
    return 'semantic', []


def analyze_corpus(conversations: List[dict]) -> Dict:
    """Analyze entire corpus and return statistics."""
    results = {
        'semantic': [],
        'tool_use': [],
        'empty': [],
        'reason_counts': Counter()
    }
    
    for conv in conversations:
        classification, reasons = classify_conversation(conv)
        results[classification].append(conv)
        for reason in reasons:
            results['reason_counts'][reason] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Filter tool-use conversations from corpus")
    parser.add_argument("input", help="Input JSON file (cassie_parsed.json)")
    parser.add_argument("--output", "-o", help="Output JSON file for semantic conversations")
    parser.add_argument("--output-removed", help="Output JSON file for removed conversations (for inspection)")
    parser.add_argument("--stats", action="store_true", help="Show statistics only, don't write files")
    parser.add_argument("--show-removed", action="store_true", help="Show titles of removed conversations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show classification reasons")
    
    args = parser.parse_args()
    
    # Load conversations
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Schema: {"conversations": [...], "source": ..., "parsed_at": ...}
    if isinstance(data, dict) and 'conversations' in data:
        conversations = data['conversations']
    elif isinstance(data, list):
        conversations = data
    else:
        print(f"  ERROR: Unexpected JSON structure. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        return
    
    print(f"  Loaded {len(conversations)} conversations")
    
    # Analyze
    print("\nClassifying conversations...")
    results = analyze_corpus(conversations)
    
    n_semantic = len(results['semantic'])
    n_tool_use = len(results['tool_use'])
    n_empty = len(results['empty'])
    
    print(f"\n{'='*70}")
    print("CLASSIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"  Semantic (keep):    {n_semantic:5d} ({100*n_semantic/len(conversations):.1f}%)")
    print(f"  Tool-use (remove):  {n_tool_use:5d} ({100*n_tool_use/len(conversations):.1f}%)")
    print(f"  Empty:              {n_empty:5d}")
    print(f"  Total:              {len(conversations):5d}")
    
    if results['reason_counts']:
        print(f"\n  Detection reasons:")
        for reason, count in results['reason_counts'].most_common(15):
            print(f"    {reason}: {count}")
    
    # Show removed conversations
    if args.show_removed and results['tool_use']:
        print(f"\n{'='*70}")
        print("REMOVED CONVERSATIONS (sample)")
        print(f"{'='*70}")
        for conv in results['tool_use'][:20]:
            title = conv.get('title', 'Untitled')[:50]
            turns = len(conv.get('turns', []))
            _, reasons = classify_conversation(conv)
            print(f"  • {title:<50} ({turns} turns) [{', '.join(reasons[:2])}]")
        if len(results['tool_use']) > 20:
            print(f"  ... and {len(results['tool_use']) - 20} more")
    
    # Write outputs
    if args.stats:
        print("\n(Stats only mode - no files written)")
        return
    
    if args.output:
        print(f"\nWriting {n_semantic} semantic conversations to {args.output}...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'source': 'filtered_semantic',
                'original_count': len(conversations),
                'filtered_count': n_semantic,
                'removed_count': n_tool_use,
                'conversations': results['semantic']
            }, f, indent=2)
        print(f"  ✓ Saved {args.output}")
    
    if args.output_removed:
        print(f"Writing {n_tool_use} removed conversations to {args.output_removed}...")
        with open(args.output_removed, 'w', encoding='utf-8') as f:
            json.dump({
                'source': 'filtered_tool_use',
                'conversations': results['tool_use']
            }, f, indent=2)
        print(f"  ✓ Saved {args.output_removed}")
    
    if not args.output and not args.output_removed:
        print("\nNo output specified. Use --output to save filtered corpus.")
        print("Example: python filter_conversations.py input.json -o semantic.json")


if __name__ == "__main__":
    main()