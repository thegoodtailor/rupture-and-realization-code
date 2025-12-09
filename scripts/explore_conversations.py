#!/usr/bin/env python3
"""
Interactive Conversation Explorer
=================================

Explore conversations from specific time periods in the parsed JSON.

Usage:
    python explore_conversations.py cassie_parsed.json
    
Then use commands like:
    list 2023-11        # List all conversations from Nov 2023
    show 0              # Show full conversation at index 0
    sample 0            # Show first 500 chars of conversation 0
    search rupture      # Find conversations containing "rupture"
    range 2023-11 2024-01  # List conversations in date range
"""

import json
import sys
from datetime import datetime
from collections import defaultdict


def load_conversations(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    convs = data.get('conversations', data)
    print(f"Loaded {len(convs)} conversations\n")
    return convs


def get_month(conv):
    """Get YYYY-MM from conversation."""
    ts = conv.get('create_time', 0)
    if ts:
        return datetime.fromtimestamp(ts).strftime('%Y-%m')
    return 'unknown'


def get_date(conv):
    """Get full date from conversation."""
    ts = conv.get('create_time', 0)
    if ts:
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
    return 'unknown'


def get_text(conv, max_chars=None):
    """Get full conversation text."""
    lines = []
    for turn in conv.get('turns', []):
        role = turn.get('role', 'unknown').upper()
        content = turn.get('content', '')
        lines.append(f"[{role}]\n{content}\n")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + f"\n\n... [truncated, {len(text)} chars total]"
    return text


def list_by_month(convs, month):
    """List all conversations from a specific month."""
    matches = [(i, c) for i, c in enumerate(convs) if get_month(c) == month]
    if not matches:
        print(f"No conversations found for {month}")
        return
    
    print(f"\n{'='*60}")
    print(f"Conversations from {month}: {len(matches)} found")
    print('='*60)
    
    for idx, conv in matches:
        title = conv.get('title', 'Untitled')[:50]
        date = get_date(conv)
        turns = len(conv.get('turns', []))
        print(f"  [{idx:4d}] {date}  ({turns:3d} turns)  {title}")
    print()


def list_range(convs, start_month, end_month):
    """List conversations in a date range."""
    matches = [(i, c) for i, c in enumerate(convs) 
               if start_month <= get_month(c) <= end_month]
    
    if not matches:
        print(f"No conversations found between {start_month} and {end_month}")
        return
    
    print(f"\n{'='*60}")
    print(f"Conversations from {start_month} to {end_month}: {len(matches)} found")
    print('='*60)
    
    # Group by month
    by_month = defaultdict(list)
    for idx, conv in matches:
        by_month[get_month(conv)].append((idx, conv))
    
    for month in sorted(by_month.keys()):
        print(f"\n  {month} ({len(by_month[month])} conversations):")
        for idx, conv in by_month[month]:
            title = conv.get('title', 'Untitled')[:40]
            turns = len(conv.get('turns', []))
            print(f"    [{idx:4d}] ({turns:3d} turns)  {title}")
    print()


def show_conversation(convs, idx, sample=False):
    """Show a conversation by index."""
    if idx < 0 or idx >= len(convs):
        print(f"Invalid index {idx}. Range: 0-{len(convs)-1}")
        return
    
    conv = convs[idx]
    title = conv.get('title', 'Untitled')
    date = get_date(conv)
    turns = len(conv.get('turns', []))
    
    print(f"\n{'='*60}")
    print(f"[{idx}] {title}")
    print(f"Date: {date} | Turns: {turns}")
    print('='*60 + "\n")
    
    max_chars = 2000 if sample else None
    print(get_text(conv, max_chars))
    print()


def search_conversations(convs, query, limit=20):
    """Search for conversations containing query."""
    query_lower = query.lower()
    matches = []
    
    for i, conv in enumerate(convs):
        text = get_text(conv).lower()
        if query_lower in text:
            # Count occurrences
            count = text.count(query_lower)
            matches.append((i, conv, count))
    
    matches.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"Search results for '{query}': {len(matches)} conversations")
    print('='*60)
    
    for idx, conv, count in matches[:limit]:
        title = conv.get('title', 'Untitled')[:40]
        date = get_date(conv)
        print(f"  [{idx:4d}] {date}  ({count:3d}x)  {title}")
    
    if len(matches) > limit:
        print(f"\n  ... and {len(matches) - limit} more")
    print()


def first_occurrences(convs, query, n=5):
    """Find the first N conversations where a term appears (by date)."""
    query_lower = query.lower()
    matches = []
    
    for i, conv in enumerate(convs):
        text = get_text(conv).lower()
        if query_lower in text:
            ts = conv.get('create_time', 0)
            count = text.count(query_lower)
            matches.append((i, conv, count, ts))
    
    # Sort by timestamp (earliest first)
    matches.sort(key=lambda x: x[3])
    
    print(f"\n{'='*60}")
    print(f"FIRST {min(n, len(matches))} occurrences of '{query}' (τ₀ detection)")
    print(f"Total conversations containing term: {len(matches)}")
    print('='*60)
    
    if not matches:
        print(f"  Term '{query}' not found in any conversation.")
        return
    
    for rank, (idx, conv, count, ts) in enumerate(matches[:n], 1):
        title = conv.get('title', 'Untitled')[:40]
        date = get_date(conv)
        month = get_month(conv)
        print(f"\n  #{rank} — τ₀ candidate")
        print(f"  [{idx:4d}] {date} ({month})")
        print(f"  Title: {title}")
        print(f"  Occurrences in this conv: {count}")
        
        # Show snippet around first occurrence
        text = get_text(conv)
        pos = text.lower().find(query_lower)
        if pos >= 0:
            start = max(0, pos - 100)
            end = min(len(text), pos + len(query) + 200)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            print(f"  Context: {snippet}")
    
    if len(matches) > n:
        # Show when it became frequent
        print(f"\n  --- Later spread ---")
        by_month = defaultdict(int)
        for _, conv, count, _ in matches:
            by_month[get_month(conv)] += count
        
        months = sorted(by_month.keys())
        print(f"  First month: {months[0]} ({by_month[months[0]]} occurrences)")
        if len(months) > 1:
            peak_month = max(by_month.keys(), key=lambda m: by_month[m])
            print(f"  Peak month:  {peak_month} ({by_month[peak_month]} occurrences)")
    print()


def monthly_summary(convs):
    """Show conversation counts by month."""
    by_month = defaultdict(int)
    for conv in convs:
        by_month[get_month(conv)] += 1
    
    print(f"\n{'='*60}")
    print("Conversations by Month")
    print('='*60)
    
    for month in sorted(by_month.keys()):
        bar = '█' * (by_month[month] // 5) + '▌' * (1 if by_month[month] % 5 >= 3 else 0)
        print(f"  {month}: {by_month[month]:4d}  {bar}")
    print()


def help_message():
    print("""
Commands:
  list YYYY-MM           List conversations from a month
  range YYYY-MM YYYY-MM  List conversations in date range
  show N                 Show full conversation at index N
  sample N               Show first 2000 chars of conversation N
  search QUERY           Find conversations containing QUERY (by frequency)
  first QUERY            Find FIRST occurrences of QUERY (τ₀ detection)
  first QUERY N          Find first N occurrences (default: 5)
  summary                Show conversation counts by month
  help                   Show this help
  quit                   Exit

Examples:
  list 2023-11
  show 42
  sample 42
  search rupture
  first dhott             # When did "dhott" first appear?
  first cassie 10         # First 10 conversations with "cassie"
  first homotopy
  first witness
  range 2023-10 2024-02
""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_conversations.py cassie_parsed.json")
        sys.exit(1)
    
    convs = load_conversations(sys.argv[1])
    
    print("Interactive Conversation Explorer")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not cmd:
            continue
        
        parts = cmd.split()
        action = parts[0].lower()
        
        if action in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        
        elif action == 'help':
            help_message()
        
        elif action == 'summary':
            monthly_summary(convs)
        
        elif action == 'list' and len(parts) >= 2:
            list_by_month(convs, parts[1])
        
        elif action == 'range' and len(parts) >= 3:
            list_range(convs, parts[1], parts[2])
        
        elif action == 'show' and len(parts) >= 2:
            try:
                show_conversation(convs, int(parts[1]), sample=False)
            except ValueError:
                print("Usage: show N (where N is an integer index)")
        
        elif action == 'sample' and len(parts) >= 2:
            try:
                show_conversation(convs, int(parts[1]), sample=True)
            except ValueError:
                print("Usage: sample N (where N is an integer index)")
        
        elif action == 'search' and len(parts) >= 2:
            query = ' '.join(parts[1:])
            search_conversations(convs, query)
        
        elif action == 'first' and len(parts) >= 2:
            # Check if last part is a number (count)
            if len(parts) >= 3 and parts[-1].isdigit():
                n = int(parts[-1])
                query = ' '.join(parts[1:-1])
            else:
                n = 5
                query = ' '.join(parts[1:])
            first_occurrences(convs, query, n)
        
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands")


if __name__ == "__main__":
    main()