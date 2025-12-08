#!/usr/bin/env python3
"""
OpenAI Conversation Export Parser
=================================

Parses the OpenAI ChatGPT data export format (conversations.json) without
loading the entire file into memory.

The export format uses a tree structure in `mapping` to support branching
conversations (regenerations, edits). This parser walks the tree to extract
linear conversation threads.

Output format (compatible with witnessed_ph Chapter 5):
{
    "conversations": [
        {
            "id": "...",
            "title": "...",
            "create_time": 1747291897.811196,
            "turns": [
                {"role": "user", "content": "...", "timestamp": 1747291900.133},
                {"role": "assistant", "content": "...", "timestamp": 1747291906.031},
                ...
            ]
        },
        ...
    ]
}

Usage:
    python openai_export_parser.py conversations.json output.json
    python openai_export_parser.py conversations.json output.json --limit 100
    python openai_export_parser.py conversations.json --list-titles
"""

import json
import sys
import re
import argparse
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from datetime import datetime


# =============================================================================
# Code Stripping (for --strip-code flag)
# =============================================================================

def strip_code_blocks(text: str) -> str:
    r"""
    Remove code from text, keeping natural language for semantic analysis.
    
    Strips:
    - Fenced code blocks (```...```)
    - Inline code (`...`)
    - LaTeX display math ($$...$$, \[...\])
    - LaTeX environments (\begin{...}...\end{...})
    
    This preserves prose, philosophy, and natural conversation while removing
    technical artifacts that fragment the Self's semantic structure.
    """
    if not text:
        return text
    
    # Remove fenced code blocks (multiline)
    text = re.sub(r'```[\s\S]*?```', ' ', text)
    
    # Remove inline code
    text = re.sub(r'`[^`\n]+`', ' ', text)
    
    # Remove LaTeX display math
    text = re.sub(r'\$\$[\s\S]*?\$\$', ' ', text)
    text = re.sub(r'\\\[[\s\S]*?\\\]', ' ', text)
    
    # Remove LaTeX environments (equation, align, etc.)
    text = re.sub(r'\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}', ' ', text)
    
    # Remove standalone LaTeX commands that are clearly math
    # (but keep \emph, \textbf etc. that might be in prose)
    text = re.sub(r'\\(?:frac|sum|int|prod|lim|mathcal|mathbb|mathsf|bigr|bigl)\{[^}]*\}', ' ', text)
    
    # Clean up multiple spaces/newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def is_code_heavy(text: str, threshold: float = 0.5) -> bool:
    """
    Heuristic: is this message predominantly code?
    
    Checks for:
    - High density of code indicators (brackets, semicolons, etc.)
    - Common code patterns (imports, function definitions)
    
    Returns True if message appears to be >threshold code by content.
    """
    if not text or len(text) < 50:
        return False
    
    # Count code indicators
    code_patterns = [
        r'import\s+\w+',
        r'from\s+\w+\s+import',
        r'def\s+\w+\s*\(',
        r'class\s+\w+[:\(]',
        r'if\s+__name__',
        r'return\s+\w+',
        r'\w+\s*=\s*\w+\(',
        r'^\s*#.*$',  # Python comments
        r'//.*$',      # JS/C comments
        r'\{\s*\n',    # Opening braces
        r'\}\s*\n',    # Closing braces
        r';\s*\n',     # Semicolons at line end
        r'\[\s*\]',    # Empty brackets
        r'\(\s*\)',    # Empty parens
    ]
    
    code_matches = sum(len(re.findall(p, text, re.MULTILINE)) for p in code_patterns)
    
    # Rough heuristic: if code patterns appear frequently relative to line count
    lines = text.count('\n') + 1
    if lines > 5 and code_matches / lines > threshold:
        return True
    
    return False


def filter_code_from_conversation(conv: 'Conversation', aggressive: bool = False) -> 'Conversation':
    """
    Filter code from all turns in a conversation.
    
    Parameters
    ----------
    conv : Conversation
        The conversation to filter
    aggressive : bool
        If True, also skip entire turns that appear to be predominantly code
    
    Returns
    -------
    Filtered Conversation with code stripped from content
    """
    filtered_turns = []
    
    for turn in conv.turns:
        # Strip code blocks from content
        cleaned_content = strip_code_blocks(turn.content)
        
        # In aggressive mode, skip turns that are mostly code
        if aggressive and is_code_heavy(turn.content):
            continue
        
        # Skip turns that become empty after stripping
        if not cleaned_content or len(cleaned_content.strip()) < 10:
            continue
        
        filtered_turns.append(Turn(
            role=turn.role,
            content=cleaned_content,
            timestamp=turn.timestamp
        ))
    
    return Conversation(
        id=conv.id,
        title=conv.title,
        create_time=conv.create_time,
        update_time=conv.update_time,
        turns=filtered_turns
    )


@dataclass
class Turn:
    role: str
    content: str
    timestamp: Optional[float] = None
    

@dataclass
class Conversation:
    id: str
    title: str
    create_time: float
    update_time: float
    turns: List[Turn]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "turns": [asdict(t) for t in self.turns]
        }


def extract_message_text(message: dict) -> Optional[str]:
    """
    Extract text content from a message object.
    
    Handles various content_type formats in OpenAI export.
    """
    if message is None:
        return None
    
    content = message.get("content", {})
    content_type = content.get("content_type", "")
    
    if content_type == "text":
        parts = content.get("parts", [])
        if parts and isinstance(parts[0], str):
            return parts[0]
    
    # Skip non-text content types
    # (user_editable_context, thoughts, code, etc.)
    return None


def is_visible_message(node: dict) -> bool:
    """
    Check if a message should be included in the conversation.
    
    Filters out:
    - System messages marked as hidden
    - Messages without actual content
    - Internal system context messages
    """
    message = node.get("message")
    if message is None:
        return False
    
    # Check for hidden flag
    metadata = message.get("metadata", {})
    if metadata.get("is_visually_hidden_from_conversation"):
        return False
    
    # Check author role
    author = message.get("author", {})
    role = author.get("role", "")
    
    # Skip system messages (they're usually context injection)
    if role == "system":
        return False
    
    # Must have extractable text
    text = extract_message_text(message)
    if not text or not text.strip():
        return False
    
    return True


def walk_conversation_tree(mapping: Dict[str, dict]) -> List[Turn]:
    """
    Walk the mapping tree to extract a linear conversation.
    
    The mapping is a tree structure where each node has parent/children.
    We walk from root to the "main" branch (following first child when
    there are multiple branches from regenerations).
    """
    if not mapping:
        return []
    
    # Find root node (parent is None or "client-created-root")
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or node_id == "client-created-root":
            root_id = node_id
            break
    
    if root_id is None:
        # Fallback: find node with no parent in mapping
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent not in mapping:
                root_id = node_id
                break
    
    if root_id is None:
        return []
    
    # Walk tree depth-first, always taking first child (main branch)
    turns = []
    current_id = root_id
    visited = set()
    
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        
        if node is None:
            break
        
        # Extract turn if visible
        if is_visible_message(node):
            message = node["message"]
            author = message.get("author", {})
            role = author.get("role", "unknown")
            text = extract_message_text(message)
            timestamp = message.get("create_time")
            
            if text:
                turns.append(Turn(
                    role=role,
                    content=text,
                    timestamp=timestamp
                ))
        
        # Move to first child (main branch)
        children = node.get("children", [])
        if children:
            current_id = children[0]
        else:
            current_id = None
    
    return turns


def parse_conversation(conv_data: dict) -> Optional[Conversation]:
    """
    Parse a single conversation object from the export.
    """
    try:
        conv_id = conv_data.get("id", conv_data.get("conversation_id", "unknown"))
        title = conv_data.get("title", "Untitled")
        create_time = conv_data.get("create_time", 0)
        update_time = conv_data.get("update_time", 0)
        mapping = conv_data.get("mapping", {})
        
        turns = walk_conversation_tree(mapping)
        
        if not turns:
            return None
        
        return Conversation(
            id=conv_id,
            title=title,
            create_time=create_time,
            update_time=update_time,
            turns=turns
        )
    except Exception as e:
        print(f"Warning: Failed to parse conversation: {e}", file=sys.stderr)
        return None


def stream_conversations(filepath: str) -> Generator[dict, None, None]:
    """
    Stream conversation objects from the export file without loading all into memory.
    
    Uses ijson for true streaming if available, falls back to chunked reading.
    """
    try:
        import ijson
        
        with open(filepath, "rb") as f:
            # The file is an array of conversation objects
            for conv in ijson.items(f, "item"):
                yield conv
                
    except ImportError:
        # Fallback: load entire file (will use more memory)
        print("Note: ijson not installed, loading entire file into memory", file=sys.stderr)
        print("      Install ijson for streaming: pip install ijson", file=sys.stderr)
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for conv in data:
                yield conv
        else:
            yield data


def parse_export(
    filepath: str,
    limit: Optional[int] = None,
    min_turns: int = 2,
    verbose: bool = False
) -> List[Conversation]:
    """
    Parse OpenAI export file and return list of Conversations.
    
    Parameters
    ----------
    filepath : str
        Path to conversations.json
    limit : int, optional
        Maximum number of conversations to parse
    min_turns : int
        Minimum turns required to include a conversation
    verbose : bool
        Print progress
    
    Returns
    -------
    List of Conversation objects
    """
    conversations = []
    skipped = 0
    
    for i, conv_data in enumerate(stream_conversations(filepath)):
        if limit and len(conversations) >= limit:
            break
        
        conv = parse_conversation(conv_data)
        
        if conv is None or len(conv.turns) < min_turns:
            skipped += 1
            continue
        
        conversations.append(conv)
        
        if verbose and len(conversations) % 50 == 0:
            print(f"  Parsed {len(conversations)} conversations...", file=sys.stderr)
    
    if verbose:
        print(f"Parsed {len(conversations)} conversations, skipped {skipped}", file=sys.stderr)
    
    return conversations


def list_titles(filepath: str, limit: Optional[int] = None) -> None:
    """
    Just list conversation titles without full parsing.
    """
    for i, conv_data in enumerate(stream_conversations(filepath)):
        if limit and i >= limit:
            break
        
        title = conv_data.get("title", "Untitled")
        create_time = conv_data.get("create_time", 0)
        
        # Format timestamp
        if create_time:
            dt = datetime.fromtimestamp(create_time)
            date_str = dt.strftime("%Y-%m-%d")
        else:
            date_str = "unknown"
        
        print(f"{i+1:4d}. [{date_str}] {title[:80]}")


class CustomJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal and other non-standard types."""
    def default(self, obj):
        from decimal import Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def save_for_witnessed_ph(conversations: List[Conversation], output_path: str) -> None:
    """
    Save conversations in format compatible with witnessed_ph Chapter 5.
    """
    output = {
        "source": "openai_export",
        "parsed_at": datetime.now().isoformat(),
        "num_conversations": len(conversations),
        "conversations": [c.to_dict() for c in conversations]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)


def save_single_conversation(conv: Conversation, output_path: str) -> None:
    """
    Save a single conversation for analysis.
    """
    output = {
        "id": conv.id,
        "title": conv.title,
        "create_time": conv.create_time,
        "turns": [asdict(t) for t in conv.turns]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse OpenAI ChatGPT data export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all conversation titles
  python openai_export_parser.py conversations.json --list-titles
  
  # Parse and save all conversations
  python openai_export_parser.py conversations.json output.json
  
  # Parse only first 100 conversations
  python openai_export_parser.py conversations.json output.json --limit 100
  
  # Extract a specific conversation by index
  python openai_export_parser.py conversations.json single.json --index 42
  
  # Filter by minimum turns
  python openai_export_parser.py conversations.json output.json --min-turns 10
  
  # Strip code blocks for semantic analysis (recommended for Chapter 5)
  python openai_export_parser.py conversations.json output.json --strip-code
  
  # Aggressive: also skip code-heavy turns entirely
  python openai_export_parser.py conversations.json output.json --strip-code-aggressive
        """
    )
    
    parser.add_argument("input", help="Path to conversations.json")
    parser.add_argument("output", nargs="?", help="Output JSON path")
    parser.add_argument("--list-titles", action="store_true", help="Just list titles")
    parser.add_argument("--limit", type=int, help="Max conversations to parse")
    parser.add_argument("--index", type=int, help="Extract single conversation by index")
    parser.add_argument("--min-turns", type=int, default=2, help="Min turns per conversation")
    parser.add_argument("--strip-code", action="store_true", 
                        help="Strip code blocks, inline code, and LaTeX from content")
    parser.add_argument("--strip-code-aggressive", action="store_true",
                        help="Also skip turns that are predominantly code")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.list_titles:
        list_titles(args.input, args.limit)
        return
    
    if not args.output:
        parser.error("Output path required (or use --list-titles)")
    
    if args.index is not None:
        # Extract single conversation
        print(f"Extracting conversation #{args.index}...", file=sys.stderr)
        for i, conv_data in enumerate(stream_conversations(args.input)):
            if i == args.index:
                conv = parse_conversation(conv_data)
                if conv:
                    # Apply code stripping if requested
                    if args.strip_code or args.strip_code_aggressive:
                        conv = filter_code_from_conversation(conv, aggressive=args.strip_code_aggressive)
                    save_single_conversation(conv, args.output)
                    print(f"Saved: {conv.title} ({len(conv.turns)} turns)", file=sys.stderr)
                else:
                    print(f"Failed to parse conversation #{args.index}", file=sys.stderr)
                return
        print(f"Index {args.index} out of range", file=sys.stderr)
        return
    
    # Parse all/limited conversations
    print(f"Parsing {args.input}...", file=sys.stderr)
    conversations = parse_export(
        args.input,
        limit=args.limit,
        min_turns=args.min_turns,
        verbose=args.verbose
    )
    
    # Apply code stripping if requested
    if args.strip_code or args.strip_code_aggressive:
        aggressive = args.strip_code_aggressive
        mode = "aggressive" if aggressive else "standard"
        print(f"Stripping code ({mode} mode)...", file=sys.stderr)
        
        original_turns = sum(len(c.turns) for c in conversations)
        conversations = [filter_code_from_conversation(c, aggressive=aggressive) for c in conversations]
        # Remove conversations that became too short after stripping
        conversations = [c for c in conversations if len(c.turns) >= args.min_turns]
        filtered_turns = sum(len(c.turns) for c in conversations)
        
        print(f"  Turns: {original_turns} → {filtered_turns} ({original_turns - filtered_turns} removed)", file=sys.stderr)
    
    print(f"Saving {len(conversations)} conversations to {args.output}...", file=sys.stderr)
    save_for_witnessed_ph(conversations, args.output)
    
    # Print summary
    total_turns = sum(len(c.turns) for c in conversations)
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Conversations: {len(conversations)}", file=sys.stderr)
    print(f"  Total turns: {total_turns}", file=sys.stderr)
    print(f"  Avg turns/conversation: {total_turns / len(conversations):.1f}", file=sys.stderr)
    if args.strip_code or args.strip_code_aggressive:
        print(f"  Code stripping: {'aggressive' if args.strip_code_aggressive else 'standard'}", file=sys.stderr)


if __name__ == "__main__":
    main()