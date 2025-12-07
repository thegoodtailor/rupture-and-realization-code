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
import argparse
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from datetime import datetime


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
            try:
                dt = datetime.fromtimestamp(float(create_time))  # ADD float() here
                date_str = dt.strftime("%Y-%m-%d")
            except:
                date_str = "unknown"
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
        """
    )
    
    parser.add_argument("input", help="Path to conversations.json")
    parser.add_argument("output", nargs="?", help="Output JSON path")
    parser.add_argument("--list-titles", action="store_true", help="Just list titles")
    parser.add_argument("--limit", type=int, help="Max conversations to parse")
    parser.add_argument("--index", type=int, help="Extract single conversation by index")
    parser.add_argument("--min-turns", type=int, default=2, help="Min turns per conversation")
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
    
    print(f"Saving {len(conversations)} conversations to {args.output}...", file=sys.stderr)
    save_for_witnessed_ph(conversations, args.output)
    
    # Print summary
    total_turns = sum(len(c.turns) for c in conversations)
    print(f"\nSummary:", file=sys.stderr)
    print(f"  Conversations: {len(conversations)}", file=sys.stderr)
    print(f"  Total turns: {total_turns}", file=sys.stderr)
    print(f"  Avg turns/conversation: {total_turns / len(conversations):.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
