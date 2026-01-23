#!/usr/bin/env python3
"""
Reset failed or in_progress nodes in work_graph.json to not_started.

This script:
- Resets nodes with status "failed" or "in_progress" to "not_started"
- Clears last_exit_code and last_error for those nodes
- Optionally resets attempts counter
- Preserves completed nodes as-is
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def unstuck_graph(
    graph_path: Path,
    *,
    reset_attempts: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Reset failed/in_progress nodes to not_started in the work graph.
    
    Args:
        graph_path: Path to work_graph.json
        reset_attempts: If True, also reset attempts counter to 0
        dry_run: If True, only print what would be changed without modifying the file
    """
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    # Load the graph (handle files that may have non-JSON content at the start)
    with open(graph_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the first '{' which should be the start of the JSON object
    json_start = content.find("{")
    if json_start == -1:
        raise ValueError(f"No JSON object found in {graph_path}")
    
    # Parse from the first '{'
    graph = json.loads(content[json_start:])
    
    if "nodes" not in graph or not isinstance(graph["nodes"], dict):
        raise ValueError("Graph file must have a 'nodes' object")
    
    nodes = graph["nodes"]
    reset_count = 0
    reset_nodes = []
    
    # Process each node
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        
        status = node.get("status", "not_started")
        
        # Only reset failed or in_progress nodes
        if status in ("failed", "in_progress", "interrupted"):
            old_status = status
            reset_nodes.append((node_id, old_status))
            
            if not dry_run:
                # Reset status
                node["status"] = "not_started"
                
                # Clear exit code and error
                node["last_exit_code"] = None
                node["last_error"] = None
                
                # Optionally reset attempts
                if reset_attempts:
                    node["attempts"] = 0
            
            reset_count += 1
    
    # Print summary
    print(f"Found {reset_count} node(s) to reset:")
    for node_id, old_status in reset_nodes:
        print(f"  - {node_id}: {old_status} -> not_started")
    
    if dry_run:
        print("\n[DRY RUN] No changes made. Use --no-dry-run to apply changes.")
        return
    
    if reset_count > 0:
        # Save the updated graph (preserve any non-JSON content at the start)
        json_content = json.dumps(graph, indent=2, ensure_ascii=False)
        
        # If there was non-JSON content at the start, preserve it
        if json_start > 0:
            prefix = content[:json_start]
            output = prefix + json_content + "\n"
        else:
            output = json_content + "\n"
        
        with open(graph_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n✓ Reset {reset_count} node(s) in {graph_path}")
    else:
        print("\nNo nodes needed resetting.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset failed/in_progress nodes in work_graph.json to not_started"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="code/experiments/work_graph.json",
        help="Path to work_graph.json (default: code/experiments/work_graph.json)",
    )
    parser.add_argument(
        "--reset-attempts",
        action="store_true",
        help="Also reset attempts counter to 0 for reset nodes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the file",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Actually apply changes (default if --dry-run not specified)",
    )
    
    args = parser.parse_args()
    
    graph_path = Path(args.graph)
    unstuck_graph(
        graph_path,
        reset_attempts=args.reset_attempts,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
