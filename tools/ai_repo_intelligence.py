#!/usr/bin/env python3
"""Deterministic, Git-aware repository intelligence for CALO-RPD."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from ai_intelligence.audit_seed import apply_audit_seed
from ai_intelligence.common import canonical_json, repo_root
from ai_intelligence.context import build_context
from ai_intelligence.indexer import current_status, update_index, validate_indexes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="repository root (auto-detected)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="build the initial repository intelligence indexes")
    commands.add_parser("update", help="incrementally refresh changed files and dependent metadata")
    status = commands.add_parser("status", help="show whether committed metadata matches source")
    status.add_argument("--json", action="store_true", dest="as_json")
    commands.add_parser("check", help="fail when source differs from committed AI metadata")
    commands.add_parser("validate", help="check cross-index referential integrity")
    context = commands.add_parser("context", help="build a compact question-specific ChatGPT context")
    context.add_argument("question")
    context.add_argument("--limit", type=int, default=6)
    args = parser.parse_args(argv)
    root = repo_root(args.root)

    if args.command in {"init", "update"}:
        result = update_index(root, initialize=args.command == "init")
        apply_audit_seed(root)
        print(canonical_json(result), end="")
        return 0
    if args.command == "status":
        result = current_status(root)
        if args.as_json:
            print(canonical_json(result), end="")
        else:
            print("AI index: " + ("STALE" if result.get("stale") else "CURRENT"))
            for key in ("modified", "added", "deleted"):
                if values := result.get(key, []):
                    print(f"{key}: {len(values)}")
                    for value in values[:20]:
                        print(f"  {value}")
        return 1 if result.get("stale") else 0
    if args.command == "check":
        status = current_status(root)
        if status.get("stale"):
            print(canonical_json(status), end="", file=sys.stderr)
            return 1
        errors = validate_indexes(root)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 2
        print("AI repository intelligence is current and internally consistent.")
        return 0
    if args.command == "validate":
        errors = validate_indexes(root)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("AI repository intelligence indexes are internally consistent.")
        return 0
    if args.command == "context":
        print(build_context(root, args.question, max(1, min(args.limit, 12))), end="")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
