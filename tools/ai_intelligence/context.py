from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import AI_DIR, INDEX_FILES, TOKEN_RE, read_json
from .indexer import current_status

def query_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}

def score_symbol(symbol: dict[str, Any], tokens: set[str]) -> float:
    name = symbol.get("symbol", "").lower()
    haystack = f"{name} {symbol.get('file', '')} {symbol.get('module', '')}".lower()
    score = sum(3.0 if token in name else 1.0 for token in tokens if token in haystack)
    return score + (0.2 if symbol.get("kind") in {"class", "function"} else 0.0)

def excerpt(root: Path, symbol: dict[str, Any], max_lines: int = 70) -> str:
    try:
        lines = (root / symbol["file"]).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    start = max(1, int(symbol["start_line"]))
    end = min(int(symbol["end_line"]), start + max_lines - 1, len(lines))
    body = "\n".join(f"{line:>5}: {lines[line - 1]}" for line in range(start, end + 1))
    return f"### `{symbol['symbol']}` — `{symbol['file']}:{start}-{end}`\n```python\n{body}\n```"

def build_context(root: Path, question: str, limit: int = 6) -> str:
    ai_dir = root / AI_DIR
    symbols = read_json(ai_dir / INDEX_FILES["symbol"], {"symbols": []}).get("symbols", [])
    deps = read_json(ai_dir / INDEX_FILES["dependency"], {})
    tests = read_json(ai_dir / INDEX_FILES["test"], {})
    findings = read_json(ai_dir / INDEX_FILES["findings"], {"findings": []})
    tokens = query_tokens(question)
    ranked = sorted(
        ((score_symbol(symbol, tokens), symbol) for symbol in symbols),
        key=lambda row: (-row[0], row[1]["file"], row[1]["start_line"]),
    )
    selected = [symbol for score, symbol in ranked if score > 0][:limit]
    if not selected:
        selected = [symbol for _, symbol in ranked[: min(3, len(ranked))]]
    modules = sorted({symbol["module"] for symbol in selected})
    output = ["# Targeted repository context", "", f"Question: {question}", ""]
    repo_map = ai_dir / "REPO_MAP.md"
    if repo_map.exists():
        output += ["## Repository overview", repo_map.read_text(encoding="utf-8")[:5000], ""]
    for module in modules[:3]:
        path = ai_dir / "modules" / f"{module}.md"
        if path.exists():
            output += [f"## Module: {module}", path.read_text(encoding="utf-8")[:4500], ""]
    output += ["## Relevant source symbols"]
    for symbol in selected:
        output += [excerpt(root, symbol), ""]
    selected_ids = {symbol["id"] for symbol in selected}
    related = [
        edge for edge in deps.get("symbol_edges", [])
        if edge.get("source") in selected_ids or edge.get("target") in selected_ids
    ]
    if related:
        output += ["## Call relationships"]
        output += [f"- `{edge['source']}` -> `{edge['target']}` ({edge['confidence']})" for edge in related[:30]]
        output.append("")
    mapped_tests = sorted({
        test for symbol in selected
        for test in tests.get("symbol_to_tests", {}).get(symbol["id"], [])
    } | {
        test for symbol in selected
        for test in tests.get("file_to_tests", {}).get(symbol["file"], [])
    })
    if mapped_tests:
        output += ["## Relevant tests", *[f"- `{test}`" for test in mapped_tests[:20]], ""]
    relevant_findings = []
    for finding in findings.get("findings", []):
        haystack = json.dumps(finding, sort_keys=True).lower()
        if any(token in haystack for token in tokens):
            relevant_findings.append(finding)
    if relevant_findings:
        output += ["## Prior findings"]
        output += [
            f"- {row.get('id')}: {row.get('severity')} / {row.get('status')} — {row.get('description')}"
            for row in relevant_findings[:10]
        ]
        output.append("")
    decisions = ai_dir / "DECISIONS.md"
    if decisions.exists():
        output += ["## Repository decisions", decisions.read_text(encoding="utf-8")[:5000], ""]
    status = current_status(root)
    changed = status.get("modified", []) + status.get("added", []) + status.get("deleted", [])
    if changed:
        output += ["## Changes since the committed AI index", *[f"- `{path}`" for path in changed[:30]], ""]
    return "\n".join(output).rstrip() + "\n"
