from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    AI_DIR, AUDIT_CATEGORIES, INDEX_FILES, MODULE_KNOWLEDGE, SCHEMA_VERSION,
    canonical_json, git_state, language_for, module_for, read_json, sha256_file,
    should_index, source_fingerprint, tracked_files, write_if_changed,
)
from .python_index import analyze_python, build_dependency_graph, internal_module_map, resolve_import_file

def build_file_index(root: Path, previous: dict[str, Any] | None = None) -> tuple[dict[str, Any], set[str], set[str]]:
    old_files = (previous or {}).get("files", {})
    files: dict[str, Any] = {}
    state = git_state(root)
    for path in tracked_files(root):
        if not should_index(path) or not (root / path).is_file():
            continue
        digest = sha256_file(root / path)
        old = old_files.get(path, {})
        same = old.get("sha256") == digest
        files[path] = {
            "path": path, "language": language_for(path), "sha256": digest,
            "module": module_for(path),
            "symbol_count": old.get("symbol_count", 0) if same else 0,
            "parse_error": old.get("parse_error") if same else None,
            "imports": old.get("imports", []) if same else [],
            "last_indexed_commit": old.get("last_indexed_commit") if same else state["commit"],
        }
    changed = {path for path, meta in files.items() if old_files.get(path, {}).get("sha256") != meta["sha256"]}
    deleted = set(old_files) - set(files)
    return {
        "schema_version": SCHEMA_VERSION, "indexed_commit": state["commit"],
        "indexed_branch": state["branch"], "indexed_at": state["indexed_at"], "files": files,
    }, changed, deleted

def build_symbols(root: Path, file_index: dict[str, Any], previous: dict[str, Any] | None, changed: set[str], deleted: set[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    kept = [
        row for row in (previous or {}).get("symbols", [])
        if row.get("file") in file_index["files"] and row.get("file") not in changed and row.get("file") not in deleted
    ]
    created: list[dict[str, Any]] = []
    imports: dict[str, list[dict[str, Any]]] = {}
    for path, meta in sorted(file_index["files"].items()):
        if meta["language"] != "python":
            continue
        if path in changed or previous is None:
            symbols, rows, error = analyze_python(root, path)
            created.extend(symbols)
            imports[path] = rows
            meta["imports"], meta["parse_error"], meta["symbol_count"] = rows, error, len(symbols)
        else:
            imports[path] = list(meta.get("imports", []))
    symbols = kept + created
    symbols.sort(key=lambda row: (row["file"], row["start_line"], row["symbol"]))
    return {"schema_version": SCHEMA_VERSION, "symbols": symbols}, imports

def build_test_map(root: Path, file_index: dict[str, Any], symbol_index: dict[str, Any], imports: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    module_map = internal_module_map(file_index["files"])
    file_to_tests: dict[str, set[str]] = defaultdict(set)
    symbol_to_tests: dict[str, set[str]] = defaultdict(set)
    module_to_tests: dict[str, set[str]] = defaultdict(set)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for symbol in symbol_index["symbols"]:
        symbol["tests"] = []
        by_file[symbol["file"]].append(symbol)
    tests = [
        path for path, meta in file_index["files"].items()
        if path.startswith("tests/") and meta["language"] == "python"
    ]
    for test in sorted(tests):
        implementation: set[str] = set()
        for row in imports.get(test, []):
            target = resolve_import_file(row.get("module", ""), module_map)
            if target and not target.startswith("tests/"):
                implementation.add(target)
                file_to_tests[target].add(test)
                module_to_tests[module_for(target)].add(test)
        try:
            text = (root / test).read_text(encoding="utf-8")
        except OSError:
            text = ""
        for impl in implementation:
            for symbol in by_file.get(impl, []):
                name = symbol["symbol"].split(".")[-1]
                if name in text:
                    symbol_to_tests[symbol["id"]].add(test)
                    symbol["tests"].append(test)
    for symbol in symbol_index["symbols"]:
        symbol["tests"] = sorted(set(symbol["tests"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "mapping_method": "confirmed internal imports; symbol mapping additionally requires referenced symbol name in test source",
        "file_to_tests": {key: sorted(value) for key, value in sorted(file_to_tests.items())},
        "symbol_to_tests": {key: sorted(value) for key, value in sorted(symbol_to_tests.items())},
        "module_to_tests": {key: sorted(value) for key, value in sorted(module_to_tests.items())},
    }

def build_change_index(file_index: dict[str, Any], old_files: dict[str, Any] | None, old_symbols: dict[str, Any] | None, symbol_index: dict[str, Any], changed: set[str], deleted: set[str]) -> dict[str, Any]:
    previous_paths = (old_files or {}).get("files", {})
    current_ids = {row["id"] for row in symbol_index["symbols"]}
    return {
        "schema_version": SCHEMA_VERSION,
        "previous_indexed_commit": (old_files or {}).get("indexed_commit"),
        "indexed_commit": file_index["indexed_commit"],
        "added_files": sorted(path for path in changed if path not in previous_paths),
        "modified_files": sorted(path for path in changed if path in previous_paths),
        "deleted_files": sorted(deleted),
        "affected_symbols": sorted(row["id"] for row in symbol_index["symbols"] if row["file"] in changed),
        "removed_symbols": sorted(
            row["id"] for row in (old_symbols or {}).get("symbols", [])
            if row.get("file") in changed | deleted and row.get("id") not in current_ids
        ),
        "note": "Affected means reparsed because the containing file changed; semantic behavior change is not asserted.",
    }

def update_audit(file_index: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    old = (previous or {}).get("units", {})
    units: dict[str, Any] = {}
    for path, meta in sorted(file_index["files"].items()):
        if meta["language"] != "python" or path.startswith("tests/"):
            continue
        prior = old.get(path)
        if prior and prior.get("current_content_hash") == meta["sha256"]:
            units[path] = prior
            continue
        previous_review = None
        if prior:
            previous_review = {
                "content_hash": prior.get("current_content_hash"), "reviewed": bool(prior.get("reviewed")),
                "audit_categories": list(prior.get("audit_categories", [])),
                "last_reviewed_commit": prior.get("last_reviewed_commit"),
            }
        units[path] = {
            "current_content_hash": meta["sha256"], "reviewed": False,
            "audit_categories": [], "last_reviewed_commit": None, "unresolved_findings": [],
            "re_audit_required": bool(prior),
            "changed_at_index_commit": file_index["indexed_commit"] if prior else None,
            "previous_review": previous_review,
        }
    return {"schema_version": SCHEMA_VERSION, "audit_categories": AUDIT_CATEGORIES, "units": units}

def _module_summary(module: str, file_index: dict[str, Any], symbols: dict[str, Any], deps: dict[str, Any], tests: dict[str, Any]) -> str:
    files = [path for path, meta in file_index["files"].items() if meta["module"] == module and not path.startswith("tests/")]
    files = sorted(files, key=lambda path: (-file_index["files"][path].get("symbol_count", 0), path))[:12]
    important = [row for row in symbols["symbols"] if row["module"] == module and row["kind"] in {"class", "function"}]
    important = sorted(important, key=lambda row: (row["kind"] != "class", row["file"], row["start_line"]))[:18]
    dependencies = sorted({edge["target"] for edge in deps["module_edges"] if edge["source"] == module})
    dependents = sorted({edge["source"] for edge in deps["module_edges"] if edge["target"] == module})
    purpose, state, flow, constraints, failures = MODULE_KNOWLEDGE.get(module, (
        "Logical retrieval group derived from repository paths and dependencies.",
        "No additional state ownership asserted by the generator.",
        "Use the dependency and call graphs for exact relationships.",
        "Follow the nearest AGENTS.md and authored architecture decisions.",
        "No subsystem-specific failure pattern asserted by the generator.",
    ))
    lines = [
        f"# {module}", "", f"**Purpose:** {purpose}", "", f"**Important state:** {state}", "",
        f"**Major flow:** {flow}", "", f"**Constraints/invariants:** {constraints}", "",
        f"**Common failure points:** {failures}", "", "## Primary files",
    ]
    lines += [f"- `{path}`" for path in files] or ["- None indexed"]
    lines += ["", "## Important public/entry symbols"]
    lines += [f"- `{row['symbol']}` — `{row['file']}:{row['start_line']}-{row['end_line']}`" for row in important] or ["- None indexed"]
    lines += ["", "## Dependencies", "- " + (", ".join(f"`{x}`" for x in dependencies) if dependencies else "None confirmed")]
    lines += ["", "## Dependents", "- " + (", ".join(f"`{x}`" for x in dependents) if dependents else "None confirmed")]
    lines += ["", "## Associated tests"]
    lines += [f"- `{path}`" for path in tests["module_to_tests"].get(module, [])[:15]] or ["- No direct import-based mapping"]
    lines += ["", "## Retrieval note", "Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.", ""]
    return "\n".join(lines)

def update_module_summaries(ai_dir: Path, file_index: dict[str, Any], symbols: dict[str, Any], deps: dict[str, Any], tests: dict[str, Any]) -> None:
    modules = sorted({
        meta["module"] for path, meta in file_index["files"].items()
        if not path.startswith("tests/") and meta["module"] not in {"repository", "documentation"}
    })
    directory = ai_dir / "modules"
    directory.mkdir(parents=True, exist_ok=True)
    wanted = {f"{module}.md" for module in modules}
    for existing in directory.glob("*.md"):
        if existing.name not in wanted:
            existing.unlink()
    for module in modules:
        write_if_changed(directory / f"{module}.md", _module_summary(module, file_index, symbols, deps, tests))

def _status_markdown(file_index: dict[str, Any], symbols: dict[str, Any], deps: dict[str, Any], tests: dict[str, Any], audit: dict[str, Any], changed: set[str], deleted: set[str]) -> str:
    parse_errors = [path for path, meta in file_index["files"].items() if meta.get("parse_error")]
    reviewed = sum(bool(row.get("reviewed")) for row in audit["units"].values())
    re_audit = sum(bool(row.get("re_audit_required")) for row in audit["units"].values())
    return f"""# AI index status\n\n- Schema: {SCHEMA_VERSION}\n- Indexed commit: `{file_index['indexed_commit']}`\n- Indexed branch: `{file_index['indexed_branch']}`\n- Source fingerprint: `{source_fingerprint(file_index)}`\n- Indexed files: {len(file_index['files'])}\n- Indexed symbols: {len(symbols['symbols'])}\n- File dependency edges: {len(deps['file_edges'])}\n- Symbol call edges: {len(deps['symbol_edges'])}\n- Implementation files with mapped tests: {len(tests['file_to_tests'])}\n- Parse errors: {len(parse_errors)}\n- Audited units retained: {reviewed}\n- Units requiring re-audit: {re_audit}\n- Files changed in this update: {len(changed)}\n- Files removed since previous index: {len(deleted)}\n\nIndexing is not auditing. Review state is valid only when `audit-coverage.json` binds it to the current content hash.\n"""

def update_index(root: Path, initialize: bool = False) -> dict[str, Any]:
    ai_dir = root / AI_DIR
    ai_dir.mkdir(parents=True, exist_ok=True)
    old_files = None if initialize else read_json(ai_dir / INDEX_FILES["file"], None)
    old_symbols = None if initialize else read_json(ai_dir / INDEX_FILES["symbol"], None)
    old_audit = read_json(ai_dir / INDEX_FILES["audit"], None)
    file_index, changed, deleted = build_file_index(root, old_files)
    symbols, imports = build_symbols(root, file_index, old_symbols, changed, deleted)
    deps = build_dependency_graph(file_index, symbols, imports)
    tests = build_test_map(root, file_index, symbols, imports)
    changes = build_change_index(file_index, old_files, old_symbols, symbols, changed, deleted)
    audit = update_audit(file_index, old_audit)
    findings = read_json(ai_dir / INDEX_FILES["findings"], {"schema_version": SCHEMA_VERSION, "findings": []})
    if not isinstance(findings, dict) or not isinstance(findings.get("findings", []), list):
        findings = {"schema_version": SCHEMA_VERSION, "findings": []}
    findings["schema_version"] = SCHEMA_VERSION
    outputs = {
        "file": file_index, "symbol": symbols, "dependency": deps, "test": tests,
        "changes": changes, "audit": audit, "findings": findings,
    }
    for key, value in outputs.items():
        write_if_changed(ai_dir / INDEX_FILES[key], canonical_json(value))
    write_if_changed(ai_dir / INDEX_FILES["status"], _status_markdown(file_index, symbols, deps, tests, audit, changed, deleted))
    update_module_summaries(ai_dir, file_index, symbols, deps, tests)
    return {
        "changed": sorted(changed), "deleted": sorted(deleted), "files": len(file_index["files"]),
        "symbols": len(symbols["symbols"]), "fingerprint": source_fingerprint(file_index),
    }

def current_status(root: Path) -> dict[str, Any]:
    indexed = read_json(root / AI_DIR / INDEX_FILES["file"], None)
    if not indexed:
        return {"initialized": False, "stale": True, "modified": [], "added": [], "deleted": []}
    current, changed, deleted = build_file_index(root, indexed)
    old_paths = set(indexed.get("files", {}))
    added = sorted(path for path in changed if path not in old_paths)
    return {
        "initialized": True, "stale": bool(changed or deleted),
        "indexed_commit": indexed.get("indexed_commit"), "current_commit": git_state(root)["commit"],
        "modified": sorted(changed - set(added)), "added": added, "deleted": sorted(deleted),
        "source_fingerprint": source_fingerprint(current),
    }

def validate_indexes(root: Path) -> list[str]:
    ai_dir = root / AI_DIR
    file_index = read_json(ai_dir / INDEX_FILES["file"], {})
    symbols = read_json(ai_dir / INDEX_FILES["symbol"], {})
    deps = read_json(ai_dir / INDEX_FILES["dependency"], {})
    tests = read_json(ai_dir / INDEX_FILES["test"], {})
    audit = read_json(ai_dir / INDEX_FILES["audit"], {})
    files = file_index.get("files", {})
    symbol_ids = {row.get("id") for row in symbols.get("symbols", [])}
    errors: list[str] = []
    for row in symbols.get("symbols", []):
        if row.get("file") not in files:
            errors.append(f"symbol references missing file: {row.get('id')}")
    for edge in deps.get("file_edges", []):
        if edge.get("source") not in files or edge.get("target") not in files:
            errors.append(f"file edge references missing file: {edge}")
    for edge in deps.get("symbol_edges", []):
        if edge.get("source") not in symbol_ids or edge.get("target") not in symbol_ids:
            errors.append(f"symbol edge references missing symbol: {edge}")
    for impl, mapped in tests.get("file_to_tests", {}).items():
        if impl not in files:
            errors.append(f"test map implementation missing: {impl}")
        for test in mapped:
            if test not in files:
                errors.append(f"test map test missing: {test}")
    for path, unit in audit.get("units", {}).items():
        if path not in files:
            errors.append(f"audit unit missing from index: {path}")
        elif unit.get("current_content_hash") != files[path].get("sha256"):
            errors.append(f"audit hash stale: {path}")
        if unit.get("reviewed") and not unit.get("audit_categories"):
            errors.append(f"reviewed unit lacks categories: {path}")
    return errors
