#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

LEGACY = (
    "file-index.json",
    "symbol-index.json",
    "dependency-graph.json",
    "test-map.json",
    "audit-coverage.json",
    "change-index.json",
)

NEW_LOAD_GRAPH = r'''def load_graph(root: Path) -> dict[str, Any]:
    man = load_manifest(root)
    calls: list[dict[str, Any]] = []
    file_imports: dict[str, Any] = {}
    for path, meta in man.get("files", {}).items():
        doc = read_json(root / INDEX_ROOT / "dependencies" / f"{meta['key']}.json", {})
        if doc.get("file_imports"):
            file_imports[path] = doc["file_imports"]
        calls.extend(doc.get("symbol_calls", []))
    module_dependencies: dict[str, list[str]] = {}
    module_dependents: dict[str, list[str]] = {}
    module_dir = root / INDEX_ROOT / "modules"
    if module_dir.exists():
        for shard in sorted(module_dir.glob("*.json")):
            doc = read_json(shard, {})
            module = doc.get("module", "")
            if module:
                module_dependencies[module] = list(doc.get("depends_on", []))
                module_dependents[module] = list(doc.get("depended_on_by", []))
    return {
        "file_imports": file_imports,
        "symbol_calls": calls,
        "module_dependencies": module_dependencies,
        "module_dependents": module_dependents,
    }
'''

NEW_LOAD_TESTS = r'''def load_tests(root: Path) -> dict[str, Any]:
    man = load_manifest(root)
    f2t: dict[str, Any] = {}
    t2f: dict[str, Any] = {}
    s2t: dict[str, Any] = {}
    for path, meta in man.get("files", {}).items():
        doc = read_json(root / INDEX_ROOT / "tests" / f"{meta['key']}.json", {})
        if doc.get("file_to_tests"):
            f2t[path] = doc["file_to_tests"]
        if doc.get("test_to_files"):
            t2f[path] = doc["test_to_files"]
        s2t.update(doc.get("symbol_to_tests", {}))
    module_to_tests: dict[str, list[str]] = {}
    module_dir = root / INDEX_ROOT / "modules"
    if module_dir.exists():
        for shard in sorted(module_dir.glob("*.json")):
            doc = read_json(shard, {})
            module = doc.get("module", "")
            if module:
                module_to_tests[module] = list(doc.get("tests", []))
    return {
        "file_to_tests": f2t,
        "test_to_files": t2f,
        "symbol_to_tests": s2t,
        "module_to_tests": module_to_tests,
    }
'''

NEW_CLEANUP = r'''def remove_legacy_generated_indexes(root: Path) -> None:
    """Remove v1 root-level generated indexes after canonical v2 shards exist."""
    for name in (
        "file-index.json",
        "symbol-index.json",
        "dependency-graph.json",
        "test-map.json",
        "audit-coverage.json",
        "change-index.json",
    ):
        try:
            (root / AI / name).unlink(missing_ok=True)
        except OSError:
            pass
'''


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    proc = subprocess.run(args, cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S | re.M)
    if count != 1:
        raise SystemExit(f"Unable to patch {label}: expected one match, found {count}")
    return updated


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    engine = root / "scripts" / "ai-index"
    tests = root / "tests" / "tooling" / "test_ai_repo_intelligence_v2.py"
    if not engine.is_file() or not tests.is_file():
        raise SystemExit("Installed v2 engine/tests not found. Run the migration first.")

    original = engine.read_text(encoding="utf-8")
    if "CALO_PROFILE_VERSION" not in original or 'INDEX_ROOT = AI / "index"' not in original:
        raise SystemExit("scripts/ai-index is not the expected CALO v2 engine; refusing to patch.")

    text = original
    if "module_dir = root / INDEX_ROOT / \"modules\"" not in text.split("def load_graph", 1)[1].split("def tokenize", 1)[0]:
        text = replace_one(text, r"def load_graph\(root: Path\).*?\n\n(?=def load_tests)", NEW_LOAD_GRAPH + "\n", "load_graph")
        text = replace_one(text, r"def load_tests\(root: Path\).*?\n\n(?=def tokenize)", NEW_LOAD_TESTS + "\n", "load_tests")

    if "def remove_legacy_generated_indexes" not in text:
        text = replace_one(text, r"def write_compat\(root: Path,.*?\n\n(?=def ensure_seed_docs)", NEW_CLEANUP + "\n", "legacy compatibility writer")
        text, replacements = re.subn(
            r"write_compat\(root, meta, snapshot_previous, idx, symbols, dg, tm, au\)",
            "remove_legacy_generated_indexes(root)",
            text,
        )
        if replacements < 1:
            raise SystemExit("No write_compat call was replaced; refusing partial patch.")
    engine.write_text(text, encoding="utf-8", newline="\n")

    original_test_text = tests.read_text(encoding="utf-8")
    test_text = original_test_text
    legacy_assert = '''\n    legacy = ("file-index.json", "symbol-index.json", "dependency-graph.json", "test-map.json", "audit-coverage.json", "change-index.json")\n    assert not [name for name in legacy if (repo / ".ai" / name).exists()]\n'''
    if "assert not [name for name in legacy" not in test_text:
        anchor = '    assert change["initialization"] is True\n'
        if anchor not in test_text:
            raise SystemExit("Unable to add legacy-index regression assertion.")
        test_text = test_text.replace(anchor, anchor + legacy_assert, 1)

    old_line = '    assert not shard_path(repo, "files", new).with_name(shard_path(repo, "files", new).name.replace("collector_renamed", "collector")).exists() or True\n'
    if old_line in test_text:
        before = '    old = "calo_rpd_studio/app/collector.py"; new = "calo_rpd_studio/app/collector_renamed.py"\n'
        if before not in test_text:
            raise SystemExit("Unable to strengthen rename-shard cleanup regression.")
        test_text = test_text.replace(before, before + '    old_shard = shard_path(repo, "files", old)\n', 1)
        test_text = test_text.replace(old_line, '    assert not old_shard.exists()\n', 1)
    tests.write_text(test_text, encoding="utf-8", newline="\n")

    run(root, sys.executable, "scripts/ai-index", "update")
    run(root, sys.executable, "scripts/ai-index", "check")

    present = [name for name in LEGACY if (root / ".ai" / name).exists()]
    if present:
        raise SystemExit("Legacy generated indexes remain after v2 update: " + ", ".join(present))

    deps = run(root, sys.executable, "scripts/ai-index", "query", "get_dependencies", "calo-policy")
    tests_query = run(root, sys.executable, "scripts/ai-index", "query", "get_tests", "calo-policy")
    try:
        dep_value = json.loads(deps.stdout)
        test_value = json.loads(tests_query.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Sharded module query did not return JSON: {exc}")
    if not isinstance(dep_value, list) or not isinstance(test_value, list):
        raise SystemExit("Sharded module dependency/test query returned the wrong shape.")

    report = {
        "patch": "v2-legacy-root-index-removal",
        "legacy_indexes_present": present,
        "module_dependency_query_count": len(dep_value),
        "module_test_query_count": len(test_value),
        "engine_patched": text != original,
        "regression_test_patched": test_text != original_test_text,
    }
    out = root / ".ai-tmp" / "legacy-index-remediation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("\nLegacy-index remediation completed.")
    print("Legacy .ai generated indexes present: none")
    print(f"calo-policy dependency count: {len(dep_value)}")
    print(f"calo-policy mapped test count: {len(test_value)}")
    print(f"Report: {out}")
    print("No scientific workload or pytest suite was run by this remediation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
