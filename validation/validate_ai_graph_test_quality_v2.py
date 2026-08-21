#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TEST_TREE_PREFIXES = ("tests/", "test/", "spec/", "specs/", "__tests__/")
TEST_CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".rb", ".php", ".java", ".kt", ".kts", ".cs",
    ".c", ".cc", ".cpp", ".cxx",
}
GENERIC_TEST_NAMES = {
    "get", "set", "run", "pass", "passed", "status", "add", "remove",
    "update", "read", "write", "open", "close", "main",
}
CLASS_KINDS = {"class", "interface", "struct", "enum", "trait"}
REPOSITORY_PREFIXES = ("calo_rpd_studio.", "calo_bootstrap.", "tests.", "tools.")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def is_test_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    name = Path(normalized).name
    suffix = Path(name).suffix
    if suffix not in TEST_CODE_SUFFIXES:
        return False
    if name in {"__init__.py", "conftest.py"}:
        return False
    if name.startswith("test_"):
        return True
    if name.endswith(("_test.py", "_test.go", "_test.rs", "_test.rb", "_test.php")):
        return True
    return any(marker in name for marker in (".test.", ".spec."))


def is_test_support_path(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    return normalized.startswith(TEST_TREE_PREFIXES) or is_test_path(path)


def edge_target(edge: dict[str, Any]) -> str:
    return str(edge.get("target") or edge.get("to") or "")


def edge_confidence(edge: dict[str, Any]) -> str:
    return str(edge.get("confidence") or "unknown")


def receiver_root(raw: str) -> str:
    return raw.split(".", 1)[0] if "." in raw else ""


def receiver_is_architecturally_proven(symbol: dict[str, Any], raw: str) -> bool:
    root = receiver_root(raw)
    if not root:
        return True
    if root in {"self", "cls", "super"}:
        return True
    bindings = symbol.get("import_bindings", {}) or {}
    if root in bindings:
        return True
    # Direct class/type receivers are distinguishable from ordinary local variables.
    if root[:1].isupper():
        return True
    return False


def attribute_parts(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return parts


def python_exact_symbol_imports(root: Path, test_file: str) -> set[str]:
    """Return exact repository symbol paths structurally imported/referenced by a Python test.

    Direct ``from package.module import NAME`` is strong symbol-level evidence even
    when NAME is a constant, callable passed as a value, or otherwise never appears
    as a call edge. Module imports are also resolved when the test references an
    attribute through the imported alias.
    """
    path = root / test_file
    if path.suffix.lower() != ".py" or not path.is_file():
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=test_file)
    except SyntaxError:
        return set()

    exact: set[str] = set()
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                qualified = f"{node.module}.{alias.name}"
                exact.add(qualified)
                local = alias.asname or alias.name
                # Imported names can themselves be modules/classes used as receivers.
                module_aliases[local] = qualified
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    module_aliases[alias.asname] = alias.name
                else:
                    # ``import a.b`` binds ``a`` locally; retain that root so an AST
                    # attribute chain ``a.b.Symbol`` reconstructs the exact path.
                    root_name = alias.name.split(".", 1)[0]
                    module_aliases[root_name] = root_name

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parts = attribute_parts(node)
        if not parts or parts[0] not in module_aliases:
            continue
        base = module_aliases[parts[0]]
        qualified = ".".join([base, *parts[1:]])
        exact.add(qualified)
    return exact


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index_root = root / ".ai" / "index"
    manifest_path = index_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing .ai/index/manifest.json; run the v2 migration first.")

    run(root, sys.executable, "scripts/ai-index", "check")

    manifest = read_json(manifest_path)
    files: dict[str, Any] = dict(manifest.get("files", {}))
    if not files:
        raise SystemExit("Manifest contains no indexed files.")

    symbols_by_q: dict[str, dict[str, Any]] = {}
    symbols_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_docs: dict[str, dict[str, Any]] = {}
    module_docs: list[dict[str, Any]] = []

    for path, meta in sorted(files.items()):
        key = meta.get("key")
        if not key:
            raise AssertionError(f"Manifest entry lacks shard key: {path}")
        symbol_path = index_root / "symbols" / f"{key}.json"
        test_path = index_root / "tests" / f"{key}.json"
        if symbol_path.is_file():
            doc = read_json(symbol_path)
            for symbol in doc.get("symbols", []) or []:
                q = str(symbol.get("qualname") or "")
                if not q:
                    continue
                symbol = dict(symbol)
                symbol.setdefault("file", path)
                symbols_by_q[q] = symbol
                symbols_by_file[path].append(symbol)
        if test_path.is_file():
            test_docs[path] = read_json(test_path)

    module_dir = index_root / "modules"
    if module_dir.is_dir():
        for shard in sorted(module_dir.glob("*.json")):
            module_docs.append(read_json(shard))

    issues: dict[str, list[dict[str, Any] | str]] = {
        "resolved_unproven_receiver_edges": [],
        "dangling_resolved_repository_targets": [],
        "bad_mapped_test_paths": [],
        "bad_test_to_source_targets": [],
        "unknown_symbol_test_keys": [],
        "unexplained_symbol_test_mappings": [],
    }
    confidence_counts: Counter[str] = Counter()
    unresolved_receiver_samples: list[dict[str, Any]] = []
    known_problem_edges: list[dict[str, Any]] = []
    resolved_repository_edges = 0

    # Full-shard call-graph review. A lowercase local receiver may be unresolved, but
    # it must not be promoted to an arbitrary repository method merely because the
    # method leaf name happens to be globally unique.
    for source_path, symbols in symbols_by_file.items():
        for symbol in symbols:
            caller = str(symbol.get("qualname") or "")
            for edge in symbol.get("call_edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                raw = str(edge.get("raw") or "")
                target = edge_target(edge)
                confidence = edge_confidence(edge)
                confidence_counts[confidence] += 1
                resolved = bool(target) and confidence != "unresolved"
                target_is_repo_symbol = target in symbols_by_q
                target_looks_repo = target.startswith(REPOSITORY_PREFIXES)

                record = {
                    "file": source_path,
                    "caller": caller,
                    "raw": raw,
                    "target": target,
                    "confidence": confidence,
                }
                if source_path == "calo_rpd_studio/experiments/experiment_runner.py" and raw.startswith("problem."):
                    known_problem_edges.append(record)

                if resolved and target_looks_repo:
                    resolved_repository_edges += 1
                    if not target_is_repo_symbol:
                        issues["dangling_resolved_repository_targets"].append(record)

                unproven_receiver = "." in raw and not receiver_is_architecturally_proven(symbol, raw)
                if unproven_receiver:
                    if resolved and target_is_repo_symbol:
                        issues["resolved_unproven_receiver_edges"].append(record)
                    elif not resolved and len(unresolved_receiver_samples) < 25:
                        unresolved_receiver_samples.append(record)

    # The synthetic regression established this exact failure class. On the real
    # repository, any problem.* receiver in experiment_runner must remain unresolved
    # unless the shard itself has architectural receiver proof (which this v2 schema
    # does not encode for ordinary local variables).
    for record in known_problem_edges:
        if record["target"] in symbols_by_q and record["confidence"] != "unresolved":
            if record not in issues["resolved_unproven_receiver_edges"]:
                issues["resolved_unproven_receiver_edges"].append(record)

    # Precompute direct resolved symbol-call and exact structural-import evidence by test file.
    direct_test_calls: dict[str, set[str]] = defaultdict(set)
    direct_test_imports: dict[str, set[str]] = {}
    for test_file, symbols in symbols_by_file.items():
        if not is_test_path(test_file):
            continue
        direct_test_imports[test_file] = python_exact_symbol_imports(root, test_file)
        for symbol in symbols:
            for edge in symbol.get("call_edges", []) or []:
                if not isinstance(edge, dict):
                    continue
                target = edge_target(edge)
                if target and edge_confidence(edge) != "unresolved":
                    direct_test_calls[test_file].add(target)

    mapped_test_paths: set[str] = set()
    file_to_tests_global: dict[str, set[str]] = defaultdict(set)
    symbol_mapping_count = 0
    evidence_counts: Counter[str] = Counter()

    for shard_owner, doc in test_docs.items():
        file_to_tests = doc.get("file_to_tests", []) or []
        if isinstance(file_to_tests, dict):
            # Defensive compatibility with aggregate-like shard variants.
            candidates = []
            for source, mapped in file_to_tests.items():
                for test in mapped or []:
                    file_to_tests_global[str(source)].add(str(test))
                    candidates.append(str(test))
            file_to_tests = candidates
        else:
            for test in file_to_tests:
                file_to_tests_global[shard_owner].add(str(test))
        for test in file_to_tests:
            mapped_test_paths.add(str(test))

        test_to_files = doc.get("test_to_files", []) or []
        if isinstance(test_to_files, dict):
            pairs = [(str(test), str(source)) for test, sources in test_to_files.items() for source in (sources or [])]
        else:
            pairs = [(shard_owner, str(source)) for source in test_to_files]
        for test, source in pairs:
            mapped_test_paths.add(test)
            if is_test_support_path(source):
                issues["bad_test_to_source_targets"].append({"test": test, "source": source})

        symbol_to_tests = doc.get("symbol_to_tests", {}) or {}
        for qualname, mapped in symbol_to_tests.items():
            q = str(qualname)
            if q not in symbols_by_q:
                issues["unknown_symbol_test_keys"].append(q)
                continue
            target_symbol = symbols_by_q[q]
            target_file = str(target_symbol.get("file") or "")
            leaf = q.rsplit(".", 1)[-1]
            kind = str(target_symbol.get("kind") or "")
            for test in mapped or []:
                test = str(test)
                symbol_mapping_count += 1
                mapped_test_paths.add(test)
                direct_call = q in direct_test_calls.get(test, set())
                exact_import = q in direct_test_imports.get(test, set())
                structural_class = (
                    leaf not in GENERIC_TEST_NAMES
                    and kind in CLASS_KINDS
                    and test in file_to_tests_global.get(target_file, set())
                )
                if direct_call:
                    evidence_counts["direct_resolved_call"] += 1
                elif exact_import:
                    evidence_counts["exact_import_or_module_reference"] += 1
                elif structural_class:
                    evidence_counts["structural_class_file_mapping"] += 1
                else:
                    issues["unexplained_symbol_test_mappings"].append(
                        {
                            "symbol": q,
                            "kind": kind,
                            "test": test,
                            "generic_leaf": leaf in GENERIC_TEST_NAMES,
                            "direct_resolved_call": False,
                            "exact_import_or_module_reference": False,
                            "structural_class_mapping": False,
                        }
                    )

    for module in module_docs:
        for test in module.get("tests", []) or []:
            mapped_test_paths.add(str(test))

    for test in sorted(mapped_test_paths):
        if test not in files:
            issues["bad_mapped_test_paths"].append({"test": test, "reason": "not in manifest"})
        elif not is_test_path(test):
            issues["bad_mapped_test_paths"].append({"test": test, "reason": "not executable test source"})

    # file_to_tests sources themselves must be implementation sources, never support/test files.
    for source, mapped in file_to_tests_global.items():
        if mapped and is_test_support_path(source):
            issues["bad_test_to_source_targets"].append(
                {"test": sorted(mapped)[0], "source": source, "reason": "support file used as implementation source"}
            )

    nonempty = {name: values for name, values in issues.items() if values}
    report = {
        "schema": "calo-graph-test-quality-v2.1",
        "passed": not nonempty,
        "manifest_files": len(files),
        "indexed_symbols_loaded": len(symbols_by_q),
        "call_confidence_counts": dict(sorted(confidence_counts.items())),
        "resolved_repository_call_edges": resolved_repository_edges,
        "unresolved_unproven_receiver_samples": unresolved_receiver_samples,
        "experiment_runner_problem_receiver_edges": known_problem_edges,
        "mapped_test_paths": len(mapped_test_paths),
        "symbol_test_mapping_pairs": symbol_mapping_count,
        "symbol_test_evidence_counts": dict(sorted(evidence_counts.items())),
        "issues": issues,
        "notes": {
            "call_policy": "Arbitrary lowercase local receiver calls must remain unresolved; imported aliases, self/cls/super, and explicit class/type receivers are allowed proof surfaces.",
            "test_policy": "Mapped tests must be executable test sources. Symbol mappings require a direct resolved call, an exact import/module-reference, or non-generic class/type structural file evidence.",
            "validator_change": "v2.1 recognizes exact structural import/reference evidence for constants and other non-call symbols; index data is not modified.",
            "scientific_runtime_executed": False,
        },
    }
    out_dir = root / ".ai-tmp" / "graph-test-quality-v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("\nGraph/test quality audit")
    print(f"Manifest files: {len(files)}")
    print(f"Symbols inspected: {len(symbols_by_q)}")
    print(f"Call confidence counts: {dict(sorted(confidence_counts.items()))}")
    print(f"Resolved repository call edges: {resolved_repository_edges}")
    print(f"Mapped executable test paths inspected: {len(mapped_test_paths)}")
    print(f"Symbol-to-test mapping pairs inspected: {symbol_mapping_count}")
    print(f"Symbol/test evidence counts: {dict(sorted(evidence_counts.items()))}")
    print(f"Unresolved arbitrary receiver samples retained: {len(unresolved_receiver_samples)}")
    print(f"Report: {report_path}")

    if nonempty:
        print("\nGRAPH/TEST QUALITY FAILED", file=sys.stderr)
        for name, values in nonempty.items():
            print(f"- {name}: {len(values)}", file=sys.stderr)
            for value in values[:10]:
                print(f"  {value}", file=sys.stderr)
        return 1

    print("GRAPH/TEST QUALITY PASSED")
    print("No false resolved arbitrary-receiver calls or unsupported test mappings were detected.")
    print("No scientific training, experiment, GUI, qualification, or protected-case workload was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
