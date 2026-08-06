"""Inventory and classify broad exception handlers without importing project modules."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("calo_rpd_studio", "calo_bootstrap", "containers")
SCIENTIFIC_PRIORITY_PREFIXES = (
    "calo_rpd_studio/statistics/",
    "calo_rpd_studio/algorithms/calo/policy_qualification.py",
)


def _exception_name(node: ast.expr | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
        current = parents.get(current)
    return "<module>"


def _classify(path: str, function: str) -> tuple[str, str]:
    cleanup_words = ("close", "exit", "release", "unlock", "shutdown", "cleanup", "cancel")
    if any(word in function.lower() for word in cleanup_words):
        return "cleanup", "best-effort cleanup or resource release"

    boundary_prefixes = (
        "calo_bootstrap/",
        "containers/",
        "calo_rpd_studio/gui/",
        "calo_rpd_studio/app/",
        "calo_rpd_studio/scripts/",
    )
    boundary_fragments = (
        "independent_validator.py",
        "publication_export.py",
        "portfolio/exporter.py",
        "experiments/provenance.py",
        "experiments/fairness_validator.py",
        "experiments/parallel_runner.py",
        "experiments/experiment_runner.py",
        "benchmarking/validation.py",
        "ai/model_io.py",
    )
    if path.startswith(boundary_prefixes) or any(item in path for item in boundary_fragments):
        return "boundary", "GUI, process, third-party, persistence, or external interface boundary"

    recovery_fragments = (
        "/compute/",
        "/accelerated/",
        "/results/database.py",
        "/algorithms/calo/training",
        "/algorithms/calo/competitive_training.py",
        "/algorithms/calo/heterogeneous_training.py",
        "/algorithms/calo/device_resident_synthetic.py",
        "/algorithms/calo/tsh_calo_",
        "/algorithms/calo/policy_registry.py",
        "/algorithms/calo/policy_readiness.py",
    )
    if any(item in path for item in recovery_fragments):
        return "recovery", "retained failure, worker recovery, or governed fallback path"

    return "defect_masking", "requires focused narrowing or an explicit boundary justification"


def audit_broad_exceptions(root: Path = PROJECT_ROOT) -> dict:
    entries = []
    parse_failures = []
    for source_root in SOURCE_ROOTS:
        for path in sorted((root / source_root).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (OSError, SyntaxError, UnicodeError) as exc:
                parse_failures.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})
                continue
            parents = {
                child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                exception = _exception_name(node.type)
                if exception not in {"Exception", "BaseException"}:
                    continue
                function = _enclosing_function(node, parents)
                category, rationale = _classify(relative, function)
                entries.append(
                    {
                        "path": relative,
                        "line": int(node.lineno),
                        "function": function,
                        "exception": exception,
                        "category": category,
                        "rationale": rationale,
                    }
                )
    entries.sort(key=lambda item: (item["path"], item["line"]))
    counts = Counter(str(item["category"]) for item in entries)
    scientific_priority = [
        item for item in entries if str(item["path"]).startswith(SCIENTIFIC_PRIORITY_PREFIXES)
    ]
    passed = not parse_failures and not scientific_priority
    return {
        "schema_version": "calo-broad-exception-classification-v1",
        "passed": passed,
        "classification_counts": dict(sorted(counts.items())),
        "scientific_priority_handlers": scientific_priority,
        "parse_failures": parse_failures,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_broad_exceptions()
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
