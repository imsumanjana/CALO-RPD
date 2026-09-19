"""Inventory GUI signal connections; static findings never replace runtime wiring tests."""

from __future__ import annotations
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(root: Path) -> dict:
    connections, duplicates, late_bindings, hashes = [], [], [], {}
    for directory in ("calo_rpd_studio/app", "calo_rpd_studio/gui"):
        for path in sorted((root / directory).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            hashes[relative] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            tree = ast.parse(text, filename=relative)
            parents = {
                child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
            }
            groups = defaultdict(list)
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect"
                    and node.args
                ):
                    continue
                ancestors, cursor = [], parents.get(node)
                while cursor is not None:
                    ancestors.append(cursor)
                    cursor = parents.get(cursor)
                method = next(
                    (item.name for item in ancestors if isinstance(item, ast.FunctionDef)),
                    "<module>",
                )
                owner = next(
                    (item.name for item in ancestors if isinstance(item, ast.ClassDef)), ""
                )
                row = {
                    "file": relative,
                    "line": node.lineno,
                    "class": owner,
                    "method": method,
                    "signal": ast.unparse(node.func.value),
                    "slot": ast.unparse(node.args[0]),
                }
                connections.append(row)
                groups[(owner, method, row["signal"], row["slot"])].append(row)
                if isinstance(node.args[0], ast.Lambda):
                    function = node.args[0]
                    parameters = {
                        arg.arg
                        for arg in [
                            *function.args.posonlyargs,
                            *function.args.args,
                            *function.args.kwonlyargs,
                        ]
                    }
                    used = {
                        item.id for item in ast.walk(function.body) if isinstance(item, ast.Name)
                    }
                    for loop in (item for item in ancestors if isinstance(item, ast.For)):
                        variables = {
                            item.id for item in ast.walk(loop.target) if isinstance(item, ast.Name)
                        }
                        captured = (variables & used) - parameters
                        if captured:
                            late_bindings.append(
                                {**row, "unbound_loop_variables": sorted(captured)}
                            )
            duplicates.extend(rows for rows in groups.values() if len(rows) > 1)
    return {
        "schema": "calo-gui-wiring-inventory-v1",
        "source_hashes": hashes,
        "connections": connections,
        "possible_duplicates": duplicates,
        "possible_late_bindings": late_bindings,
        "dynamic_validation_required": True,
        "limitations": "Does not prove semantic destinations, indirect connections or thread delivery. Run GUI acceptance tests.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "connections": len(report["connections"]),
                "possible_duplicates": len(report["possible_duplicates"]),
                "possible_late_bindings": len(report["possible_late_bindings"]),
                "dynamic_validation_required": True,
            }
        )
    )
    return int(bool(report["possible_duplicates"] or report["possible_late_bindings"]))


if __name__ == "__main__":
    raise SystemExit(main())
