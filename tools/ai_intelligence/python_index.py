from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .common import module_for

def module_name_from_path(path: str) -> str | None:
    p = PurePosixPath(path)
    if p.suffix not in {".py", ".pyi"}:
        return None
    parts = list(p.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None

def resolve_import_name(current_module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    parts = current_module.split(".")[:-1]
    trim = max(0, node.level - 1)
    if trim:
        parts = parts[:-trim] if trim <= len(parts) else []
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)

class SymbolVisitor(ast.NodeVisitor):
    def __init__(self, path: str, python_module: str) -> None:
        self.path = path
        self.python_module = python_module
        self.logical_module = module_for(path)
        self.classes: list[str] = []
        self.functions: list[str] = []
        self.symbols: list[dict[str, Any]] = []
        self.imports: list[dict[str, Any]] = []
        self.import_aliases: dict[str, str] = {}
        self.from_aliases: dict[str, str] = {}

    def _qualname(self, name: str) -> str:
        prefix = ".".join(self.classes + self.functions)
        return f"{prefix}.{name}" if prefix else name

    def _id(self, name: str) -> str:
        return f"{self.python_module}:{name}"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.import_aliases[local] = alias.name
            self.imports.append({"module": alias.name, "name": None, "alias": local, "line": node.lineno})

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = resolve_import_name(self.python_module, node)
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            target = f"{base}.{alias.name}" if base else alias.name
            self.from_aliases[local] = target
            self.imports.append({"module": base, "name": alias.name, "alias": local, "line": node.lineno})

    def _calls(self, node: ast.AST) -> list[dict[str, str]]:
        calls: dict[tuple[str, str], dict[str, str]] = {}
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            target = None
            confidence = "inferred"
            if isinstance(func, ast.Name):
                target = self.from_aliases.get(func.id, func.id)
                if func.id in self.from_aliases:
                    confidence = "confirmed"
            elif isinstance(func, ast.Attribute):
                parts = [func.attr]
                value = func.value
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    parts.append(value.id)
                    dotted = ".".join(reversed(parts))
                    if value.id in self.import_aliases:
                        target = self.import_aliases[value.id] + dotted[len(value.id):]
                        confidence = "confirmed"
                    else:
                        target = dotted
            if target:
                calls[(target, confidence)] = {"target": target, "confidence": confidence}
        return [calls[key] for key in sorted(calls)]

    def _callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = self._qualname(node.name)
        self.symbols.append({
            "id": self._id(name), "symbol": name,
            "kind": "method" if self.classes else "function",
            "file": self.path, "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "module": self.logical_module, "python_module": self.python_module,
            "async": isinstance(node, ast.AsyncFunctionDef),
            "raw_calls": self._calls(node), "calls": [], "called_by": [], "tests": [],
        })

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._callable(node)
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._callable(node)
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        name = self._qualname(node.name)
        self.symbols.append({
            "id": self._id(name), "symbol": name, "kind": "class", "file": self.path,
            "start_line": node.lineno, "end_line": getattr(node, "end_lineno", node.lineno),
            "module": self.logical_module, "python_module": self.python_module, "async": False,
            "raw_calls": [], "calls": [], "called_by": [], "tests": [],
        })
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.classes or self.functions:
            self.generic_visit(node)
            return
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                self.symbols.append({
                    "id": self._id(target.id), "symbol": target.id, "kind": "constant",
                    "file": self.path, "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "module": self.logical_module, "python_module": self.python_module, "async": False,
                    "raw_calls": [], "calls": [], "called_by": [], "tests": [],
                })
        self.generic_visit(node)

def analyze_python(root: Path, path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    module = module_name_from_path(path) or path
    try:
        tree = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:
        return [], [], f"{type(exc).__name__}: {exc}"
    visitor = SymbolVisitor(path, module)
    visitor.visit(tree)
    return visitor.symbols, visitor.imports, None

def internal_module_map(paths: Iterable[str]) -> dict[str, str]:
    return {module: path for path in paths if (module := module_name_from_path(path))}

def resolve_import_file(module: str, module_map: dict[str, str]) -> str | None:
    candidate = module
    while candidate:
        if candidate in module_map:
            return module_map[candidate]
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return None

def build_dependency_graph(file_index: dict[str, Any], symbol_index: dict[str, Any], imports: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    module_map = internal_module_map(file_index["files"])
    file_edges: dict[tuple[str, str], dict[str, str]] = {}
    module_edges: set[tuple[str, str]] = set()
    for source, rows in imports.items():
        for row in rows:
            target = resolve_import_file(row.get("module", ""), module_map)
            if target and target != source:
                file_edges[(source, target)] = {"source": source, "target": target, "relationship": "imports", "confidence": "confirmed"}
                sm, tm = file_index["files"][source]["module"], file_index["files"][target]["module"]
                if sm != tm:
                    module_edges.add((sm, tm))

    symbols = symbol_index["symbols"]
    by_id = {row["id"]: row for row in symbols}
    by_python: dict[str, list[str]] = defaultdict(list)
    by_simple: dict[str, list[str]] = defaultdict(list)
    for row in symbols:
        by_python[f"{row['python_module']}.{row['symbol']}"].append(row["id"])
        by_simple[row["symbol"].split(".")[-1]].append(row["id"])
        row["calls"] = []
        row["called_by"] = []

    symbol_edges: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in symbols:
        resolved: dict[tuple[str, str], dict[str, str]] = {}
        for call in row.get("raw_calls", []):
            raw, confidence = call["target"], call["confidence"]
            target_id = None
            exact = by_python.get(raw, [])
            if len(exact) == 1:
                target_id, confidence = exact[0], "confirmed"
            elif "." not in raw:
                local = f"{row['python_module']}:{raw}"
                if local in by_id:
                    target_id, confidence = local, "confirmed"
                elif len(by_simple.get(raw, [])) == 1:
                    target_id, confidence = by_simple[raw][0], "inferred"
            else:
                simple = raw.rsplit(".", 1)[-1]
                if len(by_simple.get(simple, [])) == 1:
                    target_id = by_simple[simple][0]
            if target_id and target_id != row["id"]:
                resolved[(target_id, confidence)] = {"symbol": target_id, "confidence": confidence}
                symbol_edges[(row["id"], target_id, confidence)] = {
                    "source": row["id"], "target": target_id,
                    "relationship": "calls", "confidence": confidence,
                }
        row["calls"] = [resolved[key] for key in sorted(resolved)]
    for edge in symbol_edges.values():
        by_id[edge["target"]]["called_by"].append({"symbol": edge["source"], "confidence": edge["confidence"]})
    for row in symbols:
        row["called_by"].sort(key=lambda item: (item["symbol"], item["confidence"]))

    return {
        "schema_version": 1,
        "file_edges": [file_edges[key] for key in sorted(file_edges)],
        "module_edges": [
            {"source": source, "target": target, "relationship": "depends_on", "confidence": "confirmed"}
            for source, target in sorted(module_edges)
        ],
        "symbol_edges": [symbol_edges[key] for key in sorted(symbol_edges)],
    }
