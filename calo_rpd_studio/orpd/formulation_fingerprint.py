"""Canonical scientific formulation fingerprints for transfer/resume/parity provenance."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import functools
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


def canonical_scientific_value(value):
    if is_dataclass(value):
        return canonical_scientific_value(asdict(value))
    if isinstance(value, Enum):
        return canonical_scientific_value(value.value)
    if isinstance(value, np.ndarray):
        return canonical_scientific_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(k): canonical_scientific_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_scientific_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical_scientific_value(v) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if isinstance(value, functools.partial):
        return {
            "callable_kind": "functools.partial",
            "func": canonical_scientific_value(value.func),
            "args": canonical_scientific_value(value.args),
            "keywords": canonical_scientific_value(value.keywords or {}),
        }
    if inspect.ismethod(value):
        owner = getattr(value, "__self__", None)
        owner_state = {}
        if owner is not None and not inspect.isclass(owner):
            try:
                owner_state = canonical_scientific_value(vars(owner))
            except TypeError:
                owner_state = {"type": f"{type(owner).__module__}.{type(owner).__qualname__}"}
        return {
            "callable_kind": "bound_method",
            "function": canonical_scientific_value(value.__func__),
            "owner_type": f"{type(owner).__module__}.{type(owner).__qualname__}" if owner is not None else "",
            "owner_state": owner_state,
        }
    if callable(value):
        defaults = getattr(value, "__defaults__", None)
        kwdefaults = getattr(value, "__kwdefaults__", None)
        closure = getattr(value, "__closure__", None) or ()
        closure_values = []
        for cell in closure:
            try:
                closure_values.append(canonical_scientific_value(cell.cell_contents))
            except (ValueError, TypeError):
                closure_values.append({"unreadable_closure_cell": True})
        code = getattr(value, "__code__", None)
        if code is None and not inspect.isbuiltin(value):
            call_impl = getattr(type(value), "__call__", None)
            try:
                state = canonical_scientific_value(vars(value))
            except TypeError:
                state = {}
            if call_impl is None and not state:
                raise ValueError(f"Cannot safely canonicalize callable scientific transform {type(value)!r}")
            return {
                "callable_kind": "callable_object",
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "call_impl": canonical_scientific_value(call_impl) if call_impl is not None else None,
                "state": state,
            }
        code_identity = ""
        if code is not None:
            payload = {
                "co_code": code.co_code.hex(),
                "co_consts": canonical_scientific_value(code.co_consts),
                "co_names": list(code.co_names),
                "co_varnames": list(code.co_varnames),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            code_identity = hashlib.sha256(encoded).hexdigest()
        return {
            "callable_kind": "function" if code is not None else "builtin",
            "callable_module": str(getattr(value, "__module__", "")),
            "callable_qualname": str(getattr(value, "__qualname__", type(value).__qualname__)),
            "defaults": canonical_scientific_value(defaults or ()),
            "kwdefaults": canonical_scientific_value(kwdefaults or {}),
            "closure": closure_values,
            "code_identity_sha256": code_identity,
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "state": canonical_scientific_value(vars(value)),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def scientific_problem_payload(problem) -> dict:
    decoder = getattr(problem, "decoder", None)
    manifest = None
    if decoder is not None and callable(getattr(decoder, "formulation_manifest", None)):
        manifest = decoder.formulation_manifest()
    scenarios = []
    for scenario in list(getattr(problem, "scenarios", []) or []):
        scenarios.append(
            {
                "name": str(getattr(scenario, "name", "")),
                "weight": float(getattr(scenario, "weight", 1.0)),
                "transform": canonical_scientific_value(getattr(scenario, "transform", None)),
            }
        )
    case = getattr(problem, "case", None)
    checksum_fn = getattr(case, "checksum", None)
    case_checksum = str(checksum_fn()) if callable(checksum_fn) else ""
    generic_identity = None
    if not case_checksum:
        generic_identity = {
            "problem_class": f"{problem.__class__.__module__}.{problem.__class__.__qualname__}",
            "lower_bounds": canonical_scientific_value(getattr(problem, "lower_bounds", None)),
            "upper_bounds": canonical_scientific_value(getattr(problem, "upper_bounds", None)),
        }
    return canonical_scientific_value(
        {
            "schema_version": "calo-rpd-formulation-v5.9",
            "case_checksum": case_checksum,
            "generic_problem_identity": generic_identity,
            "dimension": int(problem.dimension),
            "formulation_manifest": manifest,
            "problem_config": getattr(problem, "config", None),
            "scenarios": scenarios,
            "repair_schema": "normalized-clip-once-v5.9",
        }
    )


def scientific_problem_fingerprint(problem) -> str:
    encoded = json.dumps(
        scientific_problem_payload(problem),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
