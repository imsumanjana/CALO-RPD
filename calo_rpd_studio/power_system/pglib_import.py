"""Fail-closed, non-executing import for pinned official PGLib-OPF assets.

Importing an AC-OPF case does not declare an ORPD control formulation.  That
separate, reviewed boundary lives in :mod:`calo_rpd_studio.orpd.external_profile`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import resources
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal

import numpy as np

from .case_model import PowerSystemCase
from .case_validation import validate_case


PGLIB_IMPORT_SCHEMA = "calo-rpd-pglib-import-v1"
PGLIB_OFFICIAL_REPOSITORY = "https://github.com/power-grid-lib/pglib-opf"
PGLIB_DATA_LICENSE = "CC-BY-4.0"
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE = re.compile(r"v\d{2}\.\d{2}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_NUMBER = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?\Z")
_REQUIRED_FIELDS = frozenset({"version", "baseMVA", "bus", "gen", "branch", "gencost"})
_OPTIONAL_FIELDS = frozenset({"areas"})
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
_MANIFEST_FIELDS = frozenset(
    {
        "release_tag",
        "source_commit",
        "relative_path",
        "variant",
        "asset_sha256",
        "case_role",
        "attribution",
        "repository",
        "data_license",
        "schema_version",
    }
)
BUNDLED_PGLIB_CASES = {
    "pglib-case14-typical": {
        "manifest": "manifests/case14-typical.json",
        "manifest_sha256": "d387305bd9992c0feb4346d40d2d83feb4e5563076df3d478465cc32e3768a7a",
        "asset": "pglib_opf_case14_ieee.m",
    },
    "pglib-case14-api": {
        "manifest": "manifests/case14-api.json",
        "manifest_sha256": "62c4f29ee664f55aea057b8d649c4f371279686cb39b3cc859b1270b7d46e475",
        "asset": "api/pglib_opf_case14_ieee__api.m",
    },
    "pglib-case14-sad": {
        "manifest": "manifests/case14-sad.json",
        "manifest_sha256": "40a74ed69f73f241b3e0eaaa77293b34d12fec6cf2ffd161ddb9b6448c76ca86",
        "asset": "sad/pglib_opf_case14_ieee__sad.m",
    },
}


class PGLibImportError(ValueError):
    """The pinned asset or its restricted MATPOWER syntax was rejected."""


@dataclass(frozen=True, slots=True)
class PGLibSourceManifest:
    """Immutable identity required before a local PGLib asset can be parsed."""

    release_tag: str
    source_commit: str
    relative_path: str
    variant: Literal["typical", "api", "sad"]
    asset_sha256: str
    case_role: Literal["development", "validation", "protected_test"]
    attribution: str
    repository: str = PGLIB_OFFICIAL_REPOSITORY
    data_license: str = PGLIB_DATA_LICENSE
    schema_version: str = PGLIB_IMPORT_SCHEMA

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if not isinstance(value, str):
                raise PGLibImportError(f"PGLib manifest field {field} must be a string")
        if self.schema_version != PGLIB_IMPORT_SCHEMA:
            raise PGLibImportError("Unsupported PGLib import manifest schema")
        if self.repository != PGLIB_OFFICIAL_REPOSITORY:
            raise PGLibImportError("Only the official PGLib-OPF repository is accepted")
        if self.data_license != PGLIB_DATA_LICENSE:
            raise PGLibImportError("PGLib data must retain its CC-BY-4.0 license identity")
        if not _RELEASE.fullmatch(self.release_tag):
            raise PGLibImportError("PGLib release_tag must be a pinned vYY.MM release")
        if not _HEX40.fullmatch(self.source_commit):
            raise PGLibImportError("PGLib source_commit must be a lowercase 40-hex commit")
        if not _HEX64.fullmatch(self.asset_sha256):
            raise PGLibImportError("PGLib asset_sha256 must be a lowercase SHA-256")
        if self.variant not in {"typical", "api", "sad"}:
            raise PGLibImportError("PGLib variant must be typical, api, or sad")
        if self.case_role not in {"development", "validation", "protected_test"}:
            raise PGLibImportError("Unsupported PGLib case role")
        if not self.attribution.strip():
            raise PGLibImportError("PGLib attribution must be non-empty")

        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or "\\" in self.relative_path
            or ".." in path.parts
            or str(path) != self.relative_path
            or path.suffix != ".m"
        ):
            raise PGLibImportError("PGLib relative_path must be a normalized relative .m path")
        expected_parent = (
            PurePosixPath(".") if self.variant == "typical" else PurePosixPath(self.variant)
        )
        if path.parent != expected_parent:
            raise PGLibImportError("PGLib relative_path does not match its declared variant")
        if not path.stem.startswith("pglib_opf_case") or not _IDENTIFIER.fullmatch(path.stem):
            raise PGLibImportError("PGLib filename must use the official case identifier form")
        suffix = {"typical": "", "api": "__api", "sad": "__sad"}[self.variant]
        if suffix and not path.stem.endswith(suffix):
            raise PGLibImportError("PGLib filename suffix does not match its declared variant")

    @property
    def source_url(self) -> str:
        return f"{self.repository}/blob/{self.source_commit}/{self.relative_path}"


def load_pglib_source_manifest(path: str | Path, *, expected_sha256: str) -> PGLibSourceManifest:
    """Load a strict source manifest after verifying its exact serialized bytes."""

    if not _HEX64.fullmatch(expected_sha256):
        raise PGLibImportError("Manifest expected_sha256 must be a lowercase SHA-256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PGLibImportError("PGLib source manifest SHA-256 mismatch")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PGLibImportError("PGLib source manifest must be valid UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != _MANIFEST_FIELDS:
        raise PGLibImportError("PGLib source manifest fields must exactly match the schema")
    try:
        manifest = PGLibSourceManifest(**data)
        manifest.validate()
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PGLibImportError):
            raise
        raise PGLibImportError(f"Invalid PGLib source manifest: {exc}") from exc
    return manifest


def _without_comments(source: str) -> str:
    # PGLib numeric case files do not need MATLAB string comments.  Removing from
    # the first percent sign is therefore safe for the deliberately tiny grammar.
    return "\n".join(line.split("%", 1)[0] for line in source.splitlines())


def _parse_number(token: str, *, field: str) -> float:
    if not _NUMBER.fullmatch(token):
        raise PGLibImportError(f"Non-numeric or non-finite token in mpc.{field}: {token!r}")
    value = float(token)
    if not np.isfinite(value):
        raise PGLibImportError(f"Non-finite value in mpc.{field}")
    return value


def _parse_matrix(value: str, *, field: str) -> np.ndarray:
    if not (value.startswith("[") and value.endswith("]")):
        raise PGLibImportError(f"mpc.{field} must be a literal numeric matrix")
    interior = value[1:-1]
    if "[" in interior or "]" in interior:
        raise PGLibImportError(f"Nested expressions are forbidden in mpc.{field}")
    rows: list[list[float]] = []
    for raw_row in interior.split(";"):
        tokens = [token for token in re.split(r"[\s,]+", raw_row.strip()) if token]
        if not tokens:
            continue
        rows.append([_parse_number(token, field=field) for token in tokens])
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise PGLibImportError(f"mpc.{field} must be a non-empty rectangular matrix")
    return np.asarray(rows, dtype=float)


def _parse_restricted_matpower(source: str, *, expected_function: str) -> dict[str, object]:
    text = _without_comments(source).strip()
    function = re.match(
        r"function\s+(?:\[?\s*)?mpc(?:\s*\]?)?\s*=\s*([A-Za-z][A-Za-z0-9_]*)\s*(?:\r?\n|;)",
        text,
    )
    if function is None or function.group(1) != expected_function:
        raise PGLibImportError("MATPOWER function declaration must exactly match the asset name")
    position = function.end()
    assignments: dict[str, object] = {}
    while position < len(text):
        whitespace = re.match(r"\s*", text[position:])
        assert whitespace is not None
        position += whitespace.end()
        if position >= len(text):
            break
        assignment = re.match(r"mpc\.([A-Za-z][A-Za-z0-9_]*)\s*=\s*", text[position:])
        if assignment is None:
            raise PGLibImportError("Executable or unsupported MATLAB syntax is forbidden")
        field = assignment.group(1)
        position += assignment.end()
        if field not in _ALLOWED_FIELDS or field in assignments:
            raise PGLibImportError(f"Unsupported or duplicate MATPOWER field: mpc.{field}")
        if position >= len(text):
            raise PGLibImportError(f"Missing literal value for mpc.{field}")
        if text[position] == "'":
            end = text.find("'", position + 1)
            if end < 0:
                raise PGLibImportError(f"Unterminated string literal for mpc.{field}")
            raw_value = text[position : end + 1]
            position = end + 1
        elif text[position] == "[":
            end = text.find("]", position + 1)
            if end < 0:
                raise PGLibImportError(f"Unterminated matrix for mpc.{field}")
            raw_value = text[position : end + 1]
            position = end + 1
        else:
            end = text.find(";", position)
            if end < 0:
                raise PGLibImportError(f"Unterminated scalar for mpc.{field}")
            raw_value = text[position:end].strip()
            position = end
        terminator = re.match(r"\s*;", text[position:])
        if terminator is None:
            raise PGLibImportError(f"mpc.{field} assignment must end with a semicolon")
        position += terminator.end()

        if field == "version":
            if raw_value != "'2'":
                raise PGLibImportError("Only MATPOWER case format version 2 is supported")
            assignments[field] = "2"
        elif field == "baseMVA":
            assignments[field] = _parse_number(raw_value, field=field)
        else:
            assignments[field] = _parse_matrix(raw_value, field=field)

    missing = _REQUIRED_FIELDS.difference(assignments)
    if missing:
        raise PGLibImportError(f"Missing required MATPOWER fields: {sorted(missing)}")
    return assignments


def import_pglib_case(
    path: str | Path,
    manifest: PGLibSourceManifest,
    *,
    allow_protected_test: bool = False,
) -> PowerSystemCase:
    """Verify and parse a local official PGLib asset without evaluating source code.

    Protected-test assets remain closed unless the caller crosses the explicit test-only
    boundary.  No network access, download, MATLAB engine, ``eval`` or Python import occurs.
    """

    manifest.validate()
    if manifest.case_role == "protected_test" and not allow_protected_test:
        raise PGLibImportError("Protected-test PGLib assets require explicit test-only access")
    asset = Path(path)
    if not asset.is_file() or asset.name != PurePosixPath(manifest.relative_path).name:
        raise PGLibImportError("Local PGLib asset name must match the pinned manifest")
    raw = asset.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != manifest.asset_sha256:
        raise PGLibImportError("PGLib asset SHA-256 mismatch")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PGLibImportError("PGLib asset must be valid UTF-8 text") from exc
    name = PurePosixPath(manifest.relative_path).stem
    data = _parse_restricted_matpower(source, expected_function=name)
    case = PowerSystemCase.from_dict(data, name=name)
    report = validate_case(case)
    try:
        report.require_valid()
    except ValueError as exc:
        raise PGLibImportError(f"Imported PGLib case failed physical validation: {exc}") from exc
    provenance = asdict(manifest)
    provenance.update(
        {
            "source_url": manifest.source_url,
            "physical_case_checksum": case.checksum(),
            "ignored_matpower_fields": sorted(_OPTIONAL_FIELDS.intersection(data)),
            "import_semantics": "AC-OPF case only; no ORPD controls inferred",
        }
    )
    case.source_provenance = provenance
    return case


def available_bundled_pglib_cases() -> tuple[str, ...]:
    """Return the small, licensed, non-protected PGLib validation set."""

    return tuple(BUNDLED_PGLIB_CASES)


def load_bundled_pglib_case(name: str) -> PowerSystemCase:
    """Verify both manifest and asset before loading a bundled validation case."""

    try:
        record = BUNDLED_PGLIB_CASES[name]
    except KeyError as exc:
        raise PGLibImportError(f"Unknown bundled PGLib case: {name}") from exc
    root = resources.files("calo_rpd_studio").joinpath("data", "pglib", "v23.07")
    manifest_resource = root.joinpath(*record["manifest"].split("/"))
    asset_resource = root.joinpath(*record["asset"].split("/"))
    with resources.as_file(manifest_resource) as manifest_path:
        manifest = load_pglib_source_manifest(
            manifest_path, expected_sha256=record["manifest_sha256"]
        )
    with resources.as_file(asset_resource) as asset_path:
        return import_pglib_case(asset_path, manifest)
