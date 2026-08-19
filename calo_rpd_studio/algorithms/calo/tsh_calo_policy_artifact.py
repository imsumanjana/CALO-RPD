"""Public immutable TSH-CALO candidate API with ensemble guard homogeneity checks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from . import _tsh_calo_policy_artifact_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_core_inspect_tsh_calo_candidate = _core.inspect_tsh_calo_candidate
_core_assemble_tsh_calo_ensemble_candidate = _core.assemble_tsh_calo_ensemble_candidate


def _validate_ensemble_generalization_homogeneity_rows(rows: Sequence[dict]) -> None:
    """Reject mixed or differently guarded ensemble members before promotion can inspect them."""

    if len(rows) < 2:
        raise ValueError("TSH-CALO ensemble requires at least two training-provenance rows")
    guarded: list[bool] = []
    designs: set[str] = set()
    for index, source in enumerate(rows):
        if not isinstance(source, dict):
            raise ValueError(f"TSH-CALO ensemble member {index + 1} provenance is invalid")
        row = dict(source)
        payload = row.get("generalization_guard")
        declared = str(row.get("generalization_guard_sha256", "") or "").strip().lower()
        has_payload = isinstance(payload, dict) and bool(payload)
        has_declaration = bool(declared)
        if has_payload != has_declaration:
            raise ValueError(
                f"TSH-CALO ensemble member {index + 1} has an incomplete guard binding"
            )
        guarded.append(has_payload)
        if has_payload:
            designs.add(declared)
    if any(guarded) and not all(guarded):
        raise ValueError(
            "TSH-CALO ensemble cannot mix guarded and legacy unguarded policy members"
        )
    if all(guarded) and len(designs) != 1:
        raise ValueError(
            "TSH-CALO ensemble members must share one generalization-guard design"
        )


def _validate_ensemble_generalization_homogeneity(artifact) -> None:
    if artifact.artifact_kind != "ensemble_policy":
        return
    provenance = dict(artifact.training_provenance or {})
    members = list(provenance.get("members", []) or [])
    rows = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError(f"TSH-CALO ensemble member {index + 1} record is invalid")
        rows.append(dict(member.get("training_provenance", {}) or {}))
    _validate_ensemble_generalization_homogeneity_rows(rows)


def inspect_tsh_calo_candidate(
    path: str | Path, *, expected_sha256: str | None = None
):
    """Inspect a candidate and enforce guard homogeneity for every ensemble artifact."""

    artifact = _core_inspect_tsh_calo_candidate(
        path,
        expected_sha256=expected_sha256,
    )
    _validate_ensemble_generalization_homogeneity(artifact)
    return artifact


def assemble_tsh_calo_ensemble_candidate(
    path: str | Path,
    members: Sequence[tuple[str | Path, str]],
):
    """Assemble only members with a homogeneous guard declaration and design."""

    artifacts = [
        inspect_tsh_calo_candidate(member_path, expected_sha256=expected_sha256)
        for member_path, expected_sha256 in members
    ]
    _validate_ensemble_generalization_homogeneity_rows(
        [dict(artifact.training_provenance or {}) for artifact in artifacts]
    )
    artifact = _core_assemble_tsh_calo_ensemble_candidate(path, members)
    _validate_ensemble_generalization_homogeneity(artifact)
    return artifact


# Core save/load/assembly helpers resolve this global at call time.  Point them at the public
# invariant boundary so crafted ensemble checkpoints cannot bypass the added inspection rule.
_core.inspect_tsh_calo_candidate = inspect_tsh_calo_candidate


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"Path", "Sequence", "annotations"}
    )
)
