"""Public TSH-CALO generalization-guard API with ensemble-wide invariants.

The implementation remains in ``_tsh_calo_generalization_guard_core`` so this boundary can enforce
cross-member rules without duplicating the scientific evaluator.  A guarded ensemble is valid only
when every member declares and passes the same guard design; a mixed guarded/legacy ensemble must
never be presented as having passed.
"""

from __future__ import annotations

from . import _tsh_calo_generalization_guard_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_single_candidate_generalization_status = _core.candidate_generalization_status


def candidate_generalization_status(training_provenance: dict) -> tuple[bool, str]:
    """Return a fail-closed, homogeneous ensemble learning-health status."""

    provenance = dict(training_provenance or {})
    if provenance.get("source_kind") != "independent_policy_training_ensemble":
        return _single_candidate_generalization_status(provenance)

    members = list(provenance.get("members", []) or [])
    if len(members) < 2:
        return False, "TSH-CALO ensemble training provenance has fewer than two members."

    rows: list[dict] = []
    guarded: list[bool] = []
    designs: set[str] = set()
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            return False, f"TSH-CALO ensemble member {index + 1} provenance is invalid."
        row = dict(member.get("training_provenance", {}) or {})
        payload = row.get("generalization_guard")
        declared = str(row.get("generalization_guard_sha256", "") or "").strip().lower()
        has_payload = isinstance(payload, dict) and bool(payload)
        has_declaration = bool(declared)
        if has_payload != has_declaration:
            return (
                False,
                f"Ensemble member {index + 1} has an incomplete generalization-guard binding.",
            )
        rows.append(row)
        guarded.append(has_payload)
        if has_payload:
            designs.add(declared)

    if any(guarded) and not all(guarded):
        return (
            False,
            "TSH-CALO ensemble mixes guarded and legacy unguarded members; promotion is forbidden.",
        )
    if all(guarded) and len(designs) != 1:
        return (
            False,
            "TSH-CALO ensemble members use different generalization-guard designs.",
        )

    for index, row in enumerate(rows):
        allowed, reason = _single_candidate_generalization_status(row)
        if not allowed:
            return (
                False,
                f"Ensemble member {index + 1} learning guard rejected promotion: {reason}",
            )
    return (
        True,
        "passed" if all(guarded) else "legacy_candidate_without_generalization_guard",
    )


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"annotations"}
    )
)
