"""Policy-registry boundary for exact training plus learning-health accounting.

The established lifecycle registry remains in ``_policy_registry_core``.  This public boundary keeps
its legacy training-only count intact and adds a separately named cumulative counted-work contract.
Exact cumulative guard work is persisted when a completed campaign manifest is available; a root
candidate can be reconstructed from its authenticated artifact, while an extended artifact without
its predecessor manifests fails closed instead of under-reporting prior guard work.
"""

from __future__ import annotations

from pathlib import Path

from . import _policy_registry_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _valid_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _candidate_evaluation_accounting(artifact) -> dict | None:
    training = _core.count_tsh_calo_candidate_training_evaluations(artifact)
    provenance = dict(getattr(artifact, "training_provenance", {}) or {})
    if provenance.get("source_kind") == "independent_policy_training_ensemble":
        members = list(provenance.get("members", []) or [])
        rows = []
        for member in members:
            if not isinstance(member, dict):
                return None
            rows.append(dict(member.get("training_provenance", {}) or {}))
    else:
        rows = [provenance]

    guard_total = 0
    for row in rows:
        guard = row.get("generalization_guard")
        if guard is None:
            continue
        if not isinstance(guard, dict):
            return None
        offset = guard.get("segment_receipt_offset", 0)
        additional = guard.get("additional_candidate_evaluations", 0)
        if not _valid_count(offset) or not _valid_count(additional):
            return None
        # An extended candidate retains the current segment's guard evidence and cumulative policy
        # receipts, but not every predecessor segment's guard bundles.  Its manifest is required.
        if int(offset) != 0:
            return None
        guard_total += int(additional)
    return {
        "training_candidate_evaluations": int(training),
        "generalization_guard_candidate_evaluations": int(guard_total),
        "total_counted_candidate_evaluations": int(training + guard_total),
    }


def _metadata_evaluation_accounting(metadata: dict) -> dict | None:
    training = metadata.get("training_candidate_evaluations")
    guard = metadata.get("generalization_guard_candidate_evaluations")
    total = metadata.get("total_counted_candidate_evaluations")
    if not all(_valid_count(value) for value in (training, guard, total)):
        return None
    if int(total) != int(training) + int(guard):
        return None
    return {
        "training_candidate_evaluations": int(training),
        "generalization_guard_candidate_evaluations": int(guard),
        "total_counted_candidate_evaluations": int(total),
    }


class PolicyRegistry(_core.PolicyRegistry):
    """Retain exact counted-work metadata without changing lifecycle or activation authority."""

    def bind_training_evaluation_accounting(
        self,
        policy_id: str,
        accounting: dict,
    ):
        if not isinstance(accounting, dict):
            raise ValueError("Policy counted-evaluation accounting must be an object")
        values = {
            "training_candidate_evaluations": accounting.get(
                "training_candidate_evaluations"
            ),
            "generalization_guard_candidate_evaluations": accounting.get(
                "generalization_guard_candidate_evaluations"
            ),
            "total_counted_candidate_evaluations": accounting.get(
                "total_counted_candidate_evaluations"
            ),
        }
        if not all(_valid_count(value) for value in values.values()):
            raise ValueError("Policy counted-evaluation accounting is invalid")
        if values["total_counted_candidate_evaluations"] != (
            values["training_candidate_evaluations"]
            + values["generalization_guard_candidate_evaluations"]
        ):
            raise ValueError("Policy counted-evaluation accounting is inconsistent")
        policy = self.get(policy_id)
        training = _core.PolicyRegistry.training_evaluation_count(self, policy_id)
        if training is None or int(training) != values["training_candidate_evaluations"]:
            raise ValueError(
                "Policy counted-evaluation accounting does not match its authenticated receipts"
            )
        metadata = dict(policy.metadata)
        retained = _metadata_evaluation_accounting(metadata)
        if retained is not None and retained != values:
            raise ValueError("Registered policy counted-evaluation accounting changed")
        metadata.update(values)
        metadata.update(
            {
                "training_evaluation_count_scope": (
                    "cumulative_exact_training_candidate_evaluations"
                ),
                "counted_evaluation_count_scope": (
                    "cumulative_exact_training_and_learning_health_candidate_evaluations"
                ),
                "legacy_candidate_evaluation_fields_are_training_only": True,
            }
        )
        self.database.update_policy(policy.id, metadata_json=metadata)
        return self.get(policy.id)

    def training_evaluation_accounting(self, policy_id: str) -> dict | None:
        policy = self.get(policy_id)
        retained = _metadata_evaluation_accounting(dict(policy.metadata))
        if retained is not None:
            return retained
        if policy.algorithm_id != _core.TSH_CALO_ALGORITHM_ID or not policy.usable:
            return None
        try:
            artifact = _core.inspect_tsh_calo_candidate(
                policy.checkpoint_path,
                expected_sha256=policy.sha256,
            )
            return _candidate_evaluation_accounting(artifact)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

    def counted_evaluation_count(self, policy_id: str) -> int | None:
        accounting = self.training_evaluation_accounting(policy_id)
        return (
            int(accounting["total_counted_candidate_evaluations"])
            if accounting is not None
            else None
        )

    def register(
        self, path: str | Path, *, name: str | None = None, status: str | None = None
    ):
        policy = super().register(path, name=name, status=status)
        if _metadata_evaluation_accounting(dict(policy.metadata)) is not None:
            return policy
        accounting = self.training_evaluation_accounting(policy.id)
        if accounting is None:
            return policy
        try:
            return self.bind_training_evaluation_accounting(policy.id, accounting)
        except ValueError:
            # Registration integrity remains governed by the core registry.  A missing cumulative
            # extension manifest affects presentation only and must not fabricate a lower total.
            return self.get(policy.id)


_core.PolicyRegistry = PolicyRegistry


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"Path", "annotations"}
    )
)
