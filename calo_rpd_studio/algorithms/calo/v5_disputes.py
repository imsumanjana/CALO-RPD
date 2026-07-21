"""Audited CALO-RPD v5.0 scientific/performance dispute register.

This register deliberately separates implemented safeguards from unresolved research/performance
work.  PARTIAL, OPEN, and DEFERRED items must not be described as solved in release notes or
publications.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dispute:
    id: str
    status: str
    severity: str
    finding: str
    evidence_or_action: str


DISPUTES: tuple[Dispute, ...] = (
    Dispute(
        "P01",
        "RESOLVED",
        "Critical",
        "Strict experiments require an immutable policy artifact instead of silent policy fallback.",
        "Policy ID/path/SHA/schema binding is stored per experiment; changing the global active policy never mutates an existing experiment binding.",
    ),
    Dispute(
        "P02",
        "PARTIAL",
        "Critical",
        "Policy training and full CALO runtime are not one bit-identical transition implementation.",
        "The native 32-feature training environment exposes v4/v5 cognition semantics, but the lightweight PPO rollout environment remains separate. Real-optimizer Policy Qualification is mandatory.",
    ),
    Dispute(
        "P03",
        "RESOLVED",
        "High",
        "The policy could not observe HPEM/readiness/precision/variable-group state.",
        "The native 32-feature policy schema exposes eight bounded v4.1+ cognition features and is retained unchanged in v5 for checkpoint compatibility.",
    ),
    Dispute(
        "P04",
        "RESOLVED",
        "Critical",
        "No formal policy qualification gate existed.",
        "Candidate vs reference vs No-AI paired qualification, qualification history, grades, and immutable SHA provenance are implemented.",
    ),
    Dispute(
        "P05",
        "PARTIAL",
        "High",
        "Long-running policy evolution needs latest-vs-best checkpoint management.",
        "v5 policy lineages, cumulative epochs, immutable deployable checkpoints, latest/best-qualified roles, forks, exact resume, and continuation phases are implemented. Qualification scheduling is suggested but not automatically launched at every configured interval.",
    ),
    Dispute(
        "P06",
        "OPEN",
        "High",
        "The bundled legacy policy has no demonstrated native ORPD superiority.",
        "Legacy/unqualified policies remain labeled honestly. No policy becomes scientifically preferred until recorded paired qualification demonstrates it.",
    ),
    Dispute(
        "P07",
        "OPEN",
        "High",
        "AI/rule/online-credit authority weights are not established as universally optimal.",
        "Use paired IEEE 30/57 development ablations and freeze before IEEE 118/300 holdout evaluation.",
    ),
    Dispute(
        "P08",
        "DEFERRED",
        "Critical performance",
        "CALO does not yet use the same fully Torch-native optimizer-control path as Torch baseline optimizers.",
        "The scientific evaluator is accelerator-native, but a complete CUDA/XPU CALO-control rewrite remains deferred until seeded scientific parity can be proven.",
    ),
    Dispute(
        "P09",
        "DEFERRED",
        "Critical performance",
        "CALO cognitive/memory control state remains primarily compact NumPy host state.",
        "v5 does not falsely claim full device-resident CALO control; this remains a separate performance/scientific-parity project.",
    ),
    Dispute(
        "P10",
        "PARTIAL",
        "High performance",
        "Per-learner candidate decisions still include Python-level logic.",
        "Bounded tensor-shaped memories and reused 3D scratch reduce overhead while preserving seeded behavior; grouped device kernels remain future work.",
    ),
    Dispute(
        "P11",
        "OPEN",
        "High performance",
        "CUDA policy inference still materializes small action arrays on the host for NumPy CALO control.",
        "Elimination depends on the future device-resident CALO control path.",
    ),
    Dispute(
        "P12",
        "RESOLVED",
        "High diagnostic",
        "Evaluator warm-up could be mistaken for end-to-end CALO throughput.",
        "Evaluator calibration is labeled explicitly; CALO metadata records policy, control/candidate, evaluator, and learning-update timings separately.",
    ),
    Dispute(
        "P13",
        "RESOLVED",
        "Medium",
        "Persistent exact-evaluation caching could cost more than it saves at low hit rates.",
        "Within-batch exact dedup remains; persistent cross-batch cache auto-disables after measured low reuse while FE accounting is unchanged.",
    ),
    Dispute(
        "P14",
        "PARTIAL",
        "Medium-high",
        "HPEM/context/success-memory operations still include small host-side loops.",
        "Bounded memories keep cost controlled; further device kernels require parity-qualified implementation.",
    ),
    Dispute(
        "P15",
        "OPEN",
        "Medium",
        "Cross-run policy broker can add latency for an isolated single CALO run.",
        "Retain it for multi-run batching; add a direct fast path only after end-to-end profiling shows a reproducible benefit.",
    ),
    Dispute(
        "R01",
        "RESOLVED",
        "Critical reproducibility",
        "Workflow/setup state was not restored when reopening experiments.",
        "ExperimentWorkspaceRestorer restores authoritative config, workflow access, policy binding, intelligence selections, historical plots, and lightweight view state.",
    ),
    Dispute(
        "R02",
        "RESOLVED",
        "Critical reproducibility",
        "Old FE-horizon evidence could be overwritten or silently mixed after extensions.",
        "Run-horizon snapshots, revision records, horizon-aware statistics/results/export, validation-at-horizon records, and revision-specific export folders preserve evidence boundaries.",
    ),
    Dispute(
        "R03",
        "PARTIAL",
        "Medium",
        "A crash can lose live telemetry after the last committed optimizer checkpoint.",
        "CALO exact run checkpoints preserve optimization state for continuation, but every transient/downsampled GUI telemetry point between checkpoints is not guaranteed to survive a hard crash.",
    ),
    Dispute(
        "C01",
        "RESOLVED",
        "Critical",
        "Policy training could not continue safely across sessions or completed targets.",
        "Exact resume retains model, optimizer, RNG, curriculum/history and blocks scientific hyperparameter drift; cumulative, additional, and indefinite modes are supported.",
    ),
    Dispute(
        "C02",
        "RESOLVED",
        "High",
        "More policy epochs could destructively replace a better earlier policy.",
        "Immutable lineage checkpoints keep latest and best-qualified roles separate; experiments remain bound to their original checkpoint SHA.",
    ),
    Dispute(
        "C03",
        "PARTIAL",
        "Critical",
        "Exact same-trajectory FE-horizon continuation is not available for every optimizer.",
        "CALO has full exact optimizer-state continuation. For multi-algorithm publication comparisons, v5 provides paired recompute-from-original-seed at the new horizon; unsupported baselines are never mislabeled as exact continuation.",
    ),
    Dispute(
        "C04",
        "RESOLVED",
        "Critical scientific",
        "Post-hoc extension of only good-looking runs could bias primary statistics.",
        "All-paired and predeclared deterministic-subset protocols are publication-eligible; manual/post-hoc selection is explicitly exploratory and excluded from unbiased primary claims.",
    ),
    Dispute(
        "C05",
        "RESOLVED",
        "High",
        "A later longer exploratory branch could redefine or destroy the primary comparison horizon.",
        "Primary completed publication-eligible revisions are tracked independently of exploratory branches; later paired revisions can branch from the correct primary/source horizon.",
    ),
    Dispute(
        "C06",
        "RESOLVED",
        "High",
        "Exact continuation always resumed whichever branch happened to be current.",
        "v5 records/selects an explicit source FE horizon and resolves the checkpoint belonging to that preserved run segment; revision-scoped output paths prevent checkpoint overwrite.",
    ),
    Dispute(
        "C07",
        "RESOLVED",
        "High scientific",
        "A segmented 5k→10k continuation could be incorrectly presented as identical to a run planned for 10k from FE=0.",
        "Trajectory semantics are recorded explicitly. Exact continuation is labeled segmented; recompute-from-seed is the publication-safe from-start higher-horizon trajectory for paired comparisons.",
    ),
    Dispute(
        "C08",
        "OPEN",
        "Medium",
        "Configured policy qualification intervals do not automatically execute expensive qualification campaigns during training.",
        "The interval is currently advisory. Automatic asynchronous qualification/promotion requires a separately budgeted, non-training validation scheduler to avoid leakage and resource contention.",
    ),
)


def as_rows() -> list[dict[str, str]]:
    return [
        {
            "id": item.id,
            "status": item.status,
            "severity": item.severity,
            "finding": item.finding,
            "evidence_or_action": item.evidence_or_action,
        }
        for item in DISPUTES
    ]
