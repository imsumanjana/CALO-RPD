"""Audited CALO-RPD v5.9 scientific/performance dispute register.

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
        "Policy training and deployed CALO use separate code paths that must remain transition-equivalent.",
        "v5.9 versions a native 32-feature raw-policy/executed-controller ABI and regression-tests seeded multi-step training/runtime transition parity (population, pbest, RNG, epsilon, contextual credit, executed operators and reward). The implementations remain separate, so this parity test stays a mandatory release gate.",
    ),
    Dispute(
        "P03",
        "RESOLVED",
        "High",
        "The policy could not observe HPEM/readiness/precision/variable-group state.",
        "The native v5.9 32-feature policy schema exposes eight bounded runtime-context features in addition to the 24-D cognitive base; legacy policies are isolated behind explicit compatibility/migration boundaries.",
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
        "RESOLVED",
        "High",
        "Long-running policy evolution needs a non-destructive distinction between resumable working state and best deployable policy.",
        "v5.9 keeps exact per-branch resume state separate from branch champions and the logical Base, commits exact branches as immutable all-or-nothing generations, and preserves immutable experiment-bound SHA artifacts. Formal qualification remains a separate gate tracked by C08.",
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
        "v5.9 preserves the seeded core CALO search semantics while versioning policy/controller semantics explicitly. The per-learner stochastic candidate path remains partly Python/host-side; a deeper vectorized/device rewrite is deferred until exact seeded/scientific parity can be demonstrated under a separately qualified algorithm-version protocol.",
    ),
    Dispute(
        "P11",
        "RESOLVED",
        "High performance",
        "CUDA policy inference could add pointless device synchronization/host materialization while CALO control remains NumPy-host based.",
        "v5.9 defaults policy inference to CPU for the current host-control architecture and keeps the ORPD numerical evaluator accelerator-native. Explicit accelerator policy inference remains experimental until CALO control itself is device-resident.",
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
        "RESOLVED",
        "Medium",
        "Cross-run policy/evaluation brokering could add batching latency to isolated single-run execution.",
        "v5.9 creates cross-run brokers only when cross-run batching is enabled and more than one execution slot exists; single-slot execution uses the direct path.",
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
        "v5.9 supports cumulative and infinite-until-Safe-Stop sessions plus exact branch resume with model/optimizer/RNG/curriculum restoration. Session duration is decoupled from immutable curriculum milestones, and resume-critical history is bounded.",
    ),
    Dispute(
        "C02",
        "RESOLVED",
        "High",
        "More policy epochs or parallel training could destructively replace a better earlier policy.",
        "v5.9 separates exact resumable branch state from champion/Base state, never averages independent PPO weights, re-evaluates candidates under one fingerprinted validation bundle, and uses order-independent feasibility-first Base selection while preserving immutable experiment-bound SHA artifacts.",
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
        "C09",
        "RESOLVED",
        "Critical scientific",
        "Independent PPO branches were previously vulnerable to scientifically unjustified terminal parameter averaging and incoherent merged optimizer/RNG state.",
        "v5.9 competitive multi-branch policy evolution keeps every branch independent and exactly resumable in transactional immutable generations. Final candidates are evaluated under one common validation bundle with deterministic order-independent feasibility-first ranking; neural parameters, optimizer state, RNG state, and curriculum state are never arithmetically merged.",
    ),
    Dispute(
        "C08",
        "RESOLVED",
        "Medium",
        "Periodic in-training formal qualification could be mistaken for automatic scientific promotion and would conflict with the no-intermediate-permanent-snapshot training contract.",
        "v5.9 retires periodic formal qualification by design: the compatibility field is fixed at 0, and formal paired Candidate-vs-Reference-vs-No-AI qualification is run only for saved Base artifacts under a separately budgeted campaign.",
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
