"""Audited CALO v4.1 dispute register.

This is an engineering/scientific register, not a marketing checklist.  Items marked DEFERRED or
PARTIAL must not be described as solved in release notes or publications.
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
        "Strict v4.1 experiments must bind an immutable policy; legacy v2 fallback is no longer silently accepted in a strict binding.",
        "Policy registry + checksum/schema binding; legacy fallback retained only for backwards-compatible non-strict scripts.",
    ),
    Dispute(
        "P02",
        "PARTIAL",
        "Critical",
        "Historical policy-training dynamics did not expose v4 runtime cognition.",
        "Native 32-feature v4.1 training environment now includes persistent pbest, HPEM, contextual credit, variable intelligence, adaptive epsilon, dual-lane readiness and recovery semantics. A single shared transition implementation remains future work.",
    ),
    Dispute(
        "P03",
        "RESOLVED",
        "High",
        "The policy could not observe important v4 memory/precision states.",
        "Native v4.1 state adds HPEM occupancy, consensus, readiness, success-memory density, learning-lane fraction, precision state/radius and variable-group concentration.",
    ),
    Dispute(
        "P04",
        "RESOLVED",
        "Critical",
        "No formal policy qualification gate existed.",
        "Candidate vs reference vs No-AI paired qualification, immutable SHA provenance, grades, paired evidence, and qualification history are implemented.",
    ),
    Dispute(
        "P05",
        "DEFERRED",
        "High",
        "Training completion still does not automatically select the best independently qualified checkpoint across a long training lineage.",
        "Policy lineage/indefinite checkpoint evolution is intentionally reserved for v5.0; v4.1 registers each completed checkpoint as a candidate requiring qualification.",
    ),
    Dispute(
        "P06",
        "OPEN",
        "High",
        "The bundled legacy policy has no demonstrated native-v4.1 ORPD qualification.",
        "The UI labels it legacy/unqualified until the user runs a recorded qualification. No superiority claim is made.",
    ),
    Dispute(
        "P07",
        "OPEN",
        "High",
        "AI/rule/online-credit authority weights are not yet empirically calibrated as universally optimal.",
        "Use paired development-case ablation before changing frozen defaults; IEEE 118/300 remain protected holdouts.",
    ),
    Dispute(
        "P08",
        "DEFERRED",
        "Critical performance",
        "CALO does not yet use the same fully Torch-native optimizer-control path as baseline Torch optimizers.",
        "Scientific evaluator remains accelerator-native; a complete CUDA/XPU CALO-control rewrite requires strict seeded parity tests and is deferred rather than risking equation drift.",
    ),
    Dispute(
        "P09",
        "DEFERRED",
        "Critical performance",
        "CALO cognitive/memory control state remains primarily compact NumPy host state.",
        "v4.1 reduces host overhead and records control/evaluator timing, but does not falsely claim fully device-resident CALO control.",
    ),
    Dispute(
        "P10",
        "PARTIAL",
        "High performance",
        "Per-learner candidate decisions still contain Python-level logic.",
        "v4.1 removes repeated per-learner best/mean/variable-schema construction while preserving the frozen seeded peer-sampling sequence; complete grouped operator kernels remain future work.",
    ),
    Dispute(
        "P11",
        "OPEN",
        "High performance",
        "CUDA policy inference still materializes small action arrays on the host for NumPy CALO control.",
        "Eliminating this transfer depends on the future fully device-resident control path.",
    ),
    Dispute(
        "P12",
        "RESOLVED",
        "High diagnostic",
        "Microbatch warm-up looked like end-to-end CALO throughput although it measured only the evaluator.",
        "GUI/log wording now explicitly says evaluator-only; CALO result metadata separately reports policy, candidate-generation, evaluator, and learning-update timing.",
    ),
    Dispute(
        "P13",
        "RESOLVED",
        "Medium",
        "Persistent exact-evaluation caching could cost more than it saved at very low cross-batch hit rates.",
        "Within-batch exact dedup always remains; persistent cross-batch storage auto-disables after sufficient evidence when its hit rate is below threshold.",
    ),
    Dispute(
        "P14",
        "PARTIAL",
        "Medium-high",
        "HPEM/context/success-memory operations include small host-side loops.",
        "Bounded memories keep cost small and 3D scratch is reused; further device kernels require parity-qualified future work.",
    ),
    Dispute(
        "P15",
        "OPEN",
        "Medium",
        "Cross-run policy broker may add latency for an isolated single CALO run.",
        "Retain broker for multi-run batching; add a measured direct-inference fast path only after end-to-end profiling proves benefit.",
    ),
    Dispute(
        "R01",
        "RESOLVED",
        "Critical reproducibility",
        "Workflow completion state was not restored.",
        "Workflow snapshots are persisted; older experiments without snapshots infer completed setup stages from authoritative experiment existence.",
    ),
    Dispute(
        "R02",
        "RESOLVED",
        "Critical UX",
        "Old setup parameter widgets did not fully rehydrate.",
        "Power-system, algorithm, robust-scenario and CALO policy/intelligence loaders now restore from the saved ExperimentConfig; connected panels refresh from config_changed.",
    ),
    Dispute(
        "R03",
        "RESOLVED",
        "Critical reproducibility",
        "CALO Intelligence did not restore the exact experiment policy/intelligence selection.",
        "Experiment policy ID/path/SHA/schema and historical-learning controls are restored from immutable config/binding.",
    ),
    Dispute(
        "R04",
        "RESOLVED",
        "Critical UX",
        "LiveOptimizationPanel.load_experiment existed but was not wired into resume/open.",
        "Central ExperimentWorkspaceRestorer now invokes it for resume and historical workspace opening.",
    ),
    Dispute(
        "R05",
        "RESOLVED",
        "High",
        "Previously completed convergence graphs were not automatically reconstructed.",
        "Stored numeric run histories rebuild live/convergence views; no screenshot is treated as scientific source data.",
    ),
    Dispute(
        "R06",
        "DEFERRED",
        "Medium",
        "A crash mid-run does not yet persist every downsampled live-telemetry point for exact visual reconstruction before the latest committed run checkpoint.",
        "Deep run-checkpoint/telemetry continuation belongs to the planned v5.0 continuation architecture.",
    ),
    Dispute(
        "R07",
        "RESOLVED",
        "Medium",
        "Selected run/plot/view state was lost.",
        "Lightweight Live Optimization view state and last workspace are persisted separately from scientific data and restored on reopen.",
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
