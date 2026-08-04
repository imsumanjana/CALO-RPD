"""Strict topology-state construction and permutation-consistent TSH-CALO encoding.

The builder consumes an already-computed power-flow result.  It never calls a solver, so policy
state construction cannot create hidden function evaluations.  Missing or failed power-flow state
is rejected rather than represented as a fabricated zero-stress network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from calo_rpd_studio.power_system.case_model import (
    ANGMAX,
    ANGMIN,
    BASE_KV,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    BS,
    BUS_I,
    F_BUS,
    GEN_BUS,
    GEN_STATUS,
    PD,
    PG,
    QD,
    QG,
    QMAX,
    QMIN,
    RATE_A,
    SHIFT,
    TAP,
    T_BUS,
    VMAX,
    VMIN,
)

from .policy_schema import POLICY_STATE_DIM
from .tsh_calo_schema import ControlGroup, N_CONTROL_GROUPS


BUS_FEATURE_DIM = 12
BRANCH_FEATURE_DIM = 10
CONTROL_FEATURE_DIM = 9
SCENARIO_FEATURE_DIM = 7


def _finite_array(name: str, value, shape: tuple[int | None, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}; received {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class ScenarioDescriptor:
    """Declared scenario context; every numeric field must be known by the caller."""

    load_stress: float
    renewable_stress: float
    contingency_stress: float
    aggregation_role: float
    weight: float
    is_ood: float
    observed: float = 1.0

    def as_array(self) -> np.ndarray:
        values = np.asarray(
            [
                self.load_stress,
                self.renewable_stress,
                self.contingency_stress,
                self.aggregation_role,
                self.weight,
                self.is_ood,
                self.observed,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Scenario descriptors must be finite and explicitly declared")
        if self.weight < 0.0:
            raise ValueError("Scenario descriptor weight cannot be negative")
        return values


@dataclass(frozen=True, slots=True)
class TopologyGraphState:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    control_features: np.ndarray
    control_bus_index: np.ndarray
    control_groups: np.ndarray
    scenario_features: np.ndarray
    bus_numbers: tuple[int, ...]
    branch_indices: tuple[int, ...]
    control_labels: tuple[str, ...]

    def validate(self) -> None:
        nodes = _finite_array("node_features", self.node_features, (None, BUS_FEATURE_DIM))
        edges = _finite_array("edge_features", self.edge_features, (None, BRANCH_FEATURE_DIM))
        controls = _finite_array(
            "control_features", self.control_features, (None, CONTROL_FEATURE_DIM)
        )
        scenarios = _finite_array(
            "scenario_features", self.scenario_features, (None, SCENARIO_FEATURE_DIM)
        )
        edge_index = np.asarray(self.edge_index)
        control_bus_index = np.asarray(self.control_bus_index)
        groups = np.asarray(self.control_groups)
        if edge_index.shape != (2, len(edges)) or not np.issubdtype(edge_index.dtype, np.integer):
            raise ValueError("edge_index must be an integer array with shape (2, n_edges)")
        if control_bus_index.shape != (2, len(controls)) or not np.issubdtype(
            control_bus_index.dtype, np.integer
        ):
            raise ValueError(
                "control_bus_index must be an integer array with shape (2, n_controls)"
            )
        if groups.shape != (len(controls),) or not np.issubdtype(groups.dtype, np.integer):
            raise ValueError("control_groups must be an integer vector aligned with controls")
        if len(nodes) == 0 or len(scenarios) == 0:
            raise ValueError("Topology state requires at least one bus and one scenario descriptor")
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(nodes)):
            raise ValueError("edge_index references a bus outside node_features")
        if control_bus_index.size and (
            control_bus_index.min() < 0 or control_bus_index.max() >= len(nodes)
        ):
            raise ValueError("control_bus_index references a bus outside node_features")
        if groups.size and (groups.min() < 0 or groups.max() >= N_CONTROL_GROUPS):
            raise ValueError("control_groups contains an unknown control group")
        if len(self.bus_numbers) != len(nodes):
            raise ValueError("bus_numbers must align with node_features")
        if len(self.branch_indices) != len(edges):
            raise ValueError("branch_indices must align with directed edge_features")
        if len(self.control_labels) != len(controls):
            raise ValueError("control_labels must align with control_features")


@dataclass(frozen=True, slots=True)
class TopologyAwarePolicyState:
    """TSH-CALO state: frozen aggregate cognition plus structured graph context."""

    aggregate_features: np.ndarray
    topology: TopologyGraphState

    def validate(self) -> None:
        _finite_array("aggregate_features", self.aggregate_features, (POLICY_STATE_DIM,))
        self.topology.validate()


@dataclass(slots=True)
class TopologyEncoding:
    graph_embedding: torch.Tensor
    group_embeddings: torch.Tensor
    node_embeddings: torch.Tensor
    control_embeddings: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedTopologyGraphState:
    """Validated topology tensors retained on one execution device across policy epochs."""

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    control_features: torch.Tensor
    control_bus_index: torch.Tensor
    control_groups: torch.Tensor
    scenario_features: torch.Tensor


def _aggregate_generators(case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    index = case.bus_index_map()
    pg = np.zeros(case.n_bus, dtype=float)
    qg = np.zeros(case.n_bus, dtype=float)
    reserve_up = np.zeros(case.n_bus, dtype=float)
    reserve_down = np.zeros(case.n_bus, dtype=float)
    for row in case.gen:
        if row[GEN_STATUS] <= 0:
            continue
        bus = index[int(row[GEN_BUS])]
        pg[bus] += float(row[PG])
        qg[bus] += float(row[QG])
        reserve_up[bus] += max(float(row[QMAX] - row[QG]), 0.0)
        reserve_down[bus] += max(float(row[QG] - row[QMIN]), 0.0)
    return pg, qg, reserve_up, reserve_down


def _control_rows(case, decoder, normalized_controls: np.ndarray):
    manifest = decoder.formulation_manifest()
    index = case.bus_index_map()
    rows: list[list[float]] = []
    attachments: list[tuple[int, int]] = []
    groups: list[int] = []
    labels: list[str] = []
    cursor = 0

    def append(group: ControlGroup, first_bus: int, second_bus: int, *, discrete: bool, label: str):
        nonlocal cursor
        value = float(normalized_controls[cursor])
        one_hot = [float(group == item) for item in ControlGroup]
        rows.append(
            one_hot
            + [
                value,
                value,
                1.0 - value,
                float(discrete),
                1.0,
                float(cursor) / max(len(normalized_controls), 1),
            ]
        )
        attachments.append((index[int(first_bus)], index[int(second_bus)]))
        groups.append(int(group))
        labels.append(label)
        cursor += 1

    for bus in manifest["generator_voltage_buses"]:
        append(
            ControlGroup.GENERATOR_VOLTAGE,
            int(bus),
            int(bus),
            discrete=False,
            label=f"Vg@{int(bus)}",
        )
    for tap in manifest["transformer_taps"]:
        append(
            ControlGroup.TRANSFORMER_TAP,
            int(tap["from_bus"]),
            int(tap["to_bus"]),
            discrete=bool(tap["discrete"]),
            label=f"Tap {int(tap['from_bus'])}-{int(tap['to_bus'])}",
        )
    for shunt in manifest["shunt_controls"]:
        bus = int(shunt["bus_number"])
        append(
            ControlGroup.SHUNT,
            bus,
            bus,
            discrete=bool(getattr(decoder.config, "discrete_shunts", True)),
            label=f"Shunt@{bus}",
        )
    if cursor != len(normalized_controls):
        raise ValueError(
            "Topology control manifest does not align with the normalized decision vector: "
            f"{cursor} controls for {len(normalized_controls)} values"
        )
    return rows, attachments, groups, labels


def build_topology_graph_state(
    case,
    decoder,
    normalized_controls,
    power_flow_result,
    scenario_descriptors: Sequence[ScenarioDescriptor],
) -> TopologyGraphState:
    """Build graph context solely from an already-counted converged power-flow result."""

    if power_flow_result is None or not bool(power_flow_result.converged):
        raise ValueError("Topology graph state requires an already-computed converged power flow")
    if power_flow_result.branch is None:
        raise ValueError("Topology graph state requires already-computed branch-flow results")
    if (
        power_flow_result.case.n_bus != case.n_bus
        or power_flow_result.case.n_branch != case.n_branch
    ):
        raise ValueError("Power-flow topology does not match the declared ORPD case")
    controls = _finite_array("normalized_controls", normalized_controls, (int(decoder.dimension),))
    if np.any((controls < 0.0) | (controls > 1.0)):
        raise ValueError("normalized_controls must remain within [0, 1]")

    solved = power_flow_result.case
    base_mva = max(float(solved.base_mva), 1e-12)
    pg, qg, reserve_up, reserve_down = _aggregate_generators(solved)
    vm = np.asarray(power_flow_result.vm_pu, dtype=float)
    va = np.asarray(power_flow_result.va_deg, dtype=float)
    voltage_low = np.maximum(solved.bus[:, VMIN] - vm, 0.0)
    voltage_high = np.maximum(vm - solved.bus[:, VMAX], 0.0)
    node_features = np.column_stack(
        [
            vm,
            va / 180.0,
            solved.bus[:, PD] / base_mva,
            solved.bus[:, QD] / base_mva,
            pg / base_mva,
            qg / base_mva,
            reserve_up / base_mva,
            reserve_down / base_mva,
            voltage_low,
            voltage_high,
            solved.bus[:, BS] / base_mva,
            np.maximum(solved.bus[:, BASE_KV], 0.0) / 1000.0,
        ]
    )

    bus_index = solved.bus_index_map()
    directed_edges: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    branch_indices: list[int] = []
    loading = np.asarray(power_flow_result.branch.loading_percent, dtype=float)
    for branch_index, branch in enumerate(solved.branch):
        if branch[BR_STATUS] <= 0:
            continue
        first = bus_index[int(branch[F_BUS])]
        second = bus_index[int(branch[T_BUS])]
        tap = float(branch[TAP]) if float(branch[TAP]) != 0.0 else 1.0
        rate = max(float(branch[RATE_A]), 0.0)
        angle = float(va[first] - va[second])
        lower_violation = max(float(branch[ANGMIN]) - angle, 0.0)
        upper_violation = max(angle - float(branch[ANGMAX]), 0.0)
        row = [
            float(branch[BR_R]),
            float(branch[BR_X]),
            float(branch[BR_B]),
            tap,
            float(branch[SHIFT]) / 180.0,
            rate / base_mva,
            float(loading[branch_index]) / 100.0,
            max(float(loading[branch_index]) / 100.0 - 1.0, 0.0),
            lower_violation / 180.0,
            upper_violation / 180.0,
        ]
        for source, target in ((first, second), (second, first)):
            directed_edges.append((source, target))
            edge_rows.append(row)
            branch_indices.append(branch_index)

    control_rows, attachments, groups, labels = _control_rows(solved, decoder, controls)
    scenarios = np.vstack([item.as_array() for item in scenario_descriptors])
    state = TopologyGraphState(
        node_features=np.asarray(node_features, dtype=np.float64),
        edge_index=np.asarray(directed_edges, dtype=np.int64).T.reshape(2, -1),
        edge_features=np.asarray(edge_rows, dtype=np.float64).reshape(-1, BRANCH_FEATURE_DIM),
        control_features=np.asarray(control_rows, dtype=np.float64).reshape(
            -1, CONTROL_FEATURE_DIM
        ),
        control_bus_index=np.asarray(attachments, dtype=np.int64).T.reshape(2, -1),
        control_groups=np.asarray(groups, dtype=np.int64),
        scenario_features=np.asarray(scenarios, dtype=np.float64),
        bus_numbers=tuple(int(item) for item in solved.bus[:, BUS_I]),
        branch_indices=tuple(branch_indices),
        control_labels=tuple(labels),
    )
    state.validate()
    return state


class TopologyContextEncoder(nn.Module):
    """Dependency-light message-passing encoder with invariant graph/group pooling."""

    def __init__(self, hidden_dim: int = 48, message_passing_steps: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.message_passing_steps = int(message_passing_steps)
        if self.hidden_dim < 4 or self.message_passing_steps < 1:
            raise ValueError("Topology encoder requires hidden_dim >= 4 and at least one step")
        self.node_encoder = nn.Sequential(nn.Linear(BUS_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.edge_encoder = nn.Sequential(nn.Linear(BRANCH_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.message = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.node_update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.control_encoder = nn.Sequential(nn.Linear(CONTROL_FEATURE_DIM, hidden_dim), nn.Tanh())
        self.control_update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.scenario_encoder = nn.Sequential(
            nn.Linear(SCENARIO_FEATURE_DIM, hidden_dim), nn.Tanh()
        )
        self.graph_fusion = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.Tanh())

    @staticmethod
    def _mean_rows(values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] == 0:
            return values.new_zeros(values.shape[1])
        return values.mean(dim=0)

    def prepare(self, state: TopologyGraphState) -> PreparedTopologyGraphState:
        """Validate once and upload one graph once for a resident multi-epoch update."""

        state.validate()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return PreparedTopologyGraphState(
            node_features=torch.as_tensor(state.node_features, device=device, dtype=dtype),
            edge_index=torch.as_tensor(state.edge_index, device=device, dtype=torch.long),
            edge_features=torch.as_tensor(state.edge_features, device=device, dtype=dtype),
            control_features=torch.as_tensor(state.control_features, device=device, dtype=dtype),
            control_bus_index=torch.as_tensor(
                state.control_bus_index, device=device, dtype=torch.long
            ),
            control_groups=torch.as_tensor(state.control_groups, device=device, dtype=torch.long),
            scenario_features=torch.as_tensor(state.scenario_features, device=device, dtype=dtype),
        )

    def forward_prepared(self, state: PreparedTopologyGraphState) -> TopologyEncoding:
        """Encode an already resident graph without another host conversion or validation sync."""

        node_features = state.node_features
        edge_index = state.edge_index
        edge_features = state.edge_features
        control_features = state.control_features
        control_bus_index = state.control_bus_index
        control_groups = state.control_groups
        scenario_features = state.scenario_features
        device = node_features.device
        dtype = node_features.dtype

        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        if edge_index.shape[1]:
            source, target = edge_index[0], edge_index[1]
            for _ in range(self.message_passing_steps):
                messages = self.message(torch.cat([nodes[source], edges], dim=-1))
                aggregate = torch.zeros_like(nodes)
                aggregate.index_add_(0, target, messages)
                degree = torch.zeros(nodes.shape[0], device=device, dtype=dtype)
                degree.index_add_(0, target, torch.ones(len(target), device=device, dtype=dtype))
                nodes = self.node_update(
                    torch.cat([nodes, aggregate / degree.clamp_min(1.0).unsqueeze(-1)], dim=-1)
                )

        scenarios = self.scenario_encoder(scenario_features)
        graph_embedding = self.graph_fusion(
            torch.cat(
                [self._mean_rows(nodes), self._mean_rows(edges), self._mean_rows(scenarios)],
                dim=-1,
            )
        )
        controls = self.control_encoder(control_features)
        if controls.shape[0]:
            attached = 0.5 * (nodes[control_bus_index[0]] + nodes[control_bus_index[1]])
            controls = self.control_update(torch.cat([controls, attached], dim=-1))
        groups = nodes.new_zeros((N_CONTROL_GROUPS, self.hidden_dim))
        counts = nodes.new_zeros(N_CONTROL_GROUPS)
        if controls.shape[0]:
            groups.index_add_(0, control_groups, controls)
            counts.index_add_(
                0,
                control_groups,
                torch.ones(len(control_groups), device=device, dtype=dtype),
            )
            groups = groups / counts.clamp_min(1.0).unsqueeze(-1)
        return TopologyEncoding(graph_embedding, groups, nodes, controls)

    def forward(self, state: TopologyGraphState) -> TopologyEncoding:
        return self.forward_prepared(self.prepare(state))
