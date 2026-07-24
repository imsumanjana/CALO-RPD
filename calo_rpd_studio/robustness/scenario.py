"""Immutable, validated scenario transformations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

from calo_rpd_studio.power_system.case_model import BUS_I, GEN_BUS, F_BUS, T_BUS


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    weight: float = 1.0
    transform: Callable | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Scenario name must be non-empty")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Scenario weight must be finite and non-negative")
        object.__setattr__(self, "weight", weight)
        if self.transform is not None and not callable(self.transform):
            raise TypeError("Scenario transform must be callable or None")

    def apply(self, case, *, copy_base: bool = True):
        source = case
        transformed = (
            source.clone() if copy_base else source
        ) if self.transform is None else self.transform(source.clone())
        if transformed is None:
            raise ValueError(f"Scenario {self.name!r} transform returned None")
        required = ("base_mva", "bus", "gen", "branch", "clone")
        if any(not hasattr(transformed, attr) for attr in required):
            raise TypeError(
                f"Scenario {self.name!r} transform must return a PowerSystemCase-compatible object"
            )
        if not math.isfinite(float(transformed.base_mva)) or float(transformed.base_mva) <= 0.0:
            raise ValueError(f"Scenario {self.name!r} produced invalid baseMVA")
        if not math.isclose(float(transformed.base_mva), float(source.base_mva), rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                f"Scenario {self.name!r} changed baseMVA; scenario transforms must preserve the scientific base"
            )
        for label in ("bus", "gen", "branch"):
            before = np.asarray(getattr(source, label), dtype=float)
            after = np.asarray(getattr(transformed, label), dtype=float)
            if after.shape != before.shape:
                raise ValueError(
                    f"Scenario {self.name!r} changed {label} matrix dimensions from {before.shape} to {after.shape}"
                )
            if not np.all(np.isfinite(after)):
                raise ValueError(f"Scenario {self.name!r} produced non-finite {label} data")
        # Row identity/topology must remain stable so one normalized ORPD variable manifest applies
        # to every scenario. Contingencies change status, not bus/gen/branch row identity.
        if not np.array_equal(transformed.bus[:, BUS_I].astype(int), source.bus[:, BUS_I].astype(int)):
            raise ValueError(f"Scenario {self.name!r} changed bus identities/order")
        if not np.array_equal(transformed.gen[:, GEN_BUS].astype(int), source.gen[:, GEN_BUS].astype(int)):
            raise ValueError(f"Scenario {self.name!r} changed generator-to-bus row identity")
        if not np.array_equal(
            transformed.branch[:, [F_BUS, T_BUS]].astype(int),
            source.branch[:, [F_BUS, T_BUS]].astype(int),
        ):
            raise ValueError(f"Scenario {self.name!r} changed branch endpoint row identity")
        return transformed
