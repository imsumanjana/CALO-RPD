# CALO-RPD v5.7 architecture boundaries

v5.7 reduces scientific-state coupling by enforcing service boundaries even though several legacy orchestration/UI modules remain large.

- **Scientific formulation:** `orpd/`, `power_system/`, `robustness/`; no GUI dependencies.
- **Optimization:** `algorithms/`; consumes a scientific problem and explicit budget/configuration.
- **Policy runtime/training:** `algorithms/calo/ai_controller.py`, `training.py`, `competitive_training.py`, `policy_registry.py`, `policy_qualification.py` with explicit artifact/checkpoint contracts.
- **Continuation:** `continuation/experiment_evolution.py`, `resume/`, CALO run/training checkpoint services; continuation semantics are distinct from publication evidence.
- **Persistence:** `results/database.py` and result-store interfaces own durable experiment/policy/revision records.
- **Restoration:** `app/experiment_workspace_restorer.py` resolves authoritative revision/config/PF/policy state before UI hydration.
- **Publication evidence:** `portfolio/` and `results/publication_export.py`; verified-only fail-closed filters are applied before publication artifacts/statistics.
- **GUI:** panels present/edit state and call services; scientific definitions are not supposed to be reimplemented inside widgets.

Large legacy modules are still candidates for further mechanical decomposition, but v5.7 freezes the contracts above so future splitting can be performed without changing mathematical behavior. Structural module size is treated as maintainability debt, not as scientific closure evidence.
