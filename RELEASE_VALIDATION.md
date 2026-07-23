# CALO-RPD Studio v6.2.1 — Release Validation

**Release:** 6.2.1 — *Adaptive Compute Protection, Recovery and Scientific Qualification*  
**Date:** 23 July 2026

## Patch scope

v6.2.1 is a Dashboard presentation/usability patch on the v6.2.0 scientific baseline.

Implemented:
- vertically scrollable Dashboard body;
- three dedicated Dashboard tabs: **System Readiness**, **Training Queue**, **Scientific Context**;
- independent vertical scrolling within every tab;
- preserved native horizontal scrolling for the wide hardware/device table;
- larger usable System Readiness table area;
- summary metric cards arranged in a 3-column grid instead of one dense five-card row.

No policy ABI, CALO optimizer mathematics, Safe-80 governor semantics, ORPD formulation, experiment schema, workspace schema, or recovery authority was intentionally changed.

## Validation boundaries

- Python compile validation: required and executed before packaging.
- Dashboard structural regression tests: required and executed before packaging.
- Current release freeze and root package manifest: regenerated and independently verified before packaging.
- PyQt6 is unavailable in the build runtime, therefore a physical target-GUI rendering PASS is not claimed.

The target Windows machine should still perform the final visual check at the user's actual screen scaling/resolution.

## Executed patch regression evidence

- `compileall`: PASS for `calo_bootstrap`, `calo_rpd_studio`, and `tests`.
- Dashboard-layout structural regression: **4 passed**.
- Focused v6.2/v6.1/v5.9 architecture/scientific regression selection including the dashboard patch: **39 passed**.
- Current v6.2.1 freeze: **135 / 135 verified, 0 missing, 0 changed**.
- Physical PyQt6 rendering remains pending because PyQt6 is unavailable in the build runtime.
