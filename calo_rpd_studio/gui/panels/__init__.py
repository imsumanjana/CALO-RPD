"""GUI panel package with focused policy-control extensions."""

# The CALO Intelligence implementation is intentionally kept stable. Load the focused
# scientist-owned policy-control subclass here so ordinary imports of the established module
# receive the corrected lifecycle presentation without duplicating training/qualification code.
from . import calo_intelligence_panel as _calo_intelligence_panel
from .calo_intelligence_policy_controls import ScientistCALOIntelligencePanel

_calo_intelligence_panel.CALOIntelligencePanel = ScientistCALOIntelligencePanel

__all__ = ["ScientistCALOIntelligencePanel"]
