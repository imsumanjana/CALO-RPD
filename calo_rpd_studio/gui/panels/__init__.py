"""GUI panel package with focused scientist-owned policy lifecycle extensions."""

# The established training and CALO Intelligence implementations remain the semantic authorities.
# Focused subclasses are substituted at their ordinary import points to add scientist-owned file
# management without duplicating training, qualification, or governing-policy logic.
from . import calo_intelligence_panel as _calo_intelligence_panel
from . import independent_training_panel as _independent_training_panel
from calo_rpd_studio.gui.widgets import context_pane as _context_pane

from .calo_intelligence_policy_controls import ScientistCALOIntelligencePanel
from .obsolete_model_management import (
    ObsoleteAwareTrainingModelLibrary,
    SavedTrainingManagementEditor,
)
from .calo_intelligence_obsolete_models import ObsoleteAwareCALOIntelligencePanel

_independent_training_panel.TrainingModelLibrary = ObsoleteAwareTrainingModelLibrary
_context_pane.TrainingPathEditor = SavedTrainingManagementEditor
_calo_intelligence_panel.CALOIntelligencePanel = ObsoleteAwareCALOIntelligencePanel

__all__ = [
    "ObsoleteAwareCALOIntelligencePanel",
    "ObsoleteAwareTrainingModelLibrary",
    "SavedTrainingManagementEditor",
    "ScientistCALOIntelligencePanel",
]
