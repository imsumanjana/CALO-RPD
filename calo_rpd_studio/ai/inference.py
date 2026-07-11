"""Convenience policy inference entry point."""
from calo_rpd_studio.algorithms.calo.ai_controller import AIController
def infer_policy(checkpoint,state_vector,seed=0,deterministic=True):return AIController(checkpoint,seed,deterministic).decide(state_vector)
