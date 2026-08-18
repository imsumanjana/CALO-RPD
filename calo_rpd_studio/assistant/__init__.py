"""Optional local research-assistant integration.

Assistant output is explanatory text only.  It is never scientific evidence and has no authority to
change experiments, training plans, policies, parameter studies, or stored results.
"""

from .ollama_parameter_advisor import (
    LocalAssistantConfig,
    LocalAssistantResponse,
    OllamaParameterAdvisor,
)

__all__ = ["LocalAssistantConfig", "LocalAssistantResponse", "OllamaParameterAdvisor"]
