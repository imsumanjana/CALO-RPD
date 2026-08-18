import pytest

from calo_rpd_studio.assistant import LocalAssistantConfig, LocalAssistantResponse


def test_local_assistant_refuses_nonlocal_endpoint() -> None:
    with pytest.raises(ValueError, match="local Ollama"):
        LocalAssistantConfig(enabled=True, endpoint="https://example.com", model="qwen").validate()


def test_local_assistant_response_cannot_claim_scientific_authority() -> None:
    response = LocalAssistantResponse(
        text="Observed association only.",
        model="qwen3.5:9b",
        evidence_sha256="a" * 64,
    ).to_dict()
    assert response["scientific_authority"] is False
    assert response["automatic_parameter_change"] is False
