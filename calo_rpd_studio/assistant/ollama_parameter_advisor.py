"""Localhost-only Ollama client for explaining retained parameter-study evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


LOCAL_ASSISTANT_RESPONSE_SCHEMA = "calo-local-parameter-assistant-response-v1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalAssistantConfig:
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:9b"
    timeout_seconds: float = 60.0

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("Local research assistant enabled state must be Boolean")
        parsed = urlparse(str(self.endpoint).strip())
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Local research assistant endpoint must use HTTP or HTTPS")
        host = (parsed.hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Research evidence may only be sent to a local Ollama server")
        if not str(self.model).strip():
            raise ValueError("Local research assistant requires a model name")
        if not 1.0 <= float(self.timeout_seconds) <= 300.0:
            raise ValueError("Local research assistant timeout must be between 1 and 300 seconds")


@dataclass(frozen=True, slots=True)
class LocalAssistantResponse:
    text: str
    model: str
    evidence_sha256: str
    response_schema: str = LOCAL_ASSISTANT_RESPONSE_SCHEMA
    scientific_authority: bool = False
    automatic_parameter_change: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.response_schema,
            "text": self.text,
            "model": self.model,
            "evidence_sha256": self.evidence_sha256,
            "scientific_authority": self.scientific_authority,
            "automatic_parameter_change": self.automatic_parameter_change,
        }


class OllamaParameterAdvisor:
    """Explain structured evidence without giving the model any mutation capability."""

    def __init__(self, config: LocalAssistantConfig) -> None:
        config.validate()
        self.config = config

    def explain(self, *, evidence: dict[str, Any], question: str) -> LocalAssistantResponse:
        if not self.config.enabled:
            raise RuntimeError("Local research assistant is disabled")
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError("Structured parameter evidence is required")
        prompt = str(question).strip()
        if not prompt:
            raise ValueError("A research question is required")
        evidence_sha256 = _canonical_sha256(evidence)
        system = (
            "You are the local CALO-RPD parameter research assistant. Explain only the supplied "
            "structured evidence. Distinguish observational association from controlled evidence. "
            "Never invent parameter values, statistics, cases, causal claims, or superiority claims. "
            "Never instruct CALO to mutate a policy, parameter, experiment, training plan, or result. "
            "If evidence is insufficient, say exactly what evidence is missing. Protected holdout "
            "cases cannot be used for parameter selection. Your text is explanatory and is not "
            "scientific evidence."
        )
        user = json.dumps(
            {"question": prompt, "evidence_sha256": evidence_sha256, "evidence": evidence},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        body = json.dumps(
            {
                "model": self.config.model,
                "stream": False,
                "keep_alive": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.1},
            }
        ).encode("utf-8")
        endpoint = self.config.endpoint.rstrip("/") + "/api/chat"
        request = Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=float(self.config.timeout_seconds)) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Local Ollama server returned an invalid response")
        message = dict(payload.get("message", {}) or {})
        text = str(message.get("content", "")).strip()
        if not text:
            raise RuntimeError("Local Ollama model returned no explanation")
        return LocalAssistantResponse(
            text=text,
            model=str(payload.get("model", self.config.model) or self.config.model),
            evidence_sha256=evidence_sha256,
        )
