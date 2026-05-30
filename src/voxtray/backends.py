from __future__ import annotations

from .backend_contract import BackendCapabilities, TranscriptionBackend
from .cloud_backends import MistralRealtimeBackend, OpenAIRealtimeBackend
from .config import VoxtrayConfig
from .realtime import LocalVoxtralBackend


SUPPORTED_PROVIDERS = {
    "local_voxtral",
    "mistral_realtime",
    "openai_realtime",
}


def configured_provider(config: VoxtrayConfig) -> str:
    provider = str(getattr(getattr(config, "transcription", None), "provider", "") or "")
    return provider or "local_voxtral"


def create_transcription_backend(config: VoxtrayConfig) -> TranscriptionBackend:
    provider = configured_provider(config)
    if provider == "local_voxtral":
        return LocalVoxtralBackend(config)
    if provider == "mistral_realtime":
        return MistralRealtimeBackend(config)
    if provider == "openai_realtime":
        return OpenAIRealtimeBackend(config)
    raise ValueError(
        "unknown transcription provider "
        f"{provider!r}; expected one of {', '.join(sorted(SUPPORTED_PROVIDERS))}"
    )


def provider_api_key_env(config: VoxtrayConfig) -> str:
    provider = configured_provider(config)
    if provider == "mistral_realtime":
        return config.mistral_realtime.api_key_env
    if provider == "openai_realtime":
        return config.openai_realtime.api_key_env
    return ""


def local_engine_capabilities() -> BackendCapabilities:
    return LocalVoxtralBackend.capabilities
