"""Speakeasy server configuration.

All settings are environment-driven with the SPEAKEASY_ prefix.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPEAKEASY_")

    # Whisper (primary engine)
    whisper_model: str = "large-v3-turbo"
    device: str = "auto"  # auto | cuda | cpu
    compute_type: str = "auto"  # auto | float16 | int8 | int8_float16

    # Moonshine (fast path for short audio)
    moonshine_enabled: bool = True
    moonshine_model: str = "UsefulSensors/moonshine-base"
    fast_path_max_seconds: float = 10.0  # audio shorter than this can use Moonshine

    # Transcription defaults
    beam_size: int = 5
    vad_filter: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # HuggingFace cache dir (models persist across restarts)
    hf_home: str | None = None

    # Auth — static bearer token, OpenAI-compatible. Empty = no auth (dev).
    api_key: str = ""


def resolve_device_compute(settings: Settings) -> tuple[str, str]:
    """Resolve auto device/compute_type into concrete values.

    device="cuda" is correct for BOTH NVIDIA CUDA and AMD ROCm — CTranslate2's
    ROCm wheel is HIP-backed but exposes the same "cuda" device API.
    """
    device = settings.device
    compute = settings.compute_type

    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    return device, compute
