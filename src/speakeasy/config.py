"""Speakeasy server configuration.

All settings are environment-driven with the SPEAKEASY_ prefix.
"""

import os

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
    bind: str = "0.0.0.0"  # SPEAKEASY_BIND — address uvicorn listens on
    port: int = 8080

    # HuggingFace cache dir (models persist across restarts)
    hf_home: str | None = None

    # TTS — Reverb voice cloning. A no-op unless the `tts` extra is installed.
    tts_enabled: bool = True
    tts_model: str = "software-mansion/Reverb-TTS"
    tts_variant: str = (
        "int8"  # fp32 | fp16 | int8 | int4: download size only, same speed
    )
    tts_device: str = "auto"  # auto | cuda | cpu

    # Saved voice clones. When set, the value IS the voices root (no /voices
    # suffix is appended); otherwise <hf_home or ~/.cache/speakeasy>/voices.
    voices_dir: str | None = None

    def voices_dir_default(self) -> str:
        if self.voices_dir:
            return self.voices_dir
        base = self.hf_home or os.path.join(
            os.path.expanduser("~"), ".cache", "speakeasy"
        )
        return os.path.join(base, "voices")

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
