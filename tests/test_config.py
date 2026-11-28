"""TTS settings and the voices-dir default.

Env-driven settings; each test clears the vars it asserts on so a polluted
shell cannot change the result.
"""

import pytest

from speakeasy.config import Settings

TTS_ENV_VARS = (
    "SPEAKEASY_TTS_ENABLED",
    "SPEAKEASY_TTS_MODEL",
    "SPEAKEASY_TTS_VARIANT",
    "SPEAKEASY_TTS_DEVICE",
    "SPEAKEASY_TTS_VOXCPM_ENABLED",
    "SPEAKEASY_TTS_VOXCPM_MODEL",
    "SPEAKEASY_TTS_VOXCPM_IDLE_SECONDS",
    "SPEAKEASY_VOICES_DIR",
)


@pytest.fixture()
def clean_env(monkeypatch):
    for var in TTS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_tts_defaults(clean_env):
    s = Settings()
    assert s.tts_enabled is True
    assert s.tts_model == "software-mansion/Reverb-TTS"
    assert s.tts_variant == "int8"
    assert s.tts_device == "auto"
    assert s.voices_dir is None


def test_voices_dir_follows_hf_home(clean_env):
    s = Settings(hf_home="/models")
    assert s.voices_dir_default() == "/models/voices"


def test_voxcpm2_defaults(clean_env):
    s = Settings()
    assert s.tts_voxcpm_enabled is True
    assert s.tts_voxcpm_model == "openbmb/VoxCPM2"
    assert s.tts_voxcpm_idle_seconds == 600


def test_voices_dir_set_uses_value_as_is(monkeypatch, clean_env):
    monkeypatch.setenv("SPEAKEASY_VOICES_DIR", "/custom")
    s = Settings()
    assert s.voices_dir == "/custom"
    # The set value is the voices root itself — no /voices suffix is appended.
    assert s.voices_dir_default() == "/custom"


def test_config_field_names(clean_env):
    # Exactly these settings exist, and nothing else — removing the duration
    # auto-route dropped its knob and left every other name and default alone.
    names = set(Settings.model_fields)
    assert "fast_path_max_seconds" not in names
    assert names == {
        "api_key",
        "beam_size",
        "bind",
        "compute_type",
        "device",
        "hf_home",
        "moonshine_enabled",
        "moonshine_model",
        "port",
        "tts_device",
        "tts_enabled",
        "tts_model",
        "tts_variant",
        "tts_voxcpm_enabled",
        "tts_voxcpm_idle_seconds",
        "tts_voxcpm_model",
        "vad_filter",
        "voices_dir",
        "whisper_model",
    }
    s = Settings()
    assert s.whisper_model == "large-v3-turbo"
    assert s.moonshine_enabled is True
    assert s.beam_size == 5
    assert s.vad_filter is True
