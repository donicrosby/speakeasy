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


def test_voices_dir_set_uses_value_as_is(monkeypatch, clean_env):
    monkeypatch.setenv("SPEAKEASY_VOICES_DIR", "/custom")
    s = Settings()
    assert s.voices_dir == "/custom"
    # The set value is the voices root itself — no /voices suffix is appended.
    assert s.voices_dir_default() == "/custom"
