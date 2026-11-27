"""ReverbEngine unit tests — the engine half of the TTS suite.

The real `reverb_tts` package is never imported: it is faked through
`sys.modules`, so the suite passes in CI's no-extras job where the `tts`
extra is not installed. Server/endpoint TTS scenarios are added by later
slices.
"""

import sys
import threading
import time
import types

import numpy as np
import pytest
import torch

from speakeasy import engines


class FakeVoice:
    def __init__(self, language="en"):
        self.language = language
        self.text = ""
        self.saved_to = None

    def save(self, path):
        self.saved_to = str(path)


class FakeTTS:
    """Stand-in for `reverb_tts.TTS`; records what the engine does to it."""

    calls: list = []
    preset_voice_calls: list = []
    init_kwargs: dict = {}
    delay = 0.0
    in_flight = 0
    max_in_flight = 0
    _guard = threading.Lock()

    def __init__(self, name, **kw):
        FakeTTS.init_kwargs = {"name": name, **kw}
        self.sample_rate = 24000
        self._presets = {
            "en_male_narrator_1": FakeVoice(),
            "de_female_standard": FakeVoice("de"),
        }

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.preset_voice_calls = []
        cls.init_kwargs = {}
        cls.delay = 0.0
        cls.in_flight = 0
        cls.max_in_flight = 0

    @classmethod
    def from_pretrained(cls, name, **kw):
        return cls(name, **kw)

    @classmethod
    def _enter(cls):
        with cls._guard:
            cls.in_flight += 1
            cls.max_in_flight = max(cls.max_in_flight, cls.in_flight)

    @classmethod
    def _leave(cls):
        with cls._guard:
            cls.in_flight -= 1

    def preset_voices(self):
        return sorted(self._presets)

    def preset_voice(self, name):
        FakeTTS.preset_voice_calls.append(name)
        try:
            return self._presets[name]
        except KeyError:
            raise ValueError(f"unknown preset voice {name!r}") from None

    def voice(self, samples, text, language, *, sample_rate=None):
        FakeTTS.calls.append(("clone", samples.shape, sample_rate, text, language))
        FakeTTS._enter()
        try:
            time.sleep(FakeTTS.delay)
            return FakeVoice(language)
        finally:
            FakeTTS._leave()

    def generate(self, text, voice, language, *, seed=None, speed=1.0):
        FakeTTS.calls.append(("synth", text, voice.language, language, seed, speed))
        FakeTTS._enter()
        try:
            time.sleep(FakeTTS.delay)
            return np.zeros(24000, dtype=np.float32)
        finally:
            FakeTTS._leave()


@pytest.fixture()
def fake_reverb(monkeypatch):
    """Install a fake `reverb_tts` in sys.modules and reset its recorders."""
    mod = types.ModuleType("reverb_tts")
    mod.TTS = FakeTTS
    mod.LANGUAGES = {"en": "English", "de": "German"}
    monkeypatch.setitem(sys.modules, "reverb_tts", mod)
    FakeTTS.reset()
    return mod


def test_engines_do_not_import_reverb_tts_at_import_time():
    """src/ must import reverb_tts lazily — CI has no `tts` extra installed."""
    assert "reverb_tts" not in sys.modules


def test_engine_loads_presets_and_speaks(fake_reverb):
    e = engines.ReverbEngine("software-mansion/Reverb-TTS", "int8", "cuda")
    assert e.name == "reverb-tts"
    assert e.sample_rate == 24000
    assert e.languages == ["de", "en"]
    assert e.load_seconds >= 0
    assert FakeTTS.init_kwargs == {
        "name": "software-mansion/Reverb-TTS",
        "variant": "int8",
        "device": "cuda",
    }
    assert [p["id"] for p in e.presets()] == [
        "de_female_standard",
        "en_male_narrator_1",
    ]
    assert e.presets()[0] == {
        "id": "de_female_standard",
        "language": "de",
        "transcript": "",
    }

    audio, elapsed = e.synthesize(
        "Hi.", e.voice("en_male_narrator_1"), "en", seed=3, speed=1.25
    )
    assert audio.dtype == np.float32
    assert len(audio) == 24000
    assert elapsed >= 0
    assert FakeTTS.calls[-1] == ("synth", "Hi.", "en", "en", 3, 1.25)


def test_engine_languages_come_from_module(fake_reverb):
    """languages is read from reverb_tts.LANGUAGES, never a hardcoded list."""
    fake_reverb.LANGUAGES = {"en": "English", "de": "German", "xx": "Testish"}
    e = engines.ReverbEngine("m", "int8", "cpu")
    assert e.languages == ["de", "en", "xx"]


def test_engine_clones_from_samples(fake_reverb):
    e = engines.ReverbEngine("m", "int8", "cpu")
    v = e.clone(np.zeros(24000, np.float32), 24000, "exact words", "de")
    assert v.language == "de"
    assert FakeTTS.calls[0] == ("clone", (24000,), 24000, "exact words", "de")


def test_presets_cached_at_construction(fake_reverb):
    e = engines.ReverbEngine("m", "int8", "cpu")
    # Every preset the fake module offers is fetched once, during __init__.
    assert sorted(FakeTTS.preset_voice_calls) == [
        "de_female_standard",
        "en_male_narrator_1",
    ]

    FakeTTS.preset_voice_calls.clear()
    first = e.voice("en_male_narrator_1")
    assert e.voice("en_male_narrator_1") is first
    # Served from the cache: no preset_voice() call, so no network on the request path.
    assert FakeTTS.preset_voice_calls == []


def test_engine_synthesize_lock_no_overlap(fake_reverb):
    """Two threads calling synthesize never overlap inside the model."""
    e = engines.ReverbEngine("m", "int8", "cpu")
    voice = e.voice("en_male_narrator_1")
    FakeTTS.delay = 0.05

    threads = [
        threading.Thread(target=e.synthesize, args=(f"line {i}", voice, "en"))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert FakeTTS.max_in_flight == 1


def test_engine_lock_reentrant_for_clone(fake_reverb):
    """The clone-create path holds the engine lock, then calls clone.

    A plain Lock would self-deadlock here and hang the request; the engine must
    use a reentrant lock.
    """
    e = engines.ReverbEngine("m", "int8", "cpu")
    result = []

    def hold_lock_then_clone():
        with e._lock:
            result.append(e.clone(np.zeros(24000, np.float32), 24000, "words", "en"))

    t = threading.Thread(target=hold_lock_then_clone)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "clone blocked on the engine lock — it is not reentrant"
    assert result and result[0].language == "en"


def test_cudnn_disabled_only_on_hip(fake_reverb, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    # HIP build: MIOpen's default find mode is unusably slow, so it is disabled.
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    engines.ReverbEngine("m", "int8", "cpu")
    assert torch.backends.cudnn.enabled is False

    # Non-HIP (CPU/CUDA): the flag is left exactly as it was.
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.version, "hip", None)
    engines.ReverbEngine("m", "int8", "cpu")
    assert torch.backends.cudnn.enabled is True
