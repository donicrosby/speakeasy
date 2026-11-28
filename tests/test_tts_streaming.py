"""Streaming speech — the engine half.

The engine's streaming methods are unit-tested here against fake libraries,
the way `test_voxcpm2_engine.py` fakes `voxcpm` and `test_tts.py` fakes
`reverb_tts`: no GPU, no model download, and the suite passes in CI's
no-extras job. The route and the page are added by later slices.

The fakes grow the streaming entry points of the libraries they stand in for,
so the route can be exercised without a GPU.
"""

import sys
import threading
import types

import numpy as np
import pytest
import torch

from speakeasy import engines

SAMPLE_RATE = 48000
PATCH_SAMPLES = int(SAMPLE_RATE * 0.16)  # one audio patch, 7680 samples


class FakeTtsModel:
    """The `tts_model` behind a fake `VoxCPM`: a sample rate and a tokenizer.

    The tokenizer returns one token per character, so a test can pick a token
    count by picking the text.
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE

        def tokenize(text):
            return list(text)

        self.text_tokenizer = tokenize


class FakeVoxCPM:
    """Stand-in for `voxcpm.VoxCPM`; records what the engine does to it."""

    waveform = np.zeros(PATCH_SAMPLES, dtype=np.float32)
    generate_calls: list = []
    stream_calls: list = []

    def __init__(self, *args, **kwargs):
        self.tts_model = FakeTtsModel()

    @classmethod
    def reset(cls):
        cls.waveform = np.zeros(PATCH_SAMPLES, dtype=np.float32)
        cls.generate_calls = []
        cls.stream_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def generate(self, *args, **kwargs):
        FakeVoxCPM.generate_calls.append({"args": args, "kwargs": kwargs})
        return FakeVoxCPM.waveform

    def generate_streaming(self, *args, **kwargs):
        """The library's streaming twin: the same take, one patch per chunk."""
        FakeVoxCPM.stream_calls.append({"args": args, "kwargs": kwargs})
        waveform = FakeVoxCPM.waveform
        for start in range(0, len(waveform), PATCH_SAMPLES):
            yield waveform[start : start + PATCH_SAMPLES]


@pytest.fixture()
def fake_voxcpm(monkeypatch):
    """Install a fake `voxcpm` in sys.modules and reset its recorders."""
    mod = types.ModuleType("voxcpm")
    mod.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", mod)
    FakeVoxCPM.reset()
    return mod


class FakeVoice:
    def __init__(self, language="en"):
        self.language = language
        self.text = ""


class FakeTTS:
    """Stand-in for `reverb_tts.TTS`; records what the engine does to it."""

    calls: list = []

    def __init__(self, name, **kw):
        self.sample_rate = 24000
        self._presets = {
            "en_male_narrator_1": FakeVoice(),
            "de_female_standard": FakeVoice("de"),
        }

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def from_pretrained(cls, name, **kw):
        return cls(name, **kw)

    def preset_voices(self):
        return sorted(self._presets)

    def preset_voice(self, name):
        return self._presets[name]

    def generate(self, text, voice, language, *, seed=None, speed=1.0):
        FakeTTS.calls.append(("synth", text, voice.language, language, seed, speed))
        return np.zeros(24000, dtype=np.float32)


@pytest.fixture()
def fake_reverb(monkeypatch):
    """Install a fake `reverb_tts` in sys.modules and reset its recorders."""
    mod = types.ModuleType("reverb_tts")
    mod.TTS = FakeTTS
    mod.LANGUAGES = {"en": "English", "de": "German"}
    monkeypatch.setitem(sys.modules, "reverb_tts", mod)
    FakeTTS.reset()
    return mod


class LockSpy:
    """A reentrant lock that reports how many times it is currently held."""

    def __init__(self):
        self._lock = threading.RLock()
        self.held = 0

    def __enter__(self):
        self._lock.acquire()
        self.held += 1
        return self

    def __exit__(self, *exc):
        self.held -= 1
        self._lock.release()
        return False


# --- VoxCPM2 ------------------------------------------------------------


def test_voxcpm2_generate_streaming_chunk_shape(fake_voxcpm):
    """The stream is the library's chunks, in order, as float32, one per patch."""
    FakeVoxCPM.waveform = np.zeros(3 * PATCH_SAMPLES, dtype=np.float32)
    e = engines.VoxCPM2Engine("m", "cpu")

    chunks = list(e.generate_streaming("abc"))

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.dtype == np.float32
        assert len(chunk) == PATCH_SAMPLES

    call = FakeVoxCPM.stream_calls[-1]
    assert call["args"] == ()
    assert call["kwargs"]["text"] == "abc"
    assert call["kwargs"]["cfg_value"] == 2.0
    assert call["kwargs"]["inference_timesteps"] == 10
    # The streaming path never falls back to the whole-take call.
    assert FakeVoxCPM.generate_calls == []


def test_voxcpm2_streaming_matches_synthesize(fake_voxcpm):
    """The concatenation of a stream is the take synthesize returns, exactly."""
    rng = np.random.default_rng(7)
    FakeVoxCPM.waveform = rng.standard_normal(5 * PATCH_SAMPLES).astype(np.float32)
    e = engines.VoxCPM2Engine("m", "cpu")

    synthesized, _ = e.synthesize("abcdef")
    streamed = np.concatenate(list(e.generate_streaming("abcdef")))
    named = np.concatenate(list(e.synthesize_streaming("abcdef")))

    assert np.array_equal(streamed, synthesized)
    assert np.array_equal(named, synthesized)


def test_voxcpm2_streaming_keeps_reference_and_prompt(fake_voxcpm):
    """The streaming call carries the same reference/prompt arguments as synthesize."""
    e = engines.VoxCPM2Engine("m", "cpu")

    list(
        e.generate_streaming(
            "Hi", reference="/voices/voxcpm2/clone-a.wav", prompt_text="words"
        )
    )

    kw = FakeVoxCPM.stream_calls[-1]["kwargs"]
    assert kw["reference_wav_path"] == "/voices/voxcpm2/clone-a.wav"
    assert kw["prompt_wav_path"] == "/voices/voxcpm2/clone-a.wav"
    assert kw["prompt_text"] == "words"


def test_voxcpm2_streaming_seeds_under_the_lock(fake_voxcpm, monkeypatch):
    """A seed reaches torch.manual_seed under the lock, before the library runs."""
    e = engines.VoxCPM2Engine("m", "cpu")
    lock = LockSpy()
    e._lock = lock

    timeline = []
    real_seed = torch.manual_seed

    def recording_seed(seed):
        timeline.append(("seed", seed, lock.held))
        return real_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", recording_seed)

    def recording_stream(*args, **kwargs):
        timeline.append(("stream", lock.held))
        yield np.zeros(PATCH_SAMPLES, dtype=np.float32)

    e._model.generate_streaming = recording_stream

    list(e.generate_streaming("Hi", seed=42))

    assert timeline == [("seed", 42, 1), ("stream", 1)]
    # The lock is released once the generator is exhausted.
    assert lock.held == 0


def test_voxcpm2_streaming_refuses_a_bad_take(fake_voxcpm):
    """The bad-take bound still applies once the take is complete."""
    e = engines.VoxCPM2Engine("m", "cpu")
    text = "abcd"  # the fake tokenizer counts four tokens

    FakeVoxCPM.waveform = np.zeros(23 * PATCH_SAMPLES, dtype=np.float32)
    assert len(list(e.generate_streaming(text))) == 23

    FakeVoxCPM.waveform = np.zeros(24 * PATCH_SAMPLES, dtype=np.float32)
    with pytest.raises(engines.BadTake):
        list(e.generate_streaming(text))


# --- Reverb -------------------------------------------------------------


def test_reverb_synthesize_streaming_single_chunk(fake_reverb):
    """Reverb answers a streaming request with the finished take, one chunk."""
    e = engines.ReverbEngine("m", "int8", "cpu")
    voice = e.voice("en_male_narrator_1")

    chunks = list(e.synthesize_streaming("Hi.", voice, "en", seed=3, speed=1.25))

    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert len(chunks[0]) == 24000
    # The same arguments reach the engine as a non-streaming take.
    assert FakeTTS.calls[-1] == ("synth", "Hi.", "en", "en", 3, 1.25)

    audio, _ = e.synthesize("Hi.", voice, "en", seed=3, speed=1.25)
    assert np.array_equal(chunks[0], audio)
