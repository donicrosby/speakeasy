"""VoxCPM2Engine unit tests — the engine half of the VoxCPM2 suite.

The real `voxcpm` package is never imported: it is faked through
`sys.modules`, so the suite passes in CI's no-extras job where the
`tts-voxcpm` extra is not installed. Server/endpoint VoxCPM2 scenarios are
added by later slices.
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
    count by picking the text; it also records the text it was handed, so a
    test can prove the engine tokenises the flattened text.
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.tokenizer_inputs = []

        def tokenize(text):
            self.tokenizer_inputs.append(text)
            return list(text)

        self.text_tokenizer = tokenize


class FakeVoxCPM:
    """Stand-in for `voxcpm.VoxCPM`; records what the engine does to it."""

    init_args: tuple = ()
    init_kwargs: dict = {}
    instances: list = []
    waveform = np.zeros(PATCH_SAMPLES, dtype=np.float32)
    generate_calls: list = []

    def __init__(self, *args, **kwargs):
        self.tts_model = FakeTtsModel()

    @classmethod
    def reset(cls):
        cls.init_args = ()
        cls.init_kwargs = {}
        cls.instances = []
        cls.waveform = np.zeros(PATCH_SAMPLES, dtype=np.float32)
        cls.generate_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.init_args = args
        cls.init_kwargs = kwargs
        obj = cls(*args, **kwargs)
        cls.instances.append(obj)
        return obj

    def generate(self, *args, **kwargs):
        FakeVoxCPM.generate_calls.append({"args": args, "kwargs": kwargs})
        return FakeVoxCPM.waveform


@pytest.fixture()
def fake_voxcpm(monkeypatch):
    """Install a fake `voxcpm` in sys.modules and reset its recorders."""
    mod = types.ModuleType("voxcpm")
    mod.VoxCPM = FakeVoxCPM
    monkeypatch.setitem(sys.modules, "voxcpm", mod)
    FakeVoxCPM.reset()
    return mod


def test_engines_do_not_import_voxcpm_at_import_time():
    """src/ must import voxcpm lazily — CI has no `tts-voxcpm` extra installed."""
    assert "voxcpm" not in sys.modules


def test_voxcpm2_engine_constructor_args(fake_voxcpm):
    """from_pretrained gets the fixed loader arguments; the rate comes back."""
    e = engines.VoxCPM2Engine("openbmb/VoxCPM2", "cuda")

    assert e.name == "voxcpm2"
    assert e.sample_rate == SAMPLE_RATE
    assert e.load_seconds >= 0
    assert FakeVoxCPM.init_args == ("openbmb/VoxCPM2",)
    assert FakeVoxCPM.init_kwargs == {
        "load_denoiser": False,
        "optimize": False,
        "device": "cuda",
    }


def test_voxcpm2_engine_device_auto_passes_none(fake_voxcpm):
    """`device="auto"` reaches the library as None — its own auto-selection."""
    engines.VoxCPM2Engine("m", "auto")
    assert FakeVoxCPM.init_kwargs["device"] is None


def test_voxcpm2_engine_generates(fake_voxcpm):
    """Every library argument the scope fixes reaches one `generate` call."""
    e = engines.VoxCPM2Engine("m", "cpu")

    audio, elapsed = e.synthesize("Hi there")

    assert audio.dtype == np.float32
    assert len(audio) == PATCH_SAMPLES
    assert elapsed >= 0
    assert len(FakeVoxCPM.generate_calls) == 1
    call = FakeVoxCPM.generate_calls[-1]
    assert call["args"] == ()
    assert call["kwargs"] == {
        "text": "Hi there",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "min_len": 2,
        "max_len": 4096,
        "normalize": False,
        "denoise": False,
        "retry_badcase": False,
    }


def test_voxcpm2_engine_cfg_reaches_library(fake_voxcpm):
    e = engines.VoxCPM2Engine("m", "cpu")
    e.synthesize("Hi", cfg=3.5)
    assert FakeVoxCPM.generate_calls[-1]["kwargs"]["cfg_value"] == 3.5


def test_voxcpm2_engine_reference_with_transcript(fake_voxcpm):
    """A known transcript goes in as both reference and continuation audio."""
    e = engines.VoxCPM2Engine("m", "cpu")

    e.synthesize("Hi", reference="/voices/voxcpm2/clone-a.wav", prompt_text="words")

    kw = FakeVoxCPM.generate_calls[-1]["kwargs"]
    assert kw["prompt_wav_path"] == "/voices/voxcpm2/clone-a.wav"
    assert kw["prompt_text"] == "words"
    assert kw["reference_wav_path"] == "/voices/voxcpm2/clone-a.wav"


def test_voxcpm2_engine_never_prompts_without_prompt_text(fake_voxcpm):
    """The library raises on an unpaired prompt, so no mode may send one."""
    e = engines.VoxCPM2Engine("m", "cpu")

    e.synthesize("Hi")
    e.synthesize("Hi", reference="/voices/voxcpm2/clone-a.wav")
    e.synthesize("Hi", reference="/voices/voxcpm2/clone-a.wav", prompt_text="words")

    assert len(FakeVoxCPM.generate_calls) == 3
    for call in FakeVoxCPM.generate_calls:
        kwargs = call["kwargs"]
        assert ("prompt_wav_path" in kwargs) == ("prompt_text" in kwargs)


def test_voxcpm2_engine_reference_without_transcript(fake_voxcpm):
    """No transcript: the same path goes in as reference_wav_path instead."""
    e = engines.VoxCPM2Engine("m", "cpu")

    e.synthesize("Hi", reference="/voices/voxcpm2/clone-a.wav")

    kw = FakeVoxCPM.generate_calls[-1]["kwargs"]
    assert kw["reference_wav_path"] == "/voices/voxcpm2/clone-a.wav"
    assert "prompt_wav_path" not in kw
    assert "prompt_text" not in kw


def test_voxcpm2_engine_plain_has_no_reference(fake_voxcpm):
    e = engines.VoxCPM2Engine("m", "cpu")
    e.synthesize("Hi")
    kw = FakeVoxCPM.generate_calls[-1]["kwargs"]
    for key in ("prompt_wav_path", "prompt_text", "reference_wav_path"):
        assert key not in kw


def test_voxcpm2_engine_bad_take_boundary(fake_voxcpm):
    """Six audio patches per text token is the boundary; one below passes."""
    e = engines.VoxCPM2Engine("m", "cpu")
    text = "abcd"  # the fake tokenizer counts four tokens

    FakeVoxCPM.waveform = np.zeros(23 * PATCH_SAMPLES, dtype=np.float32)
    audio, _ = e.synthesize(text)
    assert len(audio) == 23 * PATCH_SAMPLES

    FakeVoxCPM.waveform = np.zeros(24 * PATCH_SAMPLES, dtype=np.float32)
    with pytest.raises(engines.BadTake):
        e.synthesize(text)


def test_voxcpm2_engine_counts_tokens_over_flattened_text(fake_voxcpm):
    """The token count follows the library's own whitespace flattening."""
    e = engines.VoxCPM2Engine("m", "cpu")
    e.synthesize("a\n  b\tc")
    model = FakeVoxCPM.instances[-1].tts_model
    assert model.tokenizer_inputs[-1] == "a b c"


def test_voxcpm2_engine_seeds(fake_voxcpm, monkeypatch):
    """With a seed, torch.manual_seed runs under the lock, before generate."""
    e = engines.VoxCPM2Engine("m", "cpu")
    lock = LockSpy()
    e._lock = lock

    timeline = []
    real_seed = torch.manual_seed

    def recording_seed(seed):
        timeline.append(("seed", seed, lock.held))
        return real_seed(seed)

    monkeypatch.setattr(torch, "manual_seed", recording_seed)

    def recording_generate(*args, **kwargs):
        timeline.append(("generate", lock.held))
        return np.zeros(PATCH_SAMPLES, dtype=np.float32)

    e._model.generate = recording_generate

    e.synthesize("Hi", seed=42)
    assert timeline == [("seed", 42, 1), ("generate", 1)]

    # No seed: torch.manual_seed never runs.
    timeline.clear()
    e.synthesize("Hi")
    assert timeline == [("generate", 1)]


def test_voxcpm2_cudnn_disabled_on_hip(fake_voxcpm, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    # HIP build: MIOpen's default find mode is unusably slow, so it is disabled.
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    engines.VoxCPM2Engine("m", "cpu")
    assert torch.backends.cudnn.enabled is False

    # Non-HIP (CPU/CUDA): the flag is left exactly as it was.
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.version, "hip", None)
    engines.VoxCPM2Engine("m", "cpu")
    assert torch.backends.cudnn.enabled is True


def test_voxcpm2_engine_unload(fake_voxcpm, monkeypatch):
    """unload() drops the model, empties the cache and collects."""
    e = engines.VoxCPM2Engine("m", "cpu")
    assert e._model is not None

    emptied = []
    collected = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    monkeypatch.setattr(engines.gc, "collect", lambda: collected.append(True))

    e.unload()

    assert e._model is None
    assert emptied == [True]
    assert collected == [True]


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
