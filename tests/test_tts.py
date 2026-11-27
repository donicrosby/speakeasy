"""ReverbEngine unit tests — the engine half of the TTS suite.

The real `reverb_tts` package is never imported: it is faked through
`sys.modules`, so the suite passes in CI's no-extras job where the `tts`
extra is not installed. Server/endpoint TTS scenarios are added by later
slices.
"""

import io
import sys
import threading
import time
import types
import wave

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import speakeasy.server as server
from speakeasy import engines, voices
from speakeasy.engines import Segment, TranscriptionResult


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


# ---------------------------------------------------------------------------
# Slice 3 — background TTS load and voice list (§8.4 lifespan/state, §8.5 list)
# ---------------------------------------------------------------------------


class ServerVoice:
    """Mirrors reverb_tts.Voice's `.npz` format without the package."""

    def __init__(self, language="en", text="ref words"):
        self.language, self.text = language, text

    def save(self, path):
        np.savez(
            path,
            codes=np.zeros((8, 250), np.int16),
            text=self.text,
            language=self.language,
        )


class StubWhisper:
    """Just enough whisper to make `/health/ready` report 200."""

    load_seconds = 0.0

    @property
    def name(self):
        return "whisper"

    def transcribe(self, path, **_: object):
        return TranscriptionResult(
            text="stub",
            language="en",
            duration=1.0,
            segments=[Segment(start=0.0, end=1.0, text="stub")],
            engine="whisper",
            elapsed_seconds=0.0,
        )


class RaisingEngine:
    """A TTS engine whose listing blows up — exercises the SEC-16 500 leg."""

    languages = ["en"]

    def presets(self):
        raise RuntimeError("preset listing exploded: /models/weights/secret")

    def voice(self, preset_id):
        raise ValueError("nope")


def _wav_bytes(seconds=0.5, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(sr * seconds))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def tts_state_reset(monkeypatch):
    """Restore the TTS module globals around every test in this module."""
    monkeypatch.setattr(server, "tts", None)
    monkeypatch.setattr(server, "voices_store", None)
    monkeypatch.setattr(server, "tts_state", "disabled")
    monkeypatch.setattr(server, "tts_error", None)
    yield


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "whisper", StubWhisper())
    monkeypatch.setattr(server, "moonshine", None)
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server.settings, "api_key", "")
    return TestClient(server.app)


# --- §8.4 Lifespan and state ------------------------------------------------


def test_ready_reports_tts_state(client, monkeypatch):
    monkeypatch.setattr(server, "tts_state", "disabled")
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["tts"] == "disabled"


def test_ready_reports_error_state(client, monkeypatch):
    monkeypatch.setattr(server, "tts_state", "error")
    monkeypatch.setattr(server, "tts_error", "reverb-tts failed to load: boom")

    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["tts"] == "error"

    # A TTS endpoint answers with the fixed error message — never the detail.
    v = client.get("/v1/audio/voices")
    assert v.status_code == 503
    assert v.json() == {
        "error": {"message": "tts encountered an error and is unavailable"}
    }


def test_models_lists_reverb_tts(client, fake_reverb, monkeypatch):
    monkeypatch.setattr(server, "tts", engines.ReverbEngine("m", "int8", "cpu"))
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert "reverb-tts" in [m["id"] for m in r.json()["data"]]


def test_models_omits_tts_when_none(client):
    assert "reverb-tts" not in [
        m["id"] for m in client.get("/v1/models").json()["data"]
    ]


def test_ready_reports_loading_while_tts_boots(client, fake_reverb, monkeypatch):
    """Readiness answers from Whisper's state; it never waits on the TTS thread."""
    started = threading.Event()
    release = threading.Event()

    def blocking_load():
        started.set()
        release.wait(10)

    monkeypatch.setattr(server, "_load_tts", blocking_load)
    server.start_tts()
    assert started.wait(5), "the tts-load thread did not start"

    try:
        assert server.tts_state == "loading"
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["tts"] == "loading"
    finally:
        release.set()


def test_tts_disabled_on_import_error(client, monkeypatch):
    """`reverb_tts` absent → disabled, no thread, transcription still served."""
    monkeypatch.setitem(sys.modules, "reverb_tts", None)  # `import reverb_tts` raises

    # Readiness first, before threading.Thread is under a spy.
    assert client.get("/health/ready").json()["tts"] == "disabled"

    started = []

    class SpyingThread:
        def __init__(self, *a, **k):
            started.append(a)
            raise AssertionError("start_tts must not spawn a thread without reverb_tts")

    monkeypatch.setattr(server.threading, "Thread", SpyingThread)
    server.start_tts()

    assert started == []
    assert server.tts_state == "disabled"
    assert server.tts is None

    files = {"file": ("tone.wav", _wav_bytes(), "audio/wav")}
    assert client.post("/v1/audio/transcriptions", files=files).status_code == 200


def test_no_exception_text_in_error_bodies(client, fake_reverb, monkeypatch):
    """SEC-16: no exception text in any response body (slice-3 reachable paths)."""
    secret = "reverb-tts failed to load: /models/weights/leaky RuntimeError boom"

    # 503 leg — error state.
    monkeypatch.setattr(server, "tts_state", "error")
    monkeypatch.setattr(server, "tts_error", secret)
    r = client.get("/v1/audio/voices")
    assert r.status_code == 503
    for leak in ("/models", "RuntimeError", "boom", "Traceback"):
        assert leak not in r.text

    # 500 leg — a handler that raises.
    monkeypatch.setattr(server, "tts_state", "ready")
    monkeypatch.setattr(server, "tts", RaisingEngine())
    raw = TestClient(server.app, raise_server_exceptions=False)
    r = raw.get("/v1/audio/voices")
    assert r.status_code == 500
    for leak in ("/models", "RuntimeError", "boom", "Traceback"):
        assert leak not in r.text


def test_load_tts_publishes_engine_store_and_state(fake_reverb, monkeypatch, tmp_path):
    monkeypatch.setattr(server.settings, "voices_dir", str(tmp_path / "voices"))
    server._load_tts()
    assert server.tts is not None
    assert server.voices_store is not None
    assert server.tts_state == "ready"
    assert server.tts_error is None


def test_load_tts_failure_sets_error_state(fake_reverb, monkeypatch, tmp_path):
    monkeypatch.setattr(server.settings, "voices_dir", str(tmp_path / "voices"))

    def boom(*a, **k):
        raise RuntimeError("no weights for you")

    monkeypatch.setattr(server, "ReverbEngine", boom)
    server._load_tts()  # must not raise

    assert server.tts_state == "error"
    assert server.tts is None
    assert server.voices_store is None
    assert "no weights for you" in (server.tts_error or "")


# --- §8.5 Voice list --------------------------------------------------------


def test_list_voices(client, fake_reverb, monkeypatch, tmp_path):
    engine = engines.ReverbEngine("m", "int8", "cpu")
    store = voices.VoiceStore(tmp_path / "voices")
    store.save("clone-aaa1", ServerVoice(language="de", text="hallo welt"))

    monkeypatch.setattr(server, "tts", engine)
    monkeypatch.setattr(server, "voices_store", store)
    monkeypatch.setattr(server, "tts_state", "ready")

    r = client.get("/v1/audio/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["languages"] == ["de", "en"]

    data = body["data"]
    presets = [d for d in data if d["type"] == "preset"]
    clones = [d for d in data if d["type"] == "clone"]

    # Presets first (sorted by id), then clones (sorted by id).
    assert data == presets + clones
    assert [p["id"] for p in presets] == ["de_female_standard", "en_male_narrator_1"]
    assert all(p["object"] == "voice" and "language" in p for p in presets)

    assert [c["id"] for c in clones] == ["clone-aaa1"]
    clone = clones[0]
    assert clone["object"] == "voice"
    assert clone["type"] == "clone"
    assert clone["language"] == "de"
    assert clone["transcript"] == "hallo welt"
    assert isinstance(clone["created"], int)


def test_list_voices_503_when_tts_none(client, monkeypatch):
    monkeypatch.setattr(server, "tts", None)
    monkeypatch.setattr(server, "tts_state", "disabled")
    assert client.get("/v1/audio/voices").status_code == 503


@pytest.mark.parametrize(
    "state, message",
    [
        (
            "loading",
            "tts is still loading (first start downloads about 1 GB of weights)",
        ),
        (
            "disabled",
            "tts is disabled: SPEAKEASY_TTS_ENABLED=false, or the tts extra is not installed",
        ),
        ("error", "tts encountered an error and is unavailable"),
    ],
)
def test_list_voices_503_reason_per_state(client, monkeypatch, state, message):
    monkeypatch.setattr(server, "tts", None)
    monkeypatch.setattr(server, "tts_state", state)
    r = client.get("/v1/audio/voices")
    assert r.status_code == 503
    assert r.json() == {"error": {"message": message}}


def test_list_voices_401(client, monkeypatch):
    monkeypatch.setattr(server.settings, "api_key", "unit-test-key")
    r = client.get("/v1/audio/voices")
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid or missing API key"}
