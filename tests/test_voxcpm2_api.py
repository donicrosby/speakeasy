"""VoxCPM2 HTTP surface — the lifecycle half.

The engine state machine, the shared TTS lock, the single-flight load, the
idle-unload timer, readiness and the model listing. Speech and clone-route
scenarios are added by later slices. The `voxcpm` package is faked through
`sys.modules`, so this module passes in CI's no-extras job where neither the
`tts` nor the `tts-voxcpm` extra is installed.
"""

import json
import struct
import sys
import threading
import time
import types

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import speakeasy.server as server
from speakeasy.engines import BadTake


class StubWhisper:
    """Just enough whisper to make `/health/ready` report 200."""

    load_seconds = 0.0

    def transcribe(self, *a, **k):
        raise AssertionError("transcribe is not used by the lifecycle tests")


class StubWhisperEngine:
    """Accepts the constructor arguments the lifespan passes."""

    def __init__(self, *a, **k):
        self.load_seconds = 0.0


class FakeVoxEngine:
    """A stand-in for `VoxCPM2Engine` that records its `unload`."""

    name = "voxcpm2"
    sample_rate = 48000
    load_seconds = 0.0

    def __init__(self):
        self.unloaded = 0

    def unload(self):
        self.unloaded += 1
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def voxcpm2_state_reset(monkeypatch):
    """Restore the VoxCPM2 module globals and locks around every test."""
    monkeypatch.setattr(server, "voxcpm2", None)
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server, "voxcpm2_state", "disabled")
    monkeypatch.setattr(server, "voxcpm2_error", None)
    monkeypatch.setattr(server, "voxcpm2_last_used", 0.0)
    monkeypatch.setattr(server, "tts_lock", threading.Lock())
    monkeypatch.setattr(server, "voxcpm2_state_lock", threading.Lock())
    yield


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "whisper", StubWhisper())
    monkeypatch.setattr(server, "moonshine", None)
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server.settings, "api_key", "")
    return TestClient(server.app)


@pytest.fixture()
def fake_voxcpm(monkeypatch):
    """Install a fake `voxcpm` so the startup import check passes."""
    monkeypatch.setitem(sys.modules, "voxcpm", types.ModuleType("voxcpm"))


@pytest.fixture()
def voices_root(tmp_path, monkeypatch):
    """Point the voices dir at a temp path, so nothing touches the home cache."""
    monkeypatch.setattr(server.settings, "voices_dir", str(tmp_path / "voices"))
    return tmp_path / "voices"


# --- Startup and state -------------------------------------------------


def test_voxcpm2_state_transitions(monkeypatch, fake_voxcpm, voices_root):
    # Off: the engine is `disabled` and no store is created.
    monkeypatch.setattr(server.settings, "tts_voxcpm_enabled", False)
    server.start_voxcpm2()
    assert server.voxcpm2_state == "disabled"
    assert server.voxcpm2_store is None

    # On: startup lands in `not_loaded`, with the store ready.
    monkeypatch.setattr(server.settings, "tts_voxcpm_enabled", True)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 0)
    server.start_voxcpm2()
    assert server.voxcpm2_state == "not_loaded"
    assert server.voxcpm2_store is not None

    # A request in `not_loaded` starts the load and answers the loading 503.
    started = threading.Event()
    release = threading.Event()

    def slow_engine(*a, **k):
        started.set()
        release.wait(10)
        return FakeVoxEngine()

    monkeypatch.setattr(server, "VoxCPM2Engine", slow_engine)
    resp = server._ensure_voxcpm2_loading()
    assert resp is not None and resp.status_code == 503
    assert server.voxcpm2_state == "loading"
    assert started.wait(5), "the loader thread did not start"
    release.set()

    # The loader publishes the engine and reaches `ready`.
    deadline = time.monotonic() + 5
    while server.voxcpm2_state != "ready" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.voxcpm2_state == "ready"
    assert isinstance(server.voxcpm2, FakeVoxEngine)


def test_voxcpm2_startup_returns_not_loaded(monkeypatch, fake_voxcpm, voices_root):
    """The lifespan never waits on the engine: it returns in `not_loaded`."""
    monkeypatch.setattr(server, "WhisperEngine", StubWhisperEngine)
    monkeypatch.setattr(server, "start_tts", lambda: None)
    monkeypatch.setattr(server, "whisper", None)
    monkeypatch.setattr(server, "moonshine", None)
    monkeypatch.setattr(server, "resolved_device", "unknown")
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server.settings, "moonshine_enabled", False)
    monkeypatch.setattr(server.settings, "tts_voxcpm_enabled", True)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 0)
    # Startup must never construct the engine.
    monkeypatch.setattr(
        server,
        "VoxCPM2Engine",
        lambda *a, **k: pytest.fail("startup loaded the VoxCPM2 engine"),
    )

    with TestClient(server.app) as c:
        assert c.get("/health/ready").json()["voxcpm2"] == "not_loaded"

    assert server.voxcpm2_state == "not_loaded"
    assert server.voxcpm2 is None
    assert server.voxcpm2_store is not None


def test_voxcpm2_disabled_when_setting_false(monkeypatch):
    """`SPEAKEASY_TTS_VOXCPM_ENABLED=false` → disabled, no store, no thread."""
    monkeypatch.setattr(server.settings, "tts_voxcpm_enabled", False)
    started = []
    monkeypatch.setattr(server.threading, "Thread", lambda *a, **k: started.append(a))
    server.start_voxcpm2()
    assert server.voxcpm2_state == "disabled"
    assert server.voxcpm2_store is None
    assert started == []


def test_voxcpm2_disabled_when_import_missing(monkeypatch, voices_root):
    """A missing `voxcpm` import → disabled, no store, no thread."""
    monkeypatch.setattr(server.settings, "tts_voxcpm_enabled", True)
    monkeypatch.setitem(sys.modules, "voxcpm", None)  # `import voxcpm` raises
    started = []
    monkeypatch.setattr(server.threading, "Thread", lambda *a, **k: started.append(a))
    server.start_voxcpm2()
    assert server.voxcpm2_state == "disabled"
    assert server.voxcpm2_store is None
    assert started == []


# --- Readiness and models ----------------------------------------------


def test_ready_reports_voxcpm2_state(client, monkeypatch):
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["voxcpm2"] == "ready"
    # Reverb's field is untouched.
    assert r.json()["tts"] == "disabled"


def test_models_lists_voxcpm2(client, monkeypatch):
    monkeypatch.setattr(server, "voxcpm2_state", "not_loaded")
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "voxcpm2" in ids


def test_models_omits_voxcpm2_when_disabled(client, monkeypatch):
    monkeypatch.setattr(server, "voxcpm2_state", "disabled")
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "voxcpm2" not in ids


# --- Single-flight load -------------------------------------------------


def test_voxcpm2_single_flight_load(monkeypatch):
    monkeypatch.setattr(server, "voxcpm2_state", "not_loaded")
    started = threading.Event()
    release = threading.Event()
    spawned = []

    def blocking_load():
        spawned.append(1)
        started.set()
        release.wait(10)

    monkeypatch.setattr(server, "_load_voxcpm2", blocking_load)

    first = server._ensure_voxcpm2_loading()
    assert started.wait(5), "the loader thread did not start"
    assert first is not None and first.status_code == 503
    assert server.voxcpm2_state == "loading"

    second = server._ensure_voxcpm2_loading()
    assert second is not None and second.status_code == 503
    assert spawned == [1], "a request during a load must start nothing"
    release.set()


# --- Idle unload --------------------------------------------------------


def test_voxcpm2_idle_unload(monkeypatch):
    engine = FakeVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)
    empties = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empties.append(1))

    server._voxcpm2_unload_tick()

    assert server.voxcpm2 is None
    assert server.voxcpm2_state == "not_loaded"
    assert engine.unloaded == 1
    assert empties == [1]  # the unload path released the accelerator cache


def test_voxcpm2_idle_unload_disabled(monkeypatch, fake_voxcpm, voices_root):
    """`tts_voxcpm_idle_seconds=0` never starts the timer and never unloads."""
    engine = FakeVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 0)

    server._voxcpm2_unload_tick()
    assert server.voxcpm2 is engine
    assert server.voxcpm2_state == "ready"
    assert engine.unloaded == 0

    # Startup with idle=0 must not start the timer thread either.
    monkeypatch.setattr(server, "voxcpm2", None)
    monkeypatch.setattr(server, "voxcpm2_state", "disabled")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server.settings, "tts_voxcpm_enabled", True)
    started = []
    monkeypatch.setattr(server.threading, "Thread", lambda *a, **k: started.append(a))
    server.start_voxcpm2()
    assert started == []


def test_voxcpm2_unload_skips_under_request(monkeypatch):
    """The timer cannot unload while a request holds the TTS lock."""
    engine = FakeVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)

    with server.tts_lock:
        server._voxcpm2_unload_tick()  # must not block and must not unload

    assert engine.unloaded == 0
    assert server.voxcpm2 is engine
    assert server.voxcpm2_state == "ready"


@pytest.mark.parametrize("state", ["disabled", "not_loaded", "loading", "error"])
def test_voxcpm2_unload_tick_noop_outside_ready(monkeypatch, state):
    """The timer does nothing in any state but `ready`."""
    engine = FakeVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", state)
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)

    server._voxcpm2_unload_tick()

    assert engine.unloaded == 0
    assert server.voxcpm2 is engine
    assert server.voxcpm2_state == state  # never clobbered


def test_voxcpm2_timer_skips_during_load(client, monkeypatch):
    """A tick during the loader's pre-lock phase must not clobber `loading`."""
    monkeypatch.setattr(server, "voxcpm2_state", "loading")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)

    server._voxcpm2_unload_tick()

    assert server.voxcpm2_state == "loading"  # not clobbered to not_loaded
    assert client.get("/health/ready").json()["voxcpm2"] == "loading"
    # A request still sees the load in flight.
    resp = server._ensure_voxcpm2_loading()
    assert resp is not None and resp.status_code == 503


def test_voxcpm2_timer_skips_after_load(monkeypatch):
    """A fresh ready transition resets the idle clock, so a tick cannot thrash."""
    monkeypatch.setattr(server, "voxcpm2_state", "loading")
    monkeypatch.setattr(server, "voxcpm2_last_used", 0.0)  # stale from before
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)
    monkeypatch.setattr(server, "VoxCPM2Engine", lambda *a, **k: FakeVoxEngine())

    server._load_voxcpm2()
    assert server.voxcpm2_state == "ready"
    assert time.monotonic() - server.voxcpm2_last_used < 60  # reset at ready
    engine = server.voxcpm2

    server._voxcpm2_unload_tick()  # immediately after the load completes

    assert server.voxcpm2 is engine
    assert server.voxcpm2_state == "ready"
    assert engine.unloaded == 0


def test_voxcpm2_unload_keeps_store(monkeypatch):
    """An idle unload drops the engine but not the clone store."""
    store = object()
    monkeypatch.setattr(server, "voxcpm2", FakeVoxEngine())
    monkeypatch.setattr(server, "voxcpm2_store", store)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)

    server._voxcpm2_unload_tick()

    assert server.voxcpm2_state == "not_loaded"
    assert server.voxcpm2_store is store


def test_voxcpm2_request_during_unload(monkeypatch):
    """A request arriving while an unload holds the lock answers 503, never 500."""
    engine = FakeVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)

    holding = threading.Event()
    release = threading.Event()

    def unload_holding_lock():
        with server.tts_lock:
            server.voxcpm2 = None
            server.voxcpm2_state = "not_loaded"
            holding.set()
            release.wait(10)

    t = threading.Thread(target=unload_holding_lock)
    t.start()
    assert holding.wait(5), "the unload did not take the lock"
    try:
        resp = server.voxcpm2_unavailable()
        assert resp is not None and resp.status_code == 503
        assert json.loads(resp.body) == {
            "error": {"message": server.VOXCPM2_503_REASONS["not_loaded"]}
        }
    finally:
        release.set()
        t.join(5)


def test_voxcpm2_state_recheck_under_lock(monkeypatch):
    """A caller that saw `ready` re-checks under the lock; a completed unload
    leaves `not_loaded`, so it starts a new load and answers 503 — never 500."""
    monkeypatch.setattr(server, "voxcpm2", FakeVoxEngine())
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_last_used", time.monotonic() - 10_000)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 600)

    # The unload wins the lock and completes while the request is still queued.
    server._voxcpm2_unload_tick()
    assert server.voxcpm2 is None
    assert server.voxcpm2_state == "not_loaded"

    loads = []
    monkeypatch.setattr(server, "_load_voxcpm2", lambda: loads.append(1))
    with server.tts_lock:
        # The request re-checks the state under the lock before using the engine.
        assert server.voxcpm2_state != "ready"
        resp = server._ensure_voxcpm2_loading()
    assert resp is not None and resp.status_code == 503
    assert json.loads(resp.body) == {
        "error": {"message": server.VOXCPM2_503_REASONS["not_loaded"]}
    }
    assert loads == [1]  # a new load started; nothing reached a model call


# --- Failure and 503 bodies --------------------------------------------


def test_voxcpm2_load_failure_sets_error(client, monkeypatch):
    secret = "/models/weights/secret-path"

    def boom(*a, **k):
        raise RuntimeError(f"could not load {secret}")

    monkeypatch.setattr(server, "VoxCPM2Engine", boom)
    monkeypatch.setattr(server, "voxcpm2_state", "loading")

    server._load_voxcpm2()  # must not raise

    assert server.voxcpm2_state == "error"
    assert server.voxcpm2 is None
    assert secret in server.voxcpm2_error  # kept for the logs
    assert client.get("/health/ready").json()["voxcpm2"] == "error"
    resp = server.voxcpm2_unavailable()
    assert resp is not None and resp.status_code == 503
    assert secret not in resp.body.decode()


@pytest.mark.parametrize(
    "state, message",
    [
        (
            "disabled",
            "voxcpm2 is disabled; set SPEAKEASY_TTS_VOXCPM_ENABLED=true to enable it",
        ),
        (
            "not_loaded",
            "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)",
        ),
        (
            "loading",
            "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)",
        ),
        ("error", "voxcpm2 encountered an error and is unavailable"),
    ],
)
def test_voxcpm2_503_body_per_state(monkeypatch, state, message):
    monkeypatch.setattr(server, "voxcpm2_state", state)
    monkeypatch.setattr(server, "voxcpm2_error", "leaked detail: /weights/secret")

    resp = server.voxcpm2_unavailable()

    assert resp is not None and resp.status_code == 503
    assert json.loads(resp.body) == {"error": {"message": message}}
    assert "secret" not in resp.body.decode()


def test_voxcpm2_503_is_none_when_ready(monkeypatch):
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    assert server.voxcpm2_unavailable() is None


# ---------------------------------------------------------------------------
# Speech — routing, validation, composition and encoding for both engines
# ---------------------------------------------------------------------------


class SpeechVoxEngine:
    """A stub VoxCPM2 engine recording the route's `synthesize` call."""

    name = "voxcpm2"
    sample_rate = 48000
    load_seconds = 0.0

    def __init__(self, seconds=1.0, *, bad=False, exc=None, delay=0.0):
        self.seconds = seconds
        self.bad = bad
        self.exc = exc
        self.delay = delay
        self.calls = []
        self.seed_lock_states = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._guard = threading.Lock()

    def synthesize(self, text, *, reference=None, prompt_text=None, cfg=2.0, seed=None):
        self.calls.append(
            {
                "text": text,
                "reference": reference,
                "prompt_text": prompt_text,
                "cfg": cfg,
                "seed": seed,
            }
        )
        if seed is not None:
            # The route must hold the shared lock across seed + generate.
            self.seed_lock_states.append(server.tts_lock.locked())
            torch.manual_seed(seed)
        if self.exc is not None:
            raise self.exc
        if self.bad:
            raise BadTake("too many audio patches for the text")
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
        finally:
            with self._guard:
                self.in_flight -= 1
        n = int(self.sample_rate * self.seconds)
        t = np.arange(n) / self.sample_rate
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 0.1

    def unload(self):
        pass


class ReverbStub:
    """A stub Reverb engine for the speech route (presets + synthesize)."""

    name = "reverb-tts"
    sample_rate = 24000
    languages = ["de", "en"]
    load_seconds = 0.0

    def __init__(self, seconds=1.0, delay=0.0):
        self.seconds = seconds
        self.delay = delay
        self.calls = []

    def presets(self):
        return [
            {"id": "en_male_narrator_1", "language": "en", "transcript": ""},
            {"id": "de_female_standard", "language": "de", "transcript": ""},
        ]

    def voice(self, preset_id):
        langs = {"en_male_narrator_1": "en", "de_female_standard": "de"}
        if preset_id not in langs:
            raise ValueError(f"unknown voice {preset_id!r}")
        return types.SimpleNamespace(language=langs[preset_id])

    def synthesize(self, text, voice, language, *, seed=None, speed=1.0):
        self.calls.append({"text": text, "seed": seed, "speed": speed})
        time.sleep(self.delay)
        n = int(self.sample_rate * self.seconds)
        t = np.arange(n) / self.sample_rate
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 0.1


class StubVoxStore:
    """A minimal VoxCPM2 clone store: every id `exists`, with a transcript or None."""

    def __init__(self, transcripts: dict):
        self._transcripts = dict(transcripts)

    def exists(self, voice_id):
        return voice_id in self._transcripts

    def ids(self):
        return sorted(self._transcripts)

    def info(self, voice_id):
        if voice_id not in self._transcripts:
            raise ValueError(voice_id)
        return {
            "engine": "voxcpm2",
            "transcript": self._transcripts[voice_id],
            "seconds": 2.0,
            "created": 1,
        }

    def reference_path(self, voice_id):
        return f"/nonexistent/voxcpm2/{voice_id}.wav"


class InstrumentedLock:
    """A real lock that signals when its *second* `with` caller is blocked on it."""

    def __init__(self, second_enter: threading.Event):
        self._lock = threading.Lock()
        self._guard = threading.Lock()
        self._enters = 0
        self._second_enter = second_enter

    def __enter__(self):
        with self._guard:
            self._enters += 1
            n = self._enters
        if n == 2:
            self._second_enter.set()
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False

    def locked(self):
        return self._lock.locked()

    def acquire(self, blocking=True):
        return self._lock.acquire(blocking=blocking)

    def release(self):
        self._lock.release()


def _wav_rate(body: bytes) -> int:
    assert body[:4] == b"RIFF"
    i = body.index(b"fmt ")
    return struct.unpack("<I", body[i + 12 : i + 16])[0]


def _post(payload):
    """Drive the route directly, bypassing auth and HTTP serialisation."""
    return server.speech(server.SpeechRequest(**payload))


@pytest.fixture()
def ready_vox(monkeypatch):
    """A ready VoxCPM2 stub with no Reverb, for the speech validation tests."""
    engine = SpeechVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    return engine


def test_speech_voxcpm2_routing(client, monkeypatch, ready_vox):
    reverb = ReverbStub()
    monkeypatch.setattr(server, "tts", reverb)
    monkeypatch.setattr(server, "tts_state", "ready")

    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "default"},
    )

    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "voxcpm2"
    assert ready_vox.calls[-1]["text"] == "Hi"
    assert reverb.calls == []  # the Reverb engine is never touched


def test_voxcpm2_speed_rejected(client, ready_vox):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "default", "speed": 2.0},
    )

    assert r.status_code == 400
    assert r.json() == {
        "error": {"message": "speed must be 1.0 for the voxcpm2 engine"}
    }
    assert ready_vox.calls == []


@pytest.mark.parametrize(
    "extra", [{"cfg": 3.0}, {"design": "A calm voice"}, {"style": "cheerful"}]
)
def test_voxcpm2_fields_rejected_on_reverb(client, monkeypatch, extra):
    monkeypatch.setattr(server, "tts", ReverbStub())
    monkeypatch.setattr(server, "tts_state", "ready")

    r = client.post("/v1/audio/speech", json={"model": "", "input": "Hi", **extra})

    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": "cfg, design and style are voxcpm2-only fields; set model=voxcpm2"
        }
    }


@pytest.mark.parametrize("cfg", [0.5, 5.5, 6.0])
def test_voxcpm2_cfg_bounds(client, ready_vox, cfg):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "default", "cfg": cfg},
    )

    assert r.status_code == 422
    assert any(e["loc"][-1] == "cfg" for e in r.json()["detail"])


@pytest.mark.parametrize(
    "field, value",
    [
        ("design", "a (wrong voice"),
        ("design", "a wrong) voice"),
        ("design", "two\nlines"),
        ("style", "cheerful)"),
        ("style", "two\nlines"),
    ],
)
def test_voxcpm2_design_style_syntax(client, ready_vox, field, value):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "default", field: value},
    )

    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": "design and style must not contain parentheses or newlines"
        }
    }
    assert ready_vox.calls == []


def test_voxcpm2_design_requires_default(client, monkeypatch, ready_vox):
    monkeypatch.setattr(server, "voxcpm2_store", StubVoxStore({"clone-xy": None}))

    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hi",
            "voice": "clone-xy",
            "design": "A calm voice",
        },
    )

    assert r.status_code == 400
    assert r.json() == {"error": {"message": "design requires voice: 'default'"}}


def test_voxcpm2_design_style_mutual_exclusion(client, ready_vox):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hi",
            "voice": "default",
            "design": "A calm voice",
            "style": "cheerful",
        },
    )

    assert r.status_code == 400
    assert r.json() == {
        "error": {"message": "at most one of design and style may be set"}
    }
    assert ready_vox.calls == []


def test_voxcpm2_voice_default(client, ready_vox):
    r = client.post(
        "/v1/audio/speech", json={"model": "voxcpm2", "input": "Hi", "voice": "default"}
    )

    assert r.status_code == 200
    call = ready_vox.calls[-1]
    assert call["reference"] is None
    assert call["prompt_text"] is None
    assert call["cfg"] == 2.0  # null cfg resolves to the engine default


def test_voxcpm2_voice_omitted_is_default(client, ready_vox):
    r = client.post("/v1/audio/speech", json={"model": "voxcpm2", "input": "Hi"})

    assert r.status_code == 200
    call = ready_vox.calls[-1]
    assert call["reference"] is None
    assert call["prompt_text"] is None


def test_voxcpm2_reverb_preset_rejected(client, monkeypatch, ready_vox):
    monkeypatch.setattr(server, "voxcpm2_store", StubVoxStore({"clone-saved": None}))

    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "en_male_narrator_1"},
    )

    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": (
                "unknown voice 'en_male_narrator_1' for voxcpm2; "
                "use 'default' or a saved clone: clone-saved"
            )
        }
    }
    assert ready_vox.calls == []


def test_voxcpm2_clone_with_transcript(client, monkeypatch, ready_vox, voices_root):
    store = server.ReferenceVoiceStore(voices_root)
    store.save("clone-abc", np.zeros(16000, dtype=np.float32), 16000, "Hello there")
    monkeypatch.setattr(server, "voxcpm2_store", store)

    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "clone-abc"},
    )

    assert r.status_code == 200
    call = ready_vox.calls[-1]
    assert call["reference"] == str(store.reference_path("clone-abc"))
    assert call["prompt_text"] == "Hello there"


def test_voxcpm2_clone_without_transcript(client, monkeypatch, ready_vox, voices_root):
    store = server.ReferenceVoiceStore(voices_root)
    store.save("clone-no", np.zeros(16000, dtype=np.float32), 16000, None)
    monkeypatch.setattr(server, "voxcpm2_store", store)

    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "clone-no"},
    )

    assert r.status_code == 200
    call = ready_vox.calls[-1]
    assert call["reference"] == str(store.reference_path("clone-no"))
    assert call["prompt_text"] is None


def test_voxcpm2_style_prefix(client, monkeypatch, ready_vox, voices_root):
    store = server.ReferenceVoiceStore(voices_root)
    store.save("clone-xy", np.zeros(16000, dtype=np.float32), 16000, "Hey")
    monkeypatch.setattr(server, "voxcpm2_store", store)

    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello.",
            "voice": "clone-xy",
            "style": "cheerful",
        },
    )

    assert r.status_code == 200
    assert ready_vox.calls[-1]["text"] == "(cheerful)Hello."


def test_voxcpm2_design_prefix(client, ready_vox):
    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hello.",
            "voice": "default",
            "design": "A young woman",
        },
    )

    assert r.status_code == 200
    call = ready_vox.calls[-1]
    assert call["text"] == "(A young woman)Hello."
    assert call["reference"] is None


def test_voxcpm2_bad_take_503(client, monkeypatch):
    monkeypatch.setattr(server, "voxcpm2", SpeechVoxEngine(bad=True))
    monkeypatch.setattr(server, "voxcpm2_state", "ready")

    r = client.post(
        "/v1/audio/speech", json={"model": "voxcpm2", "input": "Hi", "voice": "default"}
    )

    assert r.status_code == 503
    assert r.headers["content-type"] == "application/json"
    assert r.json() == {
        "error": {
            "message": (
                "voxcpm2 could not generate a good take for this input; "
                "check the input length and try again"
            )
        }
    }


def test_no_exception_text_in_voxcpm2_error_bodies(client, monkeypatch):
    secret = "/models/weights/secret-path"

    monkeypatch.setattr(
        server, "voxcpm2", SpeechVoxEngine(exc=RuntimeError(f"boom {secret}"))
    )
    monkeypatch.setattr(server, "voxcpm2_state", "ready")

    r = client.post(
        "/v1/audio/speech", json={"model": "voxcpm2", "input": "Hi", "voice": "default"}
    )

    assert r.status_code in (500, 503)
    assert secret not in r.text
    assert "boom" not in r.text


@pytest.mark.parametrize(
    "fmt, media",
    [
        ("mp3", "audio/mpeg"),
        ("wav", "audio/wav"),
        ("flac", "audio/flac"),
        ("pcm", "audio/pcm"),
    ],
)
def test_voxcpm2_response_formats(client, monkeypatch, fmt, media):
    monkeypatch.setattr(server, "voxcpm2", SpeechVoxEngine(seconds=1.0))
    monkeypatch.setattr(server, "voxcpm2_state", "ready")

    r = client.post(
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hi",
            "voice": "default",
            "response_format": fmt,
        },
    )

    assert r.status_code == 200
    assert r.headers["content-type"] == media
    assert r.headers["x-speakeasy-engine"] == "voxcpm2"
    body = r.content
    if fmt == "wav":
        assert _wav_rate(body) == 48000
    elif fmt == "pcm":
        assert len(body) == 48000 * 2
    elif fmt == "mp3":
        assert body[:3] == b"ID3" or (body[0] == 0xFF and (body[1] & 0xE0) == 0xE0)
    else:
        assert body[:4] == b"fLaC"


@pytest.mark.parametrize(
    "fmt, media",
    [
        ("mp3", "audio/mpeg"),
        ("wav", "audio/wav"),
        ("flac", "audio/flac"),
        ("pcm", "audio/pcm"),
    ],
)
def test_reverb_response_formats_24k(client, monkeypatch, fmt, media):
    monkeypatch.setattr(server, "tts", ReverbStub(seconds=1.0))
    monkeypatch.setattr(server, "tts_state", "ready")
    monkeypatch.setattr(server, "voices_store", None)

    r = client.post("/v1/audio/speech", json={"input": "Hi", "response_format": fmt})

    assert r.status_code == 200
    assert r.headers["content-type"] == media
    assert r.headers["x-speakeasy-engine"] == "reverb-tts"
    body = r.content
    if fmt == "wav":
        assert _wav_rate(body) == 24000
    elif fmt == "pcm":
        assert len(body) == 24000 * 2


def test_voxcpm2_speech_503_401(client, monkeypatch):
    monkeypatch.setattr(server, "voxcpm2", None)
    monkeypatch.setattr(server, "_load_voxcpm2", lambda: None)

    for state, message in [
        (
            "disabled",
            "voxcpm2 is disabled; set SPEAKEASY_TTS_VOXCPM_ENABLED=true to enable it",
        ),
        (
            "not_loaded",
            "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)",
        ),
        (
            "loading",
            "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)",
        ),
        ("error", "voxcpm2 encountered an error and is unavailable"),
    ]:
        monkeypatch.setattr(server, "voxcpm2_state", state)
        r = client.post(
            "/v1/audio/speech",
            json={"model": "voxcpm2", "input": "Hi", "voice": "default"},
        )
        assert r.status_code == 503
        assert r.json() == {"error": {"message": message}}

    # A seeded request in `not_loaded` started the load as a side effect.
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2", SpeechVoxEngine())
    monkeypatch.setattr(server.settings, "api_key", "unit-test-key")
    r = client.post(
        "/v1/audio/speech", json={"model": "voxcpm2", "input": "Hi", "voice": "default"}
    )
    assert r.status_code == 401


def test_voxcpm2_clone_deleted_before_library_call(client, monkeypatch):
    monkeypatch.setattr(
        server, "voxcpm2", SpeechVoxEngine(exc=FileNotFoundError("gone"))
    )
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", StubVoxStore({"clone-gone": None}))

    r = client.post(
        "/v1/audio/speech",
        json={"model": "voxcpm2", "input": "Hi", "voice": "clone-gone"},
    )

    assert r.status_code == 400
    assert r.json() == {
        "error": {
            "message": (
                "unknown voice 'clone-gone' for voxcpm2; "
                "use 'default' or a saved clone: clone-gone"
            )
        }
    }


def test_shared_tts_lock_serialises_both_engines(monkeypatch):
    state = {"in_flight": 0, "max": 0}
    guard = threading.Lock()

    def enter():
        with guard:
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])

    def leave():
        with guard:
            state["in_flight"] -= 1

    class ConcurrencyReverb(ReverbStub):
        def synthesize(self, text, voice, language, *, seed=None, speed=1.0):
            enter()
            try:
                time.sleep(0.05)
                return np.zeros(2400, dtype=np.float32), 0.05
            finally:
                leave()

    class ConcurrencyVox(SpeechVoxEngine):
        def synthesize(
            self, text, *, reference=None, prompt_text=None, cfg=2.0, seed=None
        ):
            enter()
            try:
                time.sleep(0.05)
                return np.zeros(4800, dtype=np.float32), 0.05
            finally:
                leave()

    monkeypatch.setattr(server, "tts", ConcurrencyReverb())
    monkeypatch.setattr(server, "tts_state", "ready")
    monkeypatch.setattr(server, "voices_store", None)
    monkeypatch.setattr(server, "voxcpm2", ConcurrencyVox())
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    results = []

    def reverb():
        results.append(_post({"input": "Hi"}))

    def vox():
        results.append(_post({"model": "voxcpm2", "input": "Hi", "voice": "default"}))

    threads = [threading.Thread(target=reverb), threading.Thread(target=vox)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert all(not t.is_alive() for t in threads)
    assert [r.status_code for r in results] == [200, 200]
    assert state["max"] == 1  # never two engines inferring at once


def test_voxcpm2_seeding_atomic(monkeypatch):
    engine = SpeechVoxEngine(delay=0.05)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    def call(seed):
        _post({"model": "voxcpm2", "input": "Hi", "voice": "default", "seed": seed})

    threads = [threading.Thread(target=call, args=(n,)) for n in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    assert all(not t.is_alive() for t in threads)
    assert engine.max_in_flight == 1  # each seed + generate pair is atomic
    assert engine.seed_lock_states == [True, True]  # seeded under the shared lock


def test_voxcpm2_speech_request_during_unload(monkeypatch):
    engine = SpeechVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server, "_load_voxcpm2", lambda: None)

    second_enter = threading.Event()
    monkeypatch.setattr(server, "tts_lock", InstrumentedLock(second_enter))
    release_ready = threading.Event()

    def unload_holding_lock():
        with server.tts_lock:
            release_ready.wait(10)  # the request blocks on the lock meanwhile
            server.voxcpm2 = None
            server.voxcpm2_state = "not_loaded"

    unloader = threading.Thread(target=unload_holding_lock)
    unloader.start()
    assert unloader.is_alive()

    result = []
    req = threading.Thread(
        target=lambda: result.append(
            _post({"model": "voxcpm2", "input": "Hi", "voice": "default"})
        )
    )
    req.start()
    assert second_enter.wait(5), "the request never reached the lock"
    release_ready.set()
    unloader.join(5)
    req.join(10)

    assert not req.is_alive()
    resp = result[0]
    assert resp.status_code == 503
    assert json.loads(resp.body) == {
        "error": {"message": server.VOXCPM2_503_REASONS["not_loaded"]}
    }
    assert engine.calls == []  # the unloaded model is never called


def test_voxcpm2_speech_state_recheck_under_lock(monkeypatch):
    engine = SpeechVoxEngine()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server, "_load_voxcpm2", lambda: None)

    real = threading.Lock()

    class FlipOnEnter:
        """Simulate an unload that completes before the request takes the lock."""

        def __enter__(self):
            server.voxcpm2 = None
            server.voxcpm2_state = "not_loaded"
            real.acquire()
            return self

        def __exit__(self, *exc):
            real.release()
            return False

        def locked(self):
            return real.locked()

        def acquire(self, blocking=True):
            return real.acquire(blocking=blocking)

        def release(self):
            real.release()

    monkeypatch.setattr(server, "tts_lock", FlipOnEnter())

    resp = _post({"model": "voxcpm2", "input": "Hi", "voice": "default"})

    assert resp.status_code == 503
    assert json.loads(resp.body) == {
        "error": {"message": server.VOXCPM2_503_REASONS["not_loaded"]}
    }
    assert engine.calls == []  # never reached a model call
    assert server.voxcpm2_state == "loading"  # a new single-flight load started
