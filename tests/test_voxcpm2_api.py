"""VoxCPM2 HTTP surface — the lifecycle half.

The engine state machine, the shared TTS lock, the single-flight load, the
idle-unload timer, readiness and the model listing. Speech and clone-route
scenarios are added by later slices. The `voxcpm` package is faked through
`sys.modules`, so this module passes in CI's no-extras job where neither the
`tts` nor the `tts-voxcpm` extra is installed.
"""

import json
import sys
import threading
import time
import types

import pytest
import torch
from fastapi.testclient import TestClient

import speakeasy.server as server


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
