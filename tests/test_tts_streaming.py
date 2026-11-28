"""Streaming speech — the engine half.

The engine's streaming methods are unit-tested here against fake libraries,
the way `test_voxcpm2_engine.py` fakes `voxcpm` and `test_tts.py` fakes
`reverb_tts`: no GPU, no model download, and the suite passes in CI's
no-extras job. The route and the page are added by later slices.

The fakes grow the streaming entry points of the libraries they stand in for,
so the route can be exercised without a GPU.
"""

import base64
import json
import logging
import sys
import threading
import time
import types

import anyio
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import speakeasy.server as server
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


# ---------------------------------------------------------------------------
# POST /v1/audio/speech — streaming delivery
# ---------------------------------------------------------------------------

CHUNK_SECONDS = 0.16
VOX_RATE = 48000
REVERB_RATE = 24000


class StubWhisper:
    """Just enough whisper to make `/health/ready` report 200."""

    load_seconds = 0.0

    def transcribe(self, *a, **k):
        raise AssertionError("transcribe is unused by the streaming tests")


class StubStore:
    """A voice store that serves fixed voices, for clone resolution."""

    def __init__(self, voices=None):
        self._voices = voices or {}

    def exists(self, voice_id):
        return voice_id in self._voices

    def load(self, voice_id):
        return self._voices[voice_id]

    def ids(self):
        return sorted(self._voices)


class StubVoxStore:
    """A minimal VoxCPM2 clone store: every id exists, with a transcript or None."""

    def __init__(self, transcripts):
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


class StreamVox:
    """A stub VoxCPM2 engine that streams a deterministic take in fixed chunks."""

    name = "voxcpm2"
    sample_rate = VOX_RATE
    load_seconds = 0.0

    def __init__(
        self,
        *,
        chunks=3,
        seconds=CHUNK_SECONDS,
        delay=0.0,
        exc=None,
        exc_after=None,
        gate=None,
        zero=False,
    ):
        self.chunks = chunks
        self.chunk_samples = int(self.sample_rate * seconds)
        self.delay = delay
        self.exc = exc
        self.exc_after = exc_after
        self.gate = gate
        self.zero = zero
        self.synthesize_calls = []
        self.stream_calls = []
        self.last_audio = None
        self.in_flight = 0
        self.max_in_flight = 0
        self.completed = threading.Event()
        self._guard = threading.Lock()

    def chunk(self, index):
        n = self.chunk_samples
        if self.zero:
            return np.zeros(n, dtype=np.float32)
        t = (np.arange(n) + index * n) / self.sample_rate
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def take(self):
        return np.concatenate([self.chunk(i) for i in range(self.chunks)])

    def synthesize(self, text, *, reference=None, prompt_text=None, cfg=2.0, seed=None):
        self.synthesize_calls.append(
            {
                "text": text,
                "reference": reference,
                "prompt_text": prompt_text,
                "cfg": cfg,
                "seed": seed,
            }
        )
        time.sleep(self.delay)
        self.last_audio = self.take()
        return self.last_audio, 0.1

    def synthesize_streaming(
        self, text, *, reference=None, prompt_text=None, cfg=2.0, seed=None
    ):
        self.stream_calls.append(
            {
                "text": text,
                "reference": reference,
                "prompt_text": prompt_text,
                "cfg": cfg,
                "seed": seed,
            }
        )
        if self.gate is not None:
            self.gate.wait(10)
        with self._guard:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            for i in range(self.chunks):
                if self.exc is not None and self.exc_after == i:
                    raise self.exc
                time.sleep(self.delay)
                yield self.chunk(i)
            if self.exc is not None and self.exc_after is None:
                raise self.exc
        finally:
            with self._guard:
                self.in_flight -= 1
            self.completed.set()


class StreamReverb:
    """A stub Reverb engine: no streaming entry point, so one finished chunk."""

    name = "reverb-tts"
    sample_rate = REVERB_RATE
    languages = ["de", "en"]
    load_seconds = 0.0

    def __init__(self, *, seconds=0.5, delay=0.0, exc=None):
        self.seconds = seconds
        self.delay = delay
        self.exc = exc
        self.synthesize_calls = []
        self.stream_calls = []
        self.last_audio = None

    def presets(self):
        return [
            {"id": "de_female_standard", "language": "de", "transcript": ""},
            {"id": "en_male_narrator_1", "language": "en", "transcript": ""},
        ]

    def voice(self, preset_id):
        langs = {"en_male_narrator_1": "en", "de_female_standard": "de"}
        if preset_id not in langs:
            raise ValueError(f"unknown voice {preset_id!r}")
        return types.SimpleNamespace(language=langs[preset_id])

    def take(self):
        n = int(self.sample_rate * self.seconds)
        t = np.arange(n) / self.sample_rate
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def synthesize(self, text, voice, language, *, seed=None, speed=1.0):
        self.synthesize_calls.append({"text": text, "seed": seed, "speed": speed})
        if self.exc is not None:
            raise self.exc
        time.sleep(self.delay)
        self.last_audio = self.take()
        return self.last_audio, 0.1

    def synthesize_streaming(self, text, voice, language, *, seed=None, speed=1.0):
        self.stream_calls.append({"text": text, "seed": seed, "speed": speed})
        if self.exc is not None:
            raise self.exc
        time.sleep(self.delay)
        yield self.take()


@pytest.fixture(autouse=True)
def stream_state_reset(monkeypatch):
    """Restore the speech-route globals and locks around every streaming test."""
    monkeypatch.setattr(server, "tts", None)
    monkeypatch.setattr(server, "voices_store", None)
    monkeypatch.setattr(server, "tts_state", "disabled")
    monkeypatch.setattr(server, "tts_error", None)
    monkeypatch.setattr(server, "voxcpm2", None)
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server, "voxcpm2_state", "disabled")
    monkeypatch.setattr(server, "voxcpm2_error", None)
    monkeypatch.setattr(server, "voxcpm2_last_used", 0.0)
    monkeypatch.setattr(server, "tts_lock", threading.Lock())
    monkeypatch.setattr(server, "voxcpm2_state_lock", threading.Lock())
    monkeypatch.setattr(server, "voxcpm2_create_lock", threading.Lock())
    yield


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "whisper", StubWhisper())
    monkeypatch.setattr(server, "moonshine", None)
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server.settings, "api_key", "")
    return TestClient(server.app)


@pytest.fixture()
def vox_ready(monkeypatch):
    engine = StreamVox()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    return engine


@pytest.fixture()
def reverb_ready(monkeypatch):
    engine = StreamReverb()
    monkeypatch.setattr(server, "tts", engine)
    monkeypatch.setattr(server, "voices_store", StubStore())
    monkeypatch.setattr(server, "tts_state", "ready")
    return engine


def _speech(client, **payload):
    return client.post("/v1/audio/speech", json=payload)


def _sse_events(body: bytes) -> list[dict]:
    events = []
    for line in body.decode().splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _deltas(body: bytes) -> list[dict]:
    return [e for e in _sse_events(body) if e["type"] == "speech.audio.delta"]


def _pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def _wait(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _drive_disconnect(payload, *, after_chunks=1):
    """Drive one streaming request through the ASGI app and disconnect mid-stream.

    The test client runs a streaming response to completion before it returns,
    so a mid-stream disconnect cannot be simulated through it. uvicorn answers
    an ASGI `spec_version` of 2.3, where Starlette cancels the stream when
    `receive` yields `http.disconnect`; this drives the app exactly that way,
    stopping after `after_chunks` body chunks, and returns the sent messages.
    """
    body = json.dumps(payload).encode()
    sent = []

    async def run():
        seen = 0
        disconnected = anyio.Event()
        state = {"body": False}

        async def receive():
            if not state["body"]:
                state["body"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal seen
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                seen += 1
                if seen >= after_chunks:
                    disconnected.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/audio/speech",
            "raw_path": b"/v1/audio/speech",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "scheme": "http",
            "server": ("test", 80),
            "client": ("test", 123),
            "root_path": "",
        }
        await server.app(scope, receive, send)

    anyio.run(run)
    return sent


STREAM_PAYLOAD = {
    "model": "voxcpm2",
    "input": "Hi",
    "voice": "default",
    "response_format": "pcm",
    "stream_format": "audio",
}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_streaming_format_probe():
    """Only a format whose bytes survive a sink that cannot seek streams."""
    sample_rate = 24000
    chunks = 3
    expected = chunks * int(sample_rate * 0.16)
    for fmt in server.SPEECH_FORMATS:
        decoded = server._probe_streaming_format(fmt, sample_rate, chunks=chunks)
        if fmt in server._STREAMING_FORMATS:
            assert decoded == expected, fmt
        else:
            assert decoded != expected, fmt
    assert "pcm" in server._STREAMING_FORMATS


def test_non_streaming_unchanged(client, vox_ready):
    """No `stream_format`: today's one-shot response, byte for byte."""
    r = _speech(client, model="voxcpm2", input="Hi", voice="default")

    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.headers["x-speakeasy-engine"] == "voxcpm2"
    for header in (
        "x-speakeasy-elapsed",
        "x-speakeasy-audio-duration",
        "x-speakeasy-rtf",
    ):
        assert header in r.headers
    expected, _ = server._encode(vox_ready.last_audio, "mp3", vox_ready.sample_rate)
    assert r.content == expected
    assert len(vox_ready.synthesize_calls) == 1
    assert vox_ready.stream_calls == []


def test_sse_stream_voxcpm2(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-store"
    events = _sse_events(r.content)
    deltas = [e for e in events if e["type"] == "speech.audio.delta"]
    assert len(deltas) == 3
    assert events[-1]["type"] == "speech.audio.done"
    assert b"[DONE]" not in r.content
    assert set(events[-1]["usage"]) == {"engine", "elapsed", "audio_duration", "rtf"}


def test_raw_stream_pcm(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="audio",
    )

    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/pcm"
    assert r.headers["x-speakeasy-engine"] == "voxcpm2"
    assert int(r.headers["x-speakeasy-sample-rate"]) == VOX_RATE
    assert r.content == _pcm(vox_ready.take())


def test_reverb_stream_returns_finished_take(client, reverb_ready):
    r = _speech(
        client,
        model="reverb-tts",
        input="Hi",
        response_format="pcm",
        stream_format="sse",
    )

    assert r.status_code == 200
    events = _sse_events(r.content)
    deltas = [e for e in events if e["type"] == "speech.audio.delta"]
    assert len(deltas) == 1
    assert events[-1]["type"] == "speech.audio.done"
    assert base64.b64decode(deltas[0]["audio"]) == _pcm(reverb_ready.take())
    assert events[-1]["usage"]["engine"] == "reverb-tts"


def test_invalid_stream_format_422(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        stream_format="invalid_value",
    )

    assert r.status_code == 422
    assert any(e["loc"][-1] == "stream_format" for e in r.json()["detail"])


def test_streaming_refused_format_422(client, vox_ready):
    for form in ("sse", "audio"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="wav",
            stream_format=form,
        )
        assert r.status_code == 422, form
        assert r.json() == {
            "error": {
                "message": "streaming not supported for format 'wav'. Supported: pcm"
            }
        }


def test_sse_stream_headers(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )

    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-accel-buffering"] == "no"
    assert int(r.headers["x-speakeasy-sample-rate"]) == VOX_RATE


def test_raw_stream_headers(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="audio",
    )

    assert r.headers["x-speakeasy-engine"] == "voxcpm2"
    assert int(r.headers["x-speakeasy-sample-rate"]) == VOX_RATE
    for absent in (
        "x-speakeasy-elapsed",
        "x-speakeasy-audio-duration",
        "x-speakeasy-rtf",
    ):
        assert absent not in r.headers


def test_sse_stream_final_event_carries_stats(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )

    usage = _sse_events(r.content)[-1]["usage"]
    assert set(usage) == {"engine", "elapsed", "audio_duration", "rtf"}
    assert usage["engine"] == "voxcpm2"
    assert usage["elapsed"] >= 0
    assert usage["audio_duration"] == pytest.approx(3 * CHUNK_SECONDS, abs=0.01)
    assert usage["rtf"] >= 0


def test_stream_and_non_stream_same_take(client, vox_ready):
    streamed = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )
    plain = _speech(
        client, model="voxcpm2", input="Hi", voice="default", response_format="pcm"
    )

    joined = b"".join(base64.b64decode(e["audio"]) for e in _deltas(streamed.content))
    assert joined == plain.content
    assert np.array_equal(
        np.frombuffer(joined, dtype="<i2"), np.frombuffer(plain.content, dtype="<i2")
    )


def test_engine_failure_mid_stream_clean_close(client, monkeypatch):
    engine = StreamVox(
        chunks=4, exc=RuntimeError("secret /path/to/weights"), exc_after=2
    )
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )

    assert r.status_code == 200
    events = _sse_events(r.content)
    assert len([e for e in events if e["type"] == "speech.audio.delta"]) == 2
    assert events[-1] == {
        "type": "error",
        "error": {"message": server.STREAM_ERROR_MESSAGE},
    }
    assert b"[DONE]" not in r.content
    assert b"secret" not in r.content
    assert b"weights" not in r.content


def test_engine_failure_after_final_chunk(client, monkeypatch):
    """A failure that lands after the last chunk is still mid-stream, never done."""
    engine = StreamVox(chunks=3, exc=engines.BadTake("too long"))
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )

    assert r.status_code == 200
    events = _sse_events(r.content)
    assert len([e for e in events if e["type"] == "speech.audio.delta"]) == 3
    assert events[-1]["type"] == "error"


class GeneratorBoom(BaseException):
    """A failure `except Exception` cannot catch — a library calling `sys.exit`
    inside its stream generator, or any other `BaseException`."""


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_generation_dying_after_a_chunk_ends_the_stream(client, monkeypatch, caplog):
    """A `BaseException` past the first chunk ends the stream; it never hangs it."""
    engine = StreamVox(chunks=4, exc=GeneratorBoom("gone"), exc_after=1)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with caplog.at_level(logging.ERROR, logger="speakeasy"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    assert r.status_code == 200
    events = _sse_events(r.content)
    assert len([e for e in events if e["type"] == "speech.audio.delta"]) == 1
    assert events[-1] == {
        "type": "error",
        "error": {"message": server.STREAM_ERROR_MESSAGE},
    }
    assert len([rec for rec in caplog.records if rec.levelno >= logging.ERROR]) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_generation_dying_before_a_chunk_answers_a_500(client, monkeypatch, caplog):
    """A `BaseException` before the first chunk answers with a status, not a hang.

    The handler reads the sample rate from the generating thread's first item, so
    a failure before that item must arrive as a response, never as an error item.
    """
    engine = StreamVox(chunks=4, exc=GeneratorBoom("gone"), exc_after=0)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with caplog.at_level(logging.ERROR, logger="speakeasy"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    assert r.status_code == 500
    assert r.json() == {"error": {"message": "speech generation failed"}}
    assert len([rec for rec in caplog.records if rec.levelno >= logging.ERROR]) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_raising_error_mapper_leaves_the_guard_to_answer(client, monkeypatch, caplog):
    """An `on_error` mapper that raises still cannot leave the request hanging."""
    engine = StreamVox(chunks=4, exc=RuntimeError("engine boom"), exc_after=0)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    def explode(_exc, _voice_id):
        raise RuntimeError("mapper boom")

    monkeypatch.setattr(server, "_voxcpm2_stream_error", explode)

    with caplog.at_level(logging.ERROR, logger="speakeasy"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    assert r.status_code == 500
    assert r.json() == {"error": {"message": "speech generation failed"}}
    assert len([rec for rec in caplog.records if rec.levelno >= logging.ERROR]) == 1


def test_stream_leaks_no_exception_text(client, monkeypatch, caplog):
    engine = StreamVox(
        chunks=4, exc=RuntimeError("secret /path/to/weights"), exc_after=2
    )
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with caplog.at_level(logging.ERROR, logger="speakeasy"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    assert b"secret" not in r.content
    assert b"/path" not in r.content
    assert len([rec for rec in caplog.records if rec.levelno >= logging.ERROR]) == 1


def test_health_responds_during_stream(client, monkeypatch):
    gate = threading.Event()
    engine = StreamVox(gate=gate)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    result = {}

    def request():
        result["r"] = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="audio",
        )

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert _wait(lambda: engine.stream_calls, 5), "the engine never started"
        start = time.monotonic()
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert time.monotonic() - start < 2
    finally:
        gate.set()
        thread.join(10)

    assert result["r"].status_code == 200


def test_idle_unload_skipped_during_stream(client, monkeypatch):
    gate = threading.Event()
    engine = StreamVox(gate=gate)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server.settings, "tts_voxcpm_idle_seconds", 0.1)
    monkeypatch.setattr(server, "voxcpm2_last_used", 0.0)  # long idle

    result = {}

    def request():
        result["r"] = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="audio",
        )

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert _wait(lambda: engine.stream_calls, 5), "the engine never started"
        server._voxcpm2_unload_tick()  # the lock is held by the generating thread
        assert server.voxcpm2 is engine
        assert server.voxcpm2_state == "ready"
    finally:
        gate.set()
        thread.join(10)

    assert result["r"].status_code == 200


def test_disconnect_mid_stream_no_500(client, monkeypatch, caplog):
    engine = StreamVox(chunks=5, delay=0.05)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with caplog.at_level(logging.INFO, logger="speakeasy"):
        sent = _drive_disconnect(STREAM_PAYLOAD)
        assert _wait(engine.completed.is_set, 5), "the generating thread did not finish"

    assert any("client disconnected during stream" in m for m in caplog.messages)
    # The response had already started 200; no 500 is ever served.
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert starts and starts[0]["status"] == 200
    assert _wait(lambda: not server.tts_lock.locked(), 5), "the lock was not released"

    ok = _speech(client, model="voxcpm2", input="Hi", voice="default")
    assert ok.status_code == 200


def test_engine_runs_to_completion_when_client_stalls(client, monkeypatch):
    engine = StreamVox(chunks=4, delay=0.05)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    _drive_disconnect(STREAM_PAYLOAD, after_chunks=1)

    assert _wait(engine.completed.is_set, 5), "the engine did not finish"
    assert _wait(lambda: not server.tts_lock.locked(), 5), "the lock was not released"


def test_slow_client_no_stall(client, monkeypatch):
    engine = StreamVox(chunks=5, delay=0.03)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={
            "model": "voxcpm2",
            "input": "Hi",
            "voice": "default",
            "response_format": "pcm",
            "stream_format": "audio",
        },
    ) as r:
        assert r.status_code == 200
        chunks = r.iter_bytes()
        first = next(chunks)
        assert _wait(engine.completed.is_set, 5), "the engine did not finish"
        rest = first + b"".join(chunks)

    assert len(rest) == 5 * engine.chunk_samples * 2
    assert not server.tts_lock.locked()


def test_two_concurrent_streams_serialised(client, monkeypatch):
    engine = StreamVox(chunks=3, delay=0.05)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    barrier = threading.Barrier(2)
    statuses = []

    def post():
        c = TestClient(server.app)
        barrier.wait(5)
        statuses.append(
            _speech(
                c,
                model="voxcpm2",
                input="Hi",
                voice="default",
                response_format="pcm",
                stream_format="audio",
            ).status_code
        )

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)

    assert statuses == [200, 200]
    assert engine.max_in_flight == 1


def test_buffer_over_cap_mid_stream_error(client, monkeypatch, caplog):
    engine = StreamVox(chunks=100_000, seconds=4.0, zero=True)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with caplog.at_level(logging.ERROR, logger="speakeasy"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    assert r.status_code == 200
    assert _sse_events(r.content)[-1]["type"] == "error"
    # The lock is free: the next request succeeds at once.
    assert server.tts_lock.acquire(blocking=False)
    server.tts_lock.release()
    monkeypatch.setattr(server, "voxcpm2", StreamVox(chunks=1))
    assert (
        _speech(client, model="voxcpm2", input="Hi", voice="default").status_code == 200
    )
    assert len([rec for rec in caplog.records if rec.levelno >= logging.ERROR]) == 1


def test_clone_deleted_mid_stream_no_500(client, monkeypatch):
    # A clone that cannot be read at synthesis start is today's 400, no stream.
    engine = StreamVox(chunks=3, exc=FileNotFoundError("gone"), exc_after=0)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", StubVoxStore({"clone-gone": None}))

    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="clone-gone",
        response_format="pcm",
        stream_format="sse",
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

    # A clone deleted after synthesis start cannot affect the take: it completes.
    live = StreamVox(chunks=3, delay=0.02)
    monkeypatch.setattr(server, "voxcpm2", live)
    monkeypatch.setattr(server, "voxcpm2_store", StubVoxStore({"clone-live": None}))
    r2 = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="clone-live",
        response_format="pcm",
        stream_format="sse",
    )
    assert r2.status_code == 200
    assert len(_deltas(r2.content)) == 3


def test_lock_held_by_generating_thread(client, monkeypatch):
    """The generating thread takes `tts_lock`; the handler never does."""
    owners = []

    class RecordingLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            owners.append(threading.current_thread().name)
            return self

        def __exit__(self, *exc):
            self._lock.release()
            return False

        def acquire(self, blocking=True):
            return self._lock.acquire(blocking=blocking)

        def release(self):
            self._lock.release()

        def locked(self):
            return self._lock.locked()

    engine = StreamVox()
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)
    monkeypatch.setattr(server, "tts_lock", RecordingLock())

    result = {}

    def handler():
        threading.current_thread().name = "handler-thread"
        result["r"] = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    thread = threading.Thread(target=handler, name="handler-thread")
    thread.start()
    thread.join(10)

    assert result["r"].status_code == 200
    # Only the generating thread takes `tts_lock`, and it names itself
    # "speech-stream"; the handler does not, whatever thread it runs on.
    assert owners == ["speech-stream"]


def test_sse_stream_cache_control(client, vox_ready):
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )
    assert r.headers["cache-control"] == "no-store"


def test_sse_stream_401_without_key(client, monkeypatch, vox_ready):
    monkeypatch.setattr(server.settings, "api_key", "unit-test-key")
    r = _speech(
        client,
        model="voxcpm2",
        input="Hi",
        voice="default",
        response_format="pcm",
        stream_format="sse",
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid or missing API key"}


def test_stream_buffer_bounded(client, monkeypatch, caplog):
    """Over the cap: the client gets the error, the lock frees, one error line."""
    engine = StreamVox(chunks=100_000, seconds=4.0, zero=True)
    monkeypatch.setattr(server, "voxcpm2", engine)
    monkeypatch.setattr(server, "voxcpm2_state", "ready")
    monkeypatch.setattr(server, "voxcpm2_store", None)

    with caplog.at_level(logging.ERROR, logger="speakeasy"):
        r = _speech(
            client,
            model="voxcpm2",
            input="Hi",
            voice="default",
            response_format="pcm",
            stream_format="sse",
        )

    assert r.status_code == 200
    events = _sse_events(r.content)
    assert events[-1] == {
        "type": "error",
        "error": {"message": server.STREAM_ERROR_MESSAGE},
    }
    assert _wait(lambda: not server.tts_lock.locked(), 5), "the lock was not released"
    assert len([rec for rec in caplog.records if rec.levelno >= logging.ERROR]) == 1
