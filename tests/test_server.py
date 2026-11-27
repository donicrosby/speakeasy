"""Endpoint tests using a mock whisper engine — no model downloads, no GPU.

Real-engine smoke tests live in spikes/001 and are exercised against a
running container. These tests pin the API contract.
"""

import math
import struct
import tempfile
import wave

import pytest
from fastapi.testclient import TestClient

import speakeasy.server as server
from speakeasy.engines import Segment, TranscriptionResult

EXPECTED_TEXT = "And so my fellow Americans ask not what your country can do for you"


class MockWhisper:
    load_seconds = 0.01

    @property
    def name(self):
        return "whisper"

    def transcribe(self, path, language=None, vad_filter=True, word_timestamps=False):
        return TranscriptionResult(
            text=EXPECTED_TEXT,
            language="en",
            duration=11.0,
            segments=[Segment(start=0.0, end=11.0, text=EXPECTED_TEXT)],
            engine="whisper",
            elapsed_seconds=0.5,
        )


class MockMoonshine:
    load_seconds = 0.01

    @property
    def name(self):
        return "moonshine"

    def transcribe(self, path, **_):
        return TranscriptionResult(
            text=EXPECTED_TEXT,
            language="en",
            duration=5.0,
            segments=[Segment(start=0.0, end=5.0, text=EXPECTED_TEXT)],
            engine="moonshine",
            elapsed_seconds=0.1,
        )


class RecordingEngine:
    """Records every call; a refused request must never reach it."""

    load_seconds = 0.01

    def __init__(self, name):
        self.name = name
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(path)
        raise AssertionError("engine must not run for a refused request")


@pytest.fixture()
def wav_file(tmp_path):
    """3s 440Hz tone, 16kHz mono 16-bit."""
    path = tmp_path / "tone.wav"
    sr = 16000
    n = sr * 3
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n):
            v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sr))
            wf.writeframes(struct.pack("<h", v))
    return path


def _post(client, wav_file, data=None, query=""):
    with open(wav_file, "rb") as f:
        return client.post(
            "/v1/audio/transcriptions" + query,
            files={"file": ("tone.wav", f, "audio/wav")},
            data=data or {},
        )


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "whisper", MockWhisper())
    monkeypatch.setattr(server, "moonshine", MockMoonshine())
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server, "moonshine_error", None)
    monkeypatch.setattr(server.settings, "moonshine_enabled", True)
    return TestClient(server.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_models_list_has_all_ids(client):
    # Transcription ids are listed by configuration, then the TTS ids, in order.
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert ids == [
        "whisper-1",
        "large-v3-turbo",
        "moonshine-base",
        "reverb-tts",
        "voxcpm2",
    ]


def test_models_list_without_moonshine(monkeypatch, client):
    monkeypatch.setattr(server.settings, "moonshine_enabled", False)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "moonshine-base" not in ids
    assert ids[:2] == ["whisper-1", "large-v3-turbo"]


def test_models_list_dedups_when_configured_name_is_whisper_1(monkeypatch, client):
    monkeypatch.setattr(server.settings, "whisper_model", "whisper-1")
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == ["whisper-1", "moonshine-base", "reverb-tts", "voxcpm2"]


def test_transcribe_json(client, wav_file):
    r = _post(client, wav_file, {"model": "large-v3-turbo"})
    assert r.status_code == 200
    assert r.json() == {"text": EXPECTED_TEXT}
    assert r.headers["x-speakeasy-engine"] == "whisper"


def test_transcribe_verbose_json(client, wav_file):
    r = _post(
        client, wav_file, {"model": "large-v3-turbo", "response_format": "verbose_json"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == EXPECTED_TEXT
    assert body["language"] == "en"
    assert body["segments"][0]["start"] == 0.0


def test_transcribe_srt_vtt_text(client, wav_file):
    for fmt, needle in [("srt", "-->"), ("vtt", "WEBVTT"), ("text", EXPECTED_TEXT)]:
        r = _post(client, wav_file, {"model": "large-v3-turbo", "response_format": fmt})
        assert r.status_code == 200, fmt
        assert needle in r.text, fmt


def test_transcribe_bad_format(client, wav_file):
    r = _post(client, wav_file, {"model": "large-v3-turbo", "response_format": "xml"})
    assert r.status_code == 400


# --- the model field is required and must be listed ----------------------


def test_missing_model_422(client, wav_file):
    r = _post(client, wav_file)
    assert r.status_code == 422
    detail = r.json()["detail"][0]
    assert detail["type"] == "missing"
    assert detail["loc"] == ["body", "model"]


def test_empty_model_422(client, wav_file):
    r = _post(client, wav_file, {"model": ""})
    assert r.status_code == 422
    # A required Form field rejects an empty value the same way it rejects a
    # missing one; the error names the field.
    assert r.json()["detail"][0]["loc"] == ["body", "model"]


def test_whisper_1_routes_to_whisper(client, wav_file):
    r = _post(client, wav_file, {"model": "whisper-1"})
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "whisper"


def test_configured_name_routes_to_whisper(client, wav_file):
    r = _post(client, wav_file, {"model": "large-v3-turbo"})
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "whisper"


def test_moonshine_name_routes_to_moonshine(client, wav_file):
    r = _post(client, wav_file, {"model": "moonshine-base"})
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "moonshine"


def test_unlisted_model_404(client, wav_file):
    r = _post(client, wav_file, {"model": "whisper-2"})
    assert r.status_code == 404
    assert r.json() == {
        "error": {
            "message": (
                "unknown model 'whisper-2'. Known: "
                "whisper-1, large-v3-turbo, moonshine-base"
            )
        }
    }


def test_404_echo_bound(client, wav_file):
    r = _post(client, wav_file, {"model": "x" * 200})
    assert r.status_code == 404
    message = r.json()["error"]["message"]
    assert "x" * 64 + "..." in message
    assert "x" * 65 not in message


def test_fast_with_whisper_id_is_400(client, wav_file):
    r = _post(client, wav_file, {"model": "whisper-1"}, query="?fast=true")
    assert r.status_code == 400
    assert r.json() == {"error": {"message": "fast=true requires model=moonshine-base"}}


def test_fast_with_moonshine_id_is_fine(client, wav_file):
    r = _post(client, wav_file, {"model": "moonshine-base"}, query="?fast=true")
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "moonshine"


def test_moonshine_non_english_is_400(client, wav_file):
    r = _post(client, wav_file, {"model": "moonshine-base", "language": "de"})
    assert r.status_code == 400
    assert r.json() == {
        "error": {"message": "moonshine supports only English (language unset or 'en')"}
    }


# --- availability mirrors the listing ------------------------------------


def test_whisper_unavailable_503(monkeypatch, wav_file):
    monkeypatch.setattr(server, "whisper", None)
    monkeypatch.setattr(server, "startup_error", "whisper failed to load: boom")
    monkeypatch.setattr(server, "moonshine", MockMoonshine())
    monkeypatch.setattr(server, "moonshine_error", None)
    monkeypatch.setattr(server.settings, "moonshine_enabled", True)
    c = TestClient(server.app)
    r = _post(c, wav_file, {"model": "whisper-1"})
    assert r.status_code == 503
    assert r.json() == {"error": {"message": "whisper failed to load: boom"}}


def test_moonshine_unavailable_503(monkeypatch, wav_file):
    monkeypatch.setattr(server, "whisper", MockWhisper())
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server, "moonshine", None)
    monkeypatch.setattr(server, "moonshine_error", "moonshine failed to load: OOM")
    monkeypatch.setattr(server.settings, "moonshine_enabled", True)
    c = TestClient(server.app)
    r = _post(c, wav_file, {"model": "moonshine-base"})
    assert r.status_code == 503
    assert r.json() == {"error": {"message": "moonshine failed to load: OOM"}}


def test_moonshine_still_works_when_whisper_failed(monkeypatch, wav_file):
    monkeypatch.setattr(server, "whisper", None)
    monkeypatch.setattr(server, "startup_error", "whisper failed to load: boom")
    monkeypatch.setattr(server, "moonshine", MockMoonshine())
    monkeypatch.setattr(server, "moonshine_error", None)
    monkeypatch.setattr(server.settings, "moonshine_enabled", True)
    c = TestClient(server.app)
    r = _post(c, wav_file, {"model": "moonshine-base"})
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "moonshine"


@pytest.mark.parametrize("whisper_ok", [True, False])
@pytest.mark.parametrize("moonshine_state", ["loaded", "failed", "disabled"])
def test_typo_model_is_404_on_all_states(
    monkeypatch, wav_file, whisper_ok, moonshine_state
):
    monkeypatch.setattr(server, "whisper", MockWhisper() if whisper_ok else None)
    monkeypatch.setattr(
        server, "startup_error", None if whisper_ok else "whisper failed to load: boom"
    )
    if moonshine_state == "loaded":
        monkeypatch.setattr(server, "moonshine", MockMoonshine())
        monkeypatch.setattr(server, "moonshine_error", None)
    elif moonshine_state == "failed":
        monkeypatch.setattr(server, "moonshine", None)
        monkeypatch.setattr(server, "moonshine_error", "moonshine failed to load: OOM")
    else:
        monkeypatch.setattr(server, "moonshine", None)
        monkeypatch.setattr(server, "moonshine_error", None)
    monkeypatch.setattr(
        server.settings, "moonshine_enabled", moonshine_state != "disabled"
    )
    c = TestClient(server.app)
    r = _post(c, wav_file, {"model": "whisper-2"})
    assert r.status_code == 404


def test_422_404_before_engine(monkeypatch, wav_file, tmp_path):
    """The 422 and 404 run before any engine state, lock or temp file."""
    whisper_engine = RecordingEngine("whisper")
    moonshine_engine = RecordingEngine("moonshine")
    monkeypatch.setattr(server, "whisper", whisper_engine)
    monkeypatch.setattr(server, "moonshine", moonshine_engine)
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server, "moonshine_error", None)
    monkeypatch.setattr(server.settings, "moonshine_enabled", True)
    # A private, empty directory: any route-written temp file would land here.
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    c = TestClient(server.app)

    assert _post(c, wav_file).status_code == 422
    assert _post(c, wav_file, {"model": "whisper-2"}).status_code == 404

    assert whisper_engine.calls == []
    assert moonshine_engine.calls == []
    assert list(spool.iterdir()) == []
