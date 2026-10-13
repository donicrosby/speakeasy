"""Endpoint tests using a mock whisper engine — no model downloads, no GPU.

Real-engine smoke tests live in spikes/001 and are exercised against a
running container. These tests pin the API contract.
"""

import math
import struct
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


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(server, "whisper", MockWhisper())
    monkeypatch.setattr(server, "moonshine", MockMoonshine())
    monkeypatch.setattr(server, "startup_error", None)
    return TestClient(server.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_models_list(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert ids  # at least one model


def test_transcribe_json(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("tone.wav", f, "audio/wav")},
            data={"model": "large-v3-turbo"},
        )
    assert r.status_code == 200
    assert r.json() == {"text": EXPECTED_TEXT}
    assert r.headers["x-speakeasy-engine"] == "whisper"


def test_transcribe_verbose_json(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("tone.wav", f, "audio/wav")},
            data={"model": "large-v3-turbo", "response_format": "verbose_json"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == EXPECTED_TEXT
    assert body["language"] == "en"
    assert body["segments"][0]["start"] == 0.0


def test_transcribe_srt_vtt_text(client, wav_file):
    for fmt, needle in [
        ("srt", "-->"),
        ("vtt", "WEBVTT"),
        ("text", EXPECTED_TEXT),
    ]:
        with open(wav_file, "rb") as f:
            r = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("tone.wav", f, "audio/wav")},
                data={"model": "large-v3-turbo", "response_format": fmt},
            )
        assert r.status_code == 200, fmt
        assert needle in r.text, fmt


def test_transcribe_bad_format(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("tone.wav", f, "audio/wav")},
            data={"response_format": "xml"},
        )
    assert r.status_code == 400


def test_fast_path_routes_to_moonshine(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/v1/audio/transcriptions?fast=true",
            files={"file": ("tone.wav", f, "audio/wav")},
        )
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "moonshine"


def test_auto_route_short_audio_to_moonshine(client, wav_file):
    # 3s tone, no model specified → auto-route to moonshine
    with open(wav_file, "rb") as f:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("tone.wav", f, "audio/wav")},
        )
    assert r.status_code == 200
    assert r.headers["x-speakeasy-engine"] == "moonshine"


def test_explicit_whisper_model_beats_auto_route(client, wav_file):
    with open(wav_file, "rb") as f:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("tone.wav", f, "audio/wav")},
            data={"model": "large-v3-turbo"},
        )
    assert r.headers["x-speakeasy-engine"] == "whisper"


def test_503_when_models_not_loaded(monkeypatch, wav_file):
    monkeypatch.setattr(server, "whisper", None)
    monkeypatch.setattr(server, "startup_error", "whisper failed to load: boom")
    c = TestClient(server.app)
    with open(wav_file, "rb") as f:
        r = c.post(
            "/v1/audio/transcriptions", files={"file": ("t.wav", f, "audio/wav")}
        )
    assert r.status_code == 503
