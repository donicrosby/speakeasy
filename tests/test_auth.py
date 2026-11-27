"""SPEAKEASY_API_KEY: static bearer-token auth on the /v1 endpoints.

Pins the contract documented in the README: auth is off when the key is
unset; when it is set, `/v1/*` demands `Authorization: Bearer <key>` and
answers 401 otherwise, while the health probes stay open.
"""

import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

import speakeasy.server as server
from speakeasy.engines import Segment, TranscriptionResult

KEY = "unit-test-key"


class StubWhisper:
    load_seconds = 0.0

    @property
    def name(self):
        return "whisper"

    def transcribe(self, path, language=None, vad_filter=True, word_timestamps=False):
        return TranscriptionResult(
            text="stub",
            language="en",
            duration=1.0,
            segments=[Segment(start=0.0, end=1.0, text="stub")],
            engine="whisper",
            elapsed_seconds=0.0,
        )


@pytest.fixture()
def wav_file(tmp_path):
    """1s 440Hz tone, 16kHz mono 16-bit."""
    path = tmp_path / "tone.wav"
    sr = 16000
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr)))
                for i in range(sr)
            )
        )
    return path


@pytest.fixture()
def client(monkeypatch):
    # The lifespan is not run by TestClient unless entered as a context
    # manager, so the engines are stubbed the same way test_server.py does it.
    monkeypatch.setattr(server, "whisper", StubWhisper())
    monkeypatch.setattr(server, "startup_error", None)
    monkeypatch.setattr(server.settings, "api_key", KEY)
    return TestClient(server.app)


AUTH = {"Authorization": f"Bearer {KEY}"}


def test_auth_is_off_when_the_key_is_unset(monkeypatch, wav_file):
    monkeypatch.setattr(server, "whisper", StubWhisper())
    monkeypatch.setattr(server.settings, "api_key", "")
    c = TestClient(server.app)
    assert c.get("/v1/models").status_code == 200
    files = {"file": ("tone.wav", wav_file.read_bytes(), "audio/wav")}
    assert c.post("/v1/audio/transcriptions", files=files).status_code == 200


def test_models_without_a_key_is_401(client):
    r = client.get("/v1/models")
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid or missing API key"}


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer wrong-key"},
        {"Authorization": f"Basic {KEY}"},  # scheme must be Bearer
        {"Authorization": KEY},  # no scheme
        {"X-Api-Key": KEY},  # not a header the server reads
    ],
)
def test_models_with_a_bad_key_is_401(client, headers):
    assert client.get("/v1/models", headers=headers).status_code == 401


def test_models_with_the_right_key_is_200(client):
    r = client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_transcription_demands_the_key(client, wav_file):
    files = {"file": ("tone.wav", wav_file.read_bytes(), "audio/wav")}
    assert client.post("/v1/audio/transcriptions", files=files).status_code == 401
    files["file"] = ("tone.wav", wav_file.read_bytes(), "audio/wav")
    r = client.post("/v1/audio/transcriptions", files=files, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["text"] == "stub"


def test_health_probes_stay_open(client):
    assert client.get("/health").status_code == 200
    assert client.get("/health/ready").status_code == 200
