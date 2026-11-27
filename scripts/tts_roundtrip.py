#!/usr/bin/env python3
"""End-to-end TTS round trip against a running speakeasy server.

Waits for TTS readiness, speaks a preset line, transcribes it back, clones a
voice from that take, speaks with the clone, measures ``speed`` 0.5 / 1.0 / 2.0,
then deletes the clone. One JSON line is printed per step; the exit code is 0
only when every step passes.

Intended to run inside the GPU image (it is not installed) against the server
it ships with::

    python3 scripts/tts_roundtrip.py --base-url http://localhost:8080
    SPEAKEASY_API_KEY=... python3 scripts/tts_roundtrip.py

The base URL and the API key come from arguments or the environment
(``SPEAKEASY_BASE_URL`` and ``SPEAKEASY_API_KEY``).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

LINE = (
    "The quick brown fox jumps over the lazy dog while the evening light "
    "fades across the calm quiet harbour tonight."
)
PRESET = "en_male_narrator_1"
CLONE_ID = "roundtrip"
SIMILARITY_MIN = 0.8
SPEEDS = (0.5, 1.0, 2.0)
READY_TIMEOUT = 120.0
REQUEST_TIMEOUT = 300.0


def _emit(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _headers(message) -> dict:
    """Lower-cased header map: HTTP header names are case-insensitive."""
    return {k.lower(): v for k, v in message.items()}


def _error_message(body: bytes) -> str:
    try:
        return json.loads(body)["error"]["message"]
    except Exception:  # noqa: BLE001 — any other shape means "unparseable error"
        return body[:200].decode("utf-8", "replace")


def _request(
    base,
    path,
    key,
    *,
    method="GET",
    data=None,
    content_type=None,
    timeout=REQUEST_TIMEOUT,
):
    req = urllib.request.Request(base + path, data=data, method=method)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), _headers(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), _headers(e.headers)


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode() + b"\r\n"
    for name, (filename, data, ctype) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        ).encode()
        body += f"Content-Type: {ctype}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _similarity(expected: str, actual: str) -> float:
    return difflib.SequenceMatcher(None, _norm(expected), _norm(actual)).ratio()


def wait_ready(base, key, timeout=READY_TIMEOUT) -> bool:
    """Poll GET /health/ready until `tts` is "ready" (or time out)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, body, _ = _request(base, "/health/ready", key, timeout=10.0)
            if status == 200:
                state = json.loads(body).get("tts")
                if state == "ready":
                    _emit({"step": "wait_ready", "ok": True, "tts": state})
                    return True
        except Exception:  # noqa: BLE001 — server may still be starting
            pass
        if time.monotonic() >= deadline:
            _emit(
                {
                    "step": "wait_ready",
                    "ok": False,
                    "error": f"tts not ready within {timeout:g}s",
                }
            )
            return False
        time.sleep(2.0)


def speak(base, key, text, voice, out_path, speed, step) -> dict | None:
    """POST /v1/audio/speech; write the wav to `out_path` when given."""
    payload = json.dumps(
        {"input": text, "voice": voice, "response_format": "wav", "speed": speed}
    ).encode()
    status, body, headers = _request(
        base,
        "/v1/audio/speech",
        key,
        method="POST",
        data=payload,
        content_type="application/json",
    )
    if status != 200:
        _emit(
            {"step": step, "ok": False, "status": status, "error": _error_message(body)}
        )
        return None
    if out_path:
        with open(out_path, "wb") as f:
            f.write(body)
    record = {
        "step": step,
        "ok": True,
        "status": status,
        "voice": voice,
        "speed": speed,
        "audio_seconds": _as_float(headers.get("x-speakeasy-audio-duration")),
        "elapsed_seconds": _as_float(headers.get("x-speakeasy-elapsed")),
        "rtf": _as_float(headers.get("x-speakeasy-rtf")),
    }
    if out_path:
        record["path"] = out_path
    _emit(record)
    return record


def transcribe(base, key, path, step, expected=None) -> dict | None:
    """POST /v1/audio/transcriptions; similarity to `expected` when given."""
    with open(path, "rb") as f:
        data = f.read()
    body, ctype = _multipart(
        {"response_format": "json"},
        {"file": (os.path.basename(path), data, "audio/wav")},
    )
    status, resp, _ = _request(
        base,
        "/v1/audio/transcriptions",
        key,
        method="POST",
        data=body,
        content_type=ctype,
    )
    if status != 200:
        _emit(
            {"step": step, "ok": False, "status": status, "error": _error_message(resp)}
        )
        return None
    text = json.loads(resp).get("text", "")
    record = {"step": step, "ok": True, "status": status, "text": text}
    if expected is not None:
        similarity = _similarity(expected, text)
        record["similarity"] = round(similarity, 3)
        record["ok"] = similarity >= SIMILARITY_MIN
    _emit(record)
    return record


def clone(base, key, path, text, step) -> bool:
    """POST /v1/audio/voices — 201 expected."""
    with open(path, "rb") as f:
        data = f.read()
    body, ctype = _multipart(
        {"text": text, "language": "en", "voice_id": CLONE_ID},
        {"file": (os.path.basename(path), data, "audio/wav")},
    )
    status, resp, _ = _request(
        base, "/v1/audio/voices", key, method="POST", data=body, content_type=ctype
    )
    ok = status == 201
    record = {"step": step, "ok": ok, "status": status}
    if ok:
        record.update(json.loads(resp))
    else:
        record["error"] = _error_message(resp)
    _emit(record)
    return ok


def delete_clone(base, key, step) -> bool:
    """DELETE /v1/audio/voices/<CLONE_ID> — `deleted: true` expected."""
    status, resp, _ = _request(
        base, f"/v1/audio/voices/{CLONE_ID}", key, method="DELETE"
    )
    ok = status == 200 and json.loads(resp).get("deleted") is True
    record = {"step": step, "ok": ok, "status": status}
    if not ok:
        record["error"] = _error_message(resp)
    _emit(record)
    return ok


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="End-to-end TTS round trip against a running speakeasy server."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPEAKEASY_BASE_URL", "http://localhost:8080"),
        help="server root (env: SPEAKEASY_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SPEAKEASY_API_KEY", ""),
        help="bearer token (env: SPEAKEASY_API_KEY)",
    )
    parser.add_argument("--out-dir", default=".", help="where t1.wav/t2.wav go")
    parser.add_argument(
        "--timeout",
        type=float,
        default=READY_TIMEOUT,
        help="readiness timeout, seconds",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base = args.base_url.rstrip("/")
    key = args.api_key
    os.makedirs(args.out_dir, exist_ok=True)
    t1 = os.path.join(args.out_dir, "t1.wav")
    t2 = os.path.join(args.out_dir, "t2.wav")

    failures: list[str] = []

    if not wait_ready(base, key, args.timeout):
        return 1

    if speak(base, key, LINE, PRESET, t1, 1.0, "preset_speech") is None:
        return 1

    first = transcribe(base, key, t1, "preset_transcribe", expected=LINE)
    if first is None or not first["ok"]:
        failures.append("preset_transcribe")
    transcript = (first or {}).get("text") or LINE

    if not clone(base, key, t1, transcript, "clone"):
        failures.append("clone")

    if speak(base, key, transcript, CLONE_ID, t2, 1.0, "clone_speech") is None:
        failures.append("clone_speech")
    else:
        cloned = transcribe(base, key, t2, "clone_transcribe", expected=transcript)
        if cloned is None or not cloned["ok"]:
            failures.append("clone_transcribe")

    for speed in SPEEDS:
        tag = f"{speed:g}"
        path = os.path.join(args.out_dir, f"speed_{tag}.wav")
        if speak(base, key, LINE, PRESET, path, speed, f"speed_{tag}") is None:
            failures.append(f"speed_{tag}")
            continue
        measured = transcribe(base, key, path, f"speed_{tag}_transcribe", expected=LINE)
        if measured is None or not measured["ok"]:
            failures.append(f"speed_{tag}_transcribe")

    if not delete_clone(base, key, "delete"):
        failures.append("delete")

    if failures:
        _emit({"step": "summary", "ok": False, "failed": failures})
        return 1
    _emit({"step": "summary", "ok": True, "steps": "all passed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
