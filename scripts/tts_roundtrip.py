#!/usr/bin/env python3
"""End-to-end TTS round trip against a running speakeasy server.

Reverb-TTS (the default engine, ``--engine reverb-tts``): waits for TTS
readiness, speaks a preset line, transcribes it back, clones a voice from that
take, speaks with the clone, measures ``speed`` 0.5 / 1.0 / 2.0, then deletes
the clone.

VoxCPM2 (``--engine voxcpm2``): waits for the engine to report ready on
``GET /health/ready``, then walks its features — plain speech, a designed
voice, a clone stored without a transcript, a styled clone, a
transcript-assisted clone (the library's combined mode), the bad-take path and
the ``cfg`` bounds — and, only when ``--idle-wait`` is given, the idle unload
and the reload that follows it.

One JSON line is printed per step; the exit code is 0 only when every step
passes. The script uses the standard library only and is not installed.

Intended to run inside the GPU image (it is not installed) against the server
it ships with::

    python3 scripts/tts_roundtrip.py --base-url http://localhost:8080
    SPEAKEASY_API_KEY=... python3 scripts/tts_roundtrip.py --engine voxcpm2

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

REVERB_ENGINE = "reverb-tts"
VOXCPM2_ENGINE = "voxcpm2"
DESIGN_TEXT = "A young woman, gentle and sweet voice"
STYLE_TEXT = "slightly faster, cheerful tone"
VOXCPM2_CLONE_ID = "roundtrip-vox"
VOXCPM2_TRANSCRIPT_CLONE_ID = "roundtrip-vox-t"
# A take a short input may or may not push over the bad-take line; the step
# accepts either outcome (see `probe_bad_take`).
BAD_TAKE_INPUT = "a"
BAD_TAKE_MARKER = "could not generate a good take"
# (cfg value, expected HTTP status): the two in-range values take, the two out
# of range are rejected by request validation.
CFG_BOUNDS = ((1.0, 200), (5.0, 200), (0.5, 422), (6.0, 422))


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


def _speech_payload(
    text,
    voice,
    speed,
    *,
    model=REVERB_ENGINE,
    cfg=None,
    design=None,
    style=None,
    response_format="wav",
) -> bytes:
    """The speech JSON body; the VoxCPM2-only fields are added only when set."""
    payload = {
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed,
    }
    for field, value in (
        ("model", model),
        ("cfg", cfg),
        ("design", design),
        ("style", style),
    ):
        if value is not None:
            payload[field] = value
    return json.dumps(payload).encode()


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


def wait_ready_voxcpm2(base, key, timeout, *, model, cfg) -> bool:
    """Poll GET /health/ready until `voxcpm2` is "ready" (or time out).

    VoxCPM2 loads lazily on the first request, so a `not_loaded` state is woken
    with one speech request (which is expected to answer 503 and start the
    load); the wait then continues until the engine is ready. A cold first use
    also downloads about 4.7 GB of weights, so this can take minutes.
    """
    deadline = time.monotonic() + timeout
    triggered = False
    while True:
        try:
            status, body, _ = _request(base, "/health/ready", key, timeout=10.0)
            if status == 200:
                state = json.loads(body).get("voxcpm2")
                if state == "ready":
                    _emit({"step": "wait_ready", "ok": True, "voxcpm2": state})
                    return True
                if state in ("error", "disabled"):
                    _emit({"step": "wait_ready", "ok": False, "voxcpm2": state})
                    return False
                if state == "not_loaded" and not triggered:
                    triggered = True
                    _request(
                        base,
                        "/v1/audio/speech",
                        key,
                        method="POST",
                        data=_speech_payload(
                            LINE, "default", 1.0, model=model, cfg=cfg
                        ),
                        content_type="application/json",
                    )
        except Exception:  # noqa: BLE001 — server may still be starting
            pass
        if time.monotonic() >= deadline:
            _emit(
                {
                    "step": "wait_ready",
                    "ok": False,
                    "error": f"voxcpm2 not ready within {timeout:g}s",
                }
            )
            return False
        time.sleep(2.0)


def speak(
    base,
    key,
    text,
    voice,
    out_path,
    speed,
    step,
    *,
    model=REVERB_ENGINE,
    cfg=None,
    design=None,
    style=None,
) -> dict | None:
    """POST /v1/audio/speech; write the wav to `out_path` when given."""
    status, body, headers = _request(
        base,
        "/v1/audio/speech",
        key,
        method="POST",
        data=_speech_payload(
            text, voice, speed, model=model, cfg=cfg, design=design, style=style
        ),
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
        "engine": headers.get("x-speakeasy-engine"),
        "voice": voice,
        "speed": speed,
        "audio_seconds": _as_float(headers.get("x-speakeasy-audio-duration")),
        "elapsed_seconds": _as_float(headers.get("x-speakeasy-elapsed")),
        "rtf": _as_float(headers.get("x-speakeasy-rtf")),
    }
    if cfg is not None:
        record["cfg"] = cfg
    if out_path:
        record["path"] = out_path
    _emit(record)
    return record


def probe_speech(base, key, text, voice, step, *, model, cfg, expect) -> bool:
    """One speech request checked against an expected status; no audio is kept."""
    status, body, _ = _request(
        base,
        "/v1/audio/speech",
        key,
        method="POST",
        data=_speech_payload(text, voice, 1.0, model=model, cfg=cfg),
        content_type="application/json",
    )
    ok = status == expect
    record = {"step": step, "ok": ok, "status": status, "cfg": cfg, "expected": expect}
    if not ok:
        record["error"] = _error_message(body)
    _emit(record)
    return ok


def probe_bad_take(base, key, step, *, model, cfg) -> bool:
    """Exercise the bad-take path.

    A bad take cannot be forced deterministically against a real model, so the
    step passes on either a good take (200) or the server's bad-take 503, and
    fails on anything else (e.g. a 500).
    """
    status, body, _ = _request(
        base,
        "/v1/audio/speech",
        key,
        method="POST",
        data=_speech_payload(BAD_TAKE_INPUT, "default", 1.0, model=model, cfg=cfg),
        content_type="application/json",
    )
    message = _error_message(body) if status != 200 else None
    ok = status == 200 or (status == 503 and BAD_TAKE_MARKER in (message or ""))
    record = {
        "step": step,
        "ok": ok,
        "status": status,
        "outcome": "good_take" if status == 200 else "bad_take",
    }
    if not ok:
        record["error"] = message
    _emit(record)
    return ok


def probe_idle_unload(base, key, args, *, model, cfg) -> bool:
    """Wait for the idle unload, then confirm the next request reloads.

    Skipped unless ``--idle-wait`` is given: the configured idle window
    (``SPEAKEASY_TTS_VOXCPM_IDLE_SECONDS``) can be minutes long, so a default
    run never sits for it.
    """
    step = "idle_unload"
    if args.idle_wait <= 0:
        _emit(
            {
                "step": step,
                "ok": True,
                "skipped": True,
                "note": "pass --idle-wait SECONDS to exercise the idle unload",
            }
        )
        return True
    deadline = time.monotonic() + args.idle_wait
    while time.monotonic() < deadline:
        try:
            status, body, _ = _request(base, "/health/ready", key, timeout=10.0)
            if status == 200 and json.loads(body).get("voxcpm2") == "not_loaded":
                break
        except Exception:  # noqa: BLE001 — keep polling until the deadline
            pass
        time.sleep(2.0)
    else:
        _emit(
            {
                "step": step,
                "ok": False,
                "error": f"voxcpm2 did not unload within {args.idle_wait:g}s",
            }
        )
        return False
    # The next request must answer the not_loaded 503 and start the load again.
    status, body, _ = _request(
        base,
        "/v1/audio/speech",
        key,
        method="POST",
        data=_speech_payload(LINE, "default", 1.0, model=model, cfg=cfg),
        content_type="application/json",
    )
    ok = status == 503
    record = {"step": step, "ok": ok, "status": status, "reloaded": ok}
    if not ok:
        record["error"] = _error_message(body)
    _emit(record)
    return ok


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
    """POST /v1/audio/voices (Reverb) — 201 expected."""
    with open(path, "rb") as f:
        data = f.read()
    body, ctype = _multipart(
        {
            "model": REVERB_ENGINE,
            "text": text,
            "language": "en",
            "voice_id": CLONE_ID,
        },
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


def clone_voxcpm2(base, key, path, text, voice_id, step) -> bool:
    """POST /v1/audio/voices with model=voxcpm2 — 201 expected.

    `text` (the reference's transcript) is optional for VoxCPM2 and is sent
    only when given.
    """
    with open(path, "rb") as f:
        data = f.read()
    fields = {"model": VOXCPM2_ENGINE, "voice_id": voice_id}
    if text is not None:
        fields["text"] = text
    body, ctype = _multipart(
        fields, {"file": (os.path.basename(path), data, "audio/wav")}
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


def delete_clone(base, key, step, voice_id=CLONE_ID) -> bool:
    """DELETE /v1/audio/voices/<voice_id> — `deleted: true` expected."""
    status, resp, _ = _request(
        base, f"/v1/audio/voices/{voice_id}", key, method="DELETE"
    )
    ok = status == 200 and json.loads(resp).get("deleted") is True
    record = {"step": step, "ok": ok, "status": status}
    if ok:
        record["engine"] = json.loads(resp).get("engine")
    else:
        record["error"] = _error_message(resp)
    _emit(record)
    return ok


def reverb_roundtrip(base, key, args) -> list[str]:
    """The original Reverb-TTS round trip."""
    failures: list[str] = []
    t1 = os.path.join(args.out_dir, "t1.wav")
    t2 = os.path.join(args.out_dir, "t2.wav")

    if not wait_ready(base, key, args.timeout):
        return ["wait_ready"]

    if speak(base, key, LINE, PRESET, t1, 1.0, "preset_speech") is None:
        return ["preset_speech"]

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

    return failures


def voxcpm2_roundtrip(base, key, args) -> list[str]:
    """The VoxCPM2 round trip: its features, then the cfg and lifecycle edges."""
    failures: list[str] = []
    cfg = args.cfg

    if not wait_ready_voxcpm2(base, key, args.timeout, model=VOXCPM2_ENGINE, cfg=cfg):
        return ["wait_ready"]

    # Plain speech doubles as the reference clip for the clone steps.
    plain_path = os.path.join(args.out_dir, "voxcpm2_plain.wav")
    plain = speak(
        base,
        key,
        LINE,
        "default",
        plain_path,
        1.0,
        "plain_speech",
        model=VOXCPM2_ENGINE,
        cfg=cfg,
    )
    if plain is None:
        return ["plain_speech"]  # without a take there is no reference to clone

    if (
        speak(
            base,
            key,
            LINE,
            "default",
            os.path.join(args.out_dir, "voxcpm2_design.wav"),
            1.0,
            "design_speech",
            model=VOXCPM2_ENGINE,
            cfg=cfg,
            design=DESIGN_TEXT,
        )
        is None
    ):
        failures.append("design_speech")

    clone_no_transcript = clone_voxcpm2(
        base, key, plain_path, None, VOXCPM2_CLONE_ID, "clone_no_transcript"
    )
    if clone_no_transcript:
        if (
            speak(
                base,
                key,
                LINE,
                VOXCPM2_CLONE_ID,
                os.path.join(args.out_dir, "voxcpm2_clone.wav"),
                1.0,
                "clone_speech",
                model=VOXCPM2_ENGINE,
                cfg=cfg,
            )
            is None
        ):
            failures.append("clone_speech")
        if (
            speak(
                base,
                key,
                LINE,
                VOXCPM2_CLONE_ID,
                os.path.join(args.out_dir, "voxcpm2_styled.wav"),
                1.0,
                "styled_clone_speech",
                model=VOXCPM2_ENGINE,
                cfg=cfg,
                style=STYLE_TEXT,
            )
            is None
        ):
            failures.append("styled_clone_speech")
    else:
        failures.append("clone_no_transcript")

    clone_with_transcript = clone_voxcpm2(
        base,
        key,
        plain_path,
        LINE,
        VOXCPM2_TRANSCRIPT_CLONE_ID,
        "clone_with_transcript",
    )
    if clone_with_transcript:
        if (
            speak(
                base,
                key,
                LINE,
                VOXCPM2_TRANSCRIPT_CLONE_ID,
                os.path.join(args.out_dir, "voxcpm2_combined.wav"),
                1.0,
                "transcript_assisted_speech",
                model=VOXCPM2_ENGINE,
                cfg=cfg,
            )
            is None
        ):
            failures.append("transcript_assisted_speech")
    else:
        failures.append("clone_with_transcript")

    if not probe_bad_take(base, key, "bad_take", model=VOXCPM2_ENGINE, cfg=cfg):
        failures.append("bad_take")

    for value, expected in CFG_BOUNDS:
        if not probe_speech(
            base,
            key,
            LINE,
            "default",
            f"cfg_{value:g}",
            model=VOXCPM2_ENGINE,
            cfg=value,
            expect=expected,
        ):
            failures.append(f"cfg_{value:g}")

    if not probe_idle_unload(base, key, args, model=VOXCPM2_ENGINE, cfg=cfg):
        failures.append("idle_unload")

    if clone_no_transcript and not delete_clone(
        base, key, "delete_clone", VOXCPM2_CLONE_ID
    ):
        failures.append("delete_clone")
    if clone_with_transcript and not delete_clone(
        base, key, "delete_transcript_clone", VOXCPM2_TRANSCRIPT_CLONE_ID
    ):
        failures.append("delete_transcript_clone")

    return failures


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
    parser.add_argument("--out-dir", default=".", help="where the audio files go")
    parser.add_argument(
        "--timeout",
        type=float,
        default=READY_TIMEOUT,
        help="readiness timeout, seconds",
    )
    parser.add_argument(
        "--engine",
        choices=(REVERB_ENGINE, VOXCPM2_ENGINE),
        default=REVERB_ENGINE,
        help="TTS engine to exercise (default: reverb-tts)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help="VoxCPM2 guidance strength, 1.0-5.0 (default: 2.0)",
    )
    parser.add_argument(
        "--idle-wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "with --engine voxcpm2, wait up to SECONDS for the idle unload "
            "before probing the reload (0 = skip; the engine unloads after "
            "SPEAKEASY_TTS_VOXCPM_IDLE_SECONDS idle)"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    base = args.base_url.rstrip("/")
    key = args.api_key
    os.makedirs(args.out_dir, exist_ok=True)

    if args.engine == VOXCPM2_ENGINE:
        failures = voxcpm2_roundtrip(base, key, args)
    else:
        failures = reverb_roundtrip(base, key, args)

    if failures:
        _emit({"step": "summary", "ok": False, "failed": failures})
        return 1
    _emit({"step": "summary", "ok": True, "steps": "all passed"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
