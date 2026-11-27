"""Speakeasy — OpenAI-compatible audio transcription server.

Endpoints:
    POST /v1/audio/transcriptions  — OpenAI-compatible transcription
    POST /v1/audio/speech          — TTS synthesis (Reverb), OpenAI-compatible
    GET  /v1/audio/voices          — list preset and cloned voices (TTS)
    POST /v1/audio/voices          — create a voice clone from a reference clip (TTS)
    DELETE /v1/audio/voices/{id}   — delete a saved voice clone (TTS)
    GET  /v1/models                — list loaded models
    GET  /health                   — liveness
    GET  /health/ready             — readiness (models loaded)
    GET  /                         — static web UI (assets under /static/*)
"""

from __future__ import annotations

import io
import logging
import os
import secrets
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, resolve_device_compute
from .engines import (
    BadTake,
    MoonshineEngine,
    ReverbEngine,
    VoxCPM2Engine,
    WhisperEngine,
    probe_duration,
)
from .responses import FORMATS
from .voices import ReferenceVoiceStore, VoiceStore, check_voice_id

log = logging.getLogger("speakeasy")

settings = Settings()

_bearer = HTTPBearer(auto_error=False)


def require_auth(
    creds: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Reject requests when SPEAKEASY_API_KEY is set and the token doesn't match."""
    if not settings.api_key:
        return  # auth disabled
    # Constant-time compare: a plain != leaks the token's matching prefix
    # length through response timing.
    if creds is None or not secrets.compare_digest(
        creds.credentials.encode(), settings.api_key.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


if settings.hf_home:
    os.environ["HF_HOME"] = settings.hf_home

whisper: WhisperEngine | None = None
moonshine: MoonshineEngine | None = None
resolved_device = "unknown"
startup_error: str | None = None

# TTS (Reverb) — loaded on a background thread after Whisper so readiness never
# waits on it. `disabled | loading | ready | error`; `tts_error` is logged only,
# never returned.
tts: ReverbEngine | None = None
voices_store: VoiceStore | None = None
tts_state = "disabled"
tts_error: str | None = None

# VoxCPM2 — a second engine beside Reverb, loaded on the first request and
# unloaded after an idle period. One `tts_lock` serialises inference across
# both engines; `voxcpm2_state_lock` is a small dedicated lock for the
# single-flight load check-and-set. State is `disabled | not_loaded | loading |
# ready | error`; `voxcpm2_error` is logged only, never returned.
tts_lock = threading.Lock()
voxcpm2_state_lock = threading.Lock()
voxcpm2_create_lock = threading.Lock()  # serialises clone-create (check through save)
voxcpm2: VoxCPM2Engine | None = None
voxcpm2_store: ReferenceVoiceStore | None = None
voxcpm2_state = "disabled"
voxcpm2_error: str | None = None
voxcpm2_last_used = 0.0

VOXCPM2_TIMER_INTERVAL = 15.0  # seconds between idle-unload ticks


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper, moonshine, resolved_device, startup_error

    device, compute = resolve_device_compute(settings)
    resolved_device = device
    log.info("device=%s compute=%s", device, compute)

    try:
        whisper = WhisperEngine(
            settings.whisper_model, device, compute, settings.beam_size
        )
        log.info(
            "whisper loaded in %.1fs (%s)", whisper.load_seconds, settings.whisper_model
        )
    except Exception as e:
        startup_error = f"whisper failed to load: {e}"
        log.exception(startup_error)

    if settings.moonshine_enabled:
        try:
            moonshine = MoonshineEngine(settings.moonshine_model)
            log.info(
                "moonshine loaded in %.1fs (%s)",
                moonshine.load_seconds,
                settings.moonshine_model,
            )
        except Exception as e:
            # Fast path is optional — log and continue
            log.warning("moonshine failed to load, fast path disabled: %s", e)
            moonshine = None

    start_tts()
    start_voxcpm2()

    yield


def start_tts() -> None:
    """Load Reverb off the startup path: readiness is Whisper's, not TTS's."""
    global tts_state
    if tts_state in ("loading", "ready") or tts is not None:
        return  # a load is already in flight (or done); never spawn a second
    if not settings.tts_enabled:
        log.info("tts disabled: SPEAKEASY_TTS_ENABLED=false")
        return
    try:
        import reverb_tts  # noqa: F401 — absent without the `tts` extra
        import torch  # noqa: F401
    except ImportError as e:
        log.info(
            "tts disabled: %s is not installed (uv sync --extra cpu --extra tts)",
            e.name,
        )
        return
    tts_state = "loading"
    threading.Thread(target=_load_tts, name="tts-load", daemon=True).start()


def _load_tts() -> None:
    global tts, voices_store, tts_state, tts_error
    try:
        engine = ReverbEngine(
            settings.tts_model, settings.tts_variant, settings.tts_device
        )
        store = VoiceStore(Path(settings.voices_dir_default()))
        # Publish the store before the engine: `tts` is the readiness gate, so a
        # request that passes the `tts is None` check always sees a live store.
        voices_store = store
        tts = engine
        tts_state = "ready"
        log.info(
            "reverb-tts loaded in %.1fs (%s, %d presets, %d saved voices)",
            engine.load_seconds,
            settings.tts_model,
            len(engine.presets()),
            len(store.ids()),
        )
    except Exception as e:  # noqa: BLE001 — TTS is optional; STT stays up
        tts_state = "error"
        tts_error = f"reverb-tts failed to load: {e}"
        log.exception(tts_error)


def tts_unavailable() -> JSONResponse | None:
    """503 with a fixed, state-specific reason while TTS is not ready."""
    if tts is not None:
        return None
    reason = {
        "loading": "tts is still loading (first start downloads about 1 GB of weights)",
        "disabled": "tts is disabled: SPEAKEASY_TTS_ENABLED=false, or the tts extra is not installed",
        "error": "tts encountered an error and is unavailable",
    }.get(tts_state, "tts unavailable")
    return JSONResponse(status_code=503, content={"error": {"message": reason}})


# Fixed, per-state 503 reasons. The strings are the contract; `voxcpm2_error`
# is logged only and never reaches a client body.
VOXCPM2_503_REASONS = {
    "disabled": (
        "voxcpm2 is disabled; set SPEAKEASY_TTS_VOXCPM_ENABLED=true to enable it"
    ),
    "not_loaded": (
        "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)"
    ),
    "loading": (
        "voxcpm2 is still loading (first start downloads about 4.7 GB of weights)"
    ),
    "error": "voxcpm2 encountered an error and is unavailable",
}


def voxcpm2_503(state: str) -> JSONResponse:
    """503 with the fixed reason for `state`."""
    return JSONResponse(
        status_code=503, content={"error": {"message": VOXCPM2_503_REASONS[state]}}
    )


def voxcpm2_unavailable() -> JSONResponse | None:
    """503 with a fixed, state-specific reason while VoxCPM2 is not ready."""
    if voxcpm2_state in VOXCPM2_503_REASONS:
        return voxcpm2_503(voxcpm2_state)
    return None


VOXCPM2_BAD_TAKE_REASON = (
    "voxcpm2 could not generate a good take for this input; "
    "check the input length and try again"
)


def voxcpm2_bad_take() -> JSONResponse:
    """503 with the fixed bad-take reason — only a good take is ever encoded."""
    return JSONResponse(
        status_code=503, content={"error": {"message": VOXCPM2_BAD_TAKE_REASON}}
    )


def start_voxcpm2() -> None:
    """Set up the VoxCPM2 lifecycle without loading the model.

    Readiness never waits on the engine: this only picks the starting state,
    creates the clone store, and starts the idle-unload timer. The model loads
    on the first request (`_ensure_voxcpm2_loading`).
    """
    global voxcpm2_state, voxcpm2_store
    if voxcpm2_store is not None:
        return  # already started; never spawn a second timer
    if not settings.tts_voxcpm_enabled:
        voxcpm2_state = "disabled"
        log.info("voxcpm2 disabled: SPEAKEASY_TTS_VOXCPM_ENABLED=false")
        return
    try:
        import torch  # noqa: F401
        import voxcpm  # noqa: F401 — absent without the `tts-voxcpm` extra
    except ImportError as e:
        voxcpm2_state = "disabled"
        log.info(
            "voxcpm2 disabled: %s is not installed "
            "(uv sync --extra cpu --extra tts-voxcpm)",
            e.name,
        )
        return
    voxcpm2_state = "not_loaded"
    # The store is a filesystem tree, created eagerly and never by the loader;
    # it survives an idle unload.
    voxcpm2_store = ReferenceVoiceStore(Path(settings.voices_dir_default()))
    if settings.tts_voxcpm_idle_seconds > 0:
        threading.Thread(
            target=_voxcpm2_unload_loop, name="voxcpm2-idle", daemon=True
        ).start()


def _ensure_voxcpm2_loading() -> JSONResponse | None:
    """Single-flight load trigger; 503 while the engine is not loaded, else None.

    A request in `not_loaded` starts exactly one loader thread; a request in
    `loading` starts nothing. Both answer the loading 503. Never takes
    `tts_lock` and never skips because it is busy.
    """
    global voxcpm2_state
    with voxcpm2_state_lock:
        if voxcpm2_state == "not_loaded":
            voxcpm2_state = "loading"
            threading.Thread(
                target=_load_voxcpm2, name="voxcpm2-load", daemon=True
            ).start()
            return voxcpm2_503("not_loaded")
        if voxcpm2_state == "loading":
            return voxcpm2_503("loading")
    return None  # ready / disabled / error — the caller applies its own gate


def _load_voxcpm2() -> None:
    """Load the engine under `tts_lock` and publish it. Nothing auto-retries."""
    global voxcpm2, voxcpm2_state, voxcpm2_error, voxcpm2_last_used
    try:
        with tts_lock:
            engine = VoxCPM2Engine(settings.tts_voxcpm_model, settings.tts_device)
            voxcpm2 = engine
            # The idle clock starts at the ready transition, so a tick cannot
            # unload a freshly loaded engine before its first request arrives.
            voxcpm2_last_used = time.monotonic()
            voxcpm2_state = "ready"
    except Exception as e:  # noqa: BLE001 — TTS is optional; STT stays up
        with tts_lock:
            voxcpm2_state = "error"
            voxcpm2_error = f"voxcpm2 failed to load: {e}"
        log.exception(voxcpm2_error)
        return
    log.info(
        "voxcpm2 loaded in %.1fs (%s)",
        engine.load_seconds,
        settings.tts_voxcpm_model,
    )


def _voxcpm2_unload_loop() -> None:
    """Idle-unload timer: one pass every `VOXCPM2_TIMER_INTERVAL` seconds."""
    while True:
        time.sleep(VOXCPM2_TIMER_INTERVAL)
        _voxcpm2_unload_tick()


def _voxcpm2_unload_tick() -> None:
    """One idle-unload pass.

    Acts only when the state is `ready`, the idle time has passed and the TTS
    lock is free at once — so it never unloads under a running request and
    never clobbers a load in flight or a freshly loaded engine.
    """
    global voxcpm2, voxcpm2_state
    if settings.tts_voxcpm_idle_seconds <= 0:
        return  # idle unload disabled
    if not tts_lock.acquire(blocking=False):
        return  # a request holds the lock; the next tick tries again
    try:
        if voxcpm2_state != "ready":
            return  # nothing to unload in loading / error / disabled / not_loaded
        if time.monotonic() - voxcpm2_last_used <= settings.tts_voxcpm_idle_seconds:
            return  # still warm
        engine = voxcpm2
        voxcpm2 = None
        voxcpm2_state = "not_loaded"
        if engine is not None:
            engine.unload()  # drops the model, empties the cache, collects
        log.info("voxcpm2 unloaded after idle; the next request reloads it")
    finally:
        tts_lock.release()


app = FastAPI(title="speakeasy", version="0.1.0", lifespan=lifespan)

# Static UI at `/` — three files served as-is, no build step, no third-party
# script. The CSP has no `unsafe-inline`, so the page carries no
# inline <script>/<style> and no inline event handlers.
STATIC = Path(__file__).parent / "static"
CSP = (
    "default-src 'self'; media-src 'self' blob:; connect-src 'self'; "
    "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'"
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(
        STATIC / "index.html",
        headers={
            "Content-Security-Policy": CSP,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    if startup_error:
        return JSONResponse(
            status_code=503, content={"status": "error", "detail": startup_error}
        )
    if whisper is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ready",
        "device": resolved_device,
        "models": _model_ids(),
        "tts": tts_state,
        "voxcpm2": voxcpm2_state,
    }


@app.get("/v1/models", dependencies=[Depends(require_auth)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "speakeasy"}
            for m in _model_ids()
        ],
    }


def _model_ids() -> list[str]:
    ids = []
    if whisper is not None:
        ids.append(settings.whisper_model)
    if moonshine is not None:
        ids.append("moonshine-base")
    if tts is not None:
        ids.append("reverb-tts")
    if voxcpm2_state != "disabled":
        ids.append("voxcpm2")
    return ids


def _reverb_voice_objects() -> list[dict]:
    """Reverb presets first then clones, each group sorted by id; needs a live engine."""
    data = [
        {
            "id": p["id"],
            "object": "voice",
            "type": "preset",
            "language": p["language"],
            "engine": "reverb-tts",
        }
        for p in tts.presets()
    ]
    if voices_store is not None:
        data += [
            {
                "id": voice_id,
                "object": "voice",
                "type": "clone",
                **info,
                "engine": "reverb-tts",
            }
            for voice_id, info in voices_store.listing()
        ]
    return data


def _voxcpm2_voice_objects() -> list[dict]:
    """VoxCPM2 clones sorted by id, read from disk — no loaded engine needed."""
    if voxcpm2_store is None:
        return []
    return [
        {
            "id": voice_id,
            "object": "voice",
            "type": "clone",
            "engine": "voxcpm2",
            "language": info.get("language"),
            "transcript": info.get("transcript"),
            "created": info.get("created"),
        }
        for voice_id, info in voxcpm2_store.listing()
    ]


@app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
def list_voices():
    # 503 only when VoxCPM2 is disabled and Reverb is not ready. Either engine
    # alone answers 200; presets appear only while Reverb is ready. The Reverb
    # 503 bodies are byte-identical to today's.
    if voxcpm2_state == "disabled":
        if resp := tts_unavailable():
            return resp
    data: list[dict] = []
    if tts is not None:
        data += _reverb_voice_objects()
    if voxcpm2_state != "disabled":
        data += _voxcpm2_voice_objects()
    body: dict = {
        "object": "list",
        "engines": sorted({obj["engine"] for obj in data}),
        "data": data,
    }
    if tts is not None:
        body["languages"] = tts.languages
    return body


SPEECH_FORMATS = {  # response_format -> (libsndfile format, subtype, content type)
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "pcm": (None, None, "audio/pcm"),  # raw s16le mono at the engine's rate
}


class SpeechRequest(BaseModel):
    input: str = Field(max_length=4096)
    # Optional only so a `voxcpm2` request can omit or null it (resolves to
    # `default`); a Reverb request with a null voice is answered by the route
    # with the 422 the old `str` annotation produced.
    voice: str | None = "en_male_narrator_1"
    response_format: Literal["wav", "mp3", "flac", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)  # a length factor; see README
    model: str = ""  # "voxcpm2" selects the second engine; every other value is Reverb
    language: str | None = None  # default: the voice's language
    seed: int | None = None
    # VoxCPM2-only controls; a set value on a Reverb request is a 400. `cfg`
    # left null means the engine's own default (2.0).
    cfg: float | None = Field(default=None, ge=1.0, le=5.0)
    design: str | None = Field(default=None, min_length=1, max_length=300)
    style: str | None = Field(default=None, min_length=1, max_length=200)


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {"message": message}})


def _resolve_voice(voice_id: str):
    """A saved clone by id, else a preset. ValueError if neither."""
    if voices_store is not None and voices_store.exists(voice_id):
        try:
            return voices_store.load(voice_id)
        except FileNotFoundError:
            # The clone was deleted between the existence check and the read;
            # fall through and report it exactly like an id never saved.
            pass
    try:
        return tts.voice(voice_id)
    except ValueError:
        known = ", ".join(
            [p["id"] for p in tts.presets()]
            + (voices_store.ids() if voices_store else [])
        )
        raise ValueError(f"unknown voice {voice_id!r}. Known: {known}") from None


def _encode(audio: np.ndarray, fmt: str, sample_rate: int) -> tuple[bytes, str]:
    """Encode `audio` at the sample rate of the engine that produced it."""
    snd, subtype, media = SPEECH_FORMATS[fmt]
    if snd is None:
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes(), media
    buf = io.BytesIO()
    with sf.SoundFile(
        buf, "w", samplerate=sample_rate, channels=1, format=snd, subtype=subtype
    ) as f:
        f.write(audio)
    return buf.getvalue(), media


@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
def speech(req: SpeechRequest):
    # A plain `def`, not `async def`: FastAPI runs it on the threadpool, so the
    # engine lock blocks a worker thread and never the event loop.
    if req.model == "voxcpm2":
        return _speech_voxcpm2(req)
    return _speech_reverb(req)


def _speech_reverb(req: SpeechRequest):
    """Reverb's speech path — unchanged for a request with no new fields."""
    # A null voice is the 422 the field's old `str` annotation produced; the
    # optional type a `voxcpm2` request needs must not move Reverb's contract.
    if req.voice is None:
        return _null_voice_422()
    if resp := tts_unavailable():
        return resp
    # `cfg`, `design` and `style` are VoxCPM2-only: a set value asks for the
    # other engine.
    if req.cfg is not None or req.design is not None or req.style is not None:
        return _bad_request(
            "cfg, design and style are voxcpm2-only fields; set model=voxcpm2"
        )
    text = req.input.strip()
    if not text:
        return _bad_request("input must not be empty")
    try:
        voice = _resolve_voice(req.voice)
    except ValueError as e:
        return _bad_request(str(e))
    language = req.language or voice.language
    if language not in tts.languages:
        return _bad_request(
            f"unsupported language {language!r}. Supported: {', '.join(tts.languages)}"
        )
    # The shared lock serialises both engines' inference; Reverb keeps its own
    # engine lock too.
    with tts_lock:
        audio, elapsed = tts.synthesize(
            text, voice, language, seed=req.seed, speed=req.speed
        )
    return _speech_response(
        audio, elapsed, req.response_format, tts.name, tts.sample_rate
    )


def _voxcpm2_unknown_voice(voice_id: str) -> str:
    """The fixed 400 message for a voxcpm2 voice that is not `default` or a clone."""
    saved = ", ".join(voxcpm2_store.ids()) if voxcpm2_store is not None else ""
    return (
        f"unknown voice {voice_id!r} for voxcpm2; "
        f"use 'default' or a saved clone: {saved}"
    )


def _speech_voxcpm2(req: SpeechRequest):
    """VoxCPM2's speech path: lifecycle gate, validation, one locked inference."""
    global voxcpm2_last_used

    # 1. Single-flight lifecycle gate: a request in `not_loaded` starts exactly
    #    one load and answers 503; `loading` answers 503 and starts nothing.
    if resp := _ensure_voxcpm2_loading():
        return resp
    # 2. Gate check without the lock; a pass is not a promise.
    if resp := voxcpm2_unavailable():
        return resp
    # 3. Validation (needs no lock).
    text = req.input.strip()
    if not text:
        return _bad_request("input must not be empty")
    if text.startswith("("):
        return _bad_request(
            "input starting with '(' is reserved for design and style; "
            "remove the leading parenthesis or use the design field"
        )
    if req.speed != 1.0:
        return _bad_request("speed must be 1.0 for the voxcpm2 engine")
    if req.design is not None and req.style is not None:
        return _bad_request("at most one of design and style may be set")
    # An omitted `voice` is the model's own voice, and an explicit null resolves
    # to `default` too; any other id must be a saved clone.
    voice_id = (
        "default"
        if "voice" not in req.model_fields_set or req.voice is None
        else req.voice
    )
    if req.design is not None and voice_id != "default":
        return _bad_request("design requires voice: 'default'")
    for value in (req.design, req.style):
        if value is not None and (
            "(" in value or ")" in value or "\n" in value or "\r" in value
        ):
            return _bad_request(
                "design and style must not contain parentheses or newlines"
            )
    reference = None
    prompt_text = None
    if voice_id != "default":
        store = voxcpm2_store
        if store is None or not store.exists(voice_id):
            return _bad_request(_voxcpm2_unknown_voice(voice_id))
        try:
            info = store.info(voice_id)
        except ValueError:
            return _bad_request(_voxcpm2_unknown_voice(voice_id))
        reference = str(store.reference_path(voice_id))
        prompt_text = info.get("transcript")
    # The library reads a leading parenthesized prefix as design/style syntax;
    # the server builds it itself and hands the engine one text.
    prefix = req.design if req.design is not None else req.style
    library_text = f"({prefix}){text}" if prefix is not None else text
    cfg = req.cfg if req.cfg is not None else 2.0

    # 4-9. One inference at a time: acquire the shared lock, re-check the engine
    # under it, seed and generate, then bump last-used — all under the lock.
    with tts_lock:
        if voxcpm2_state == "error":
            return voxcpm2_503("error")
        if voxcpm2_state == "disabled":
            return voxcpm2_503("disabled")
        if voxcpm2_state != "ready" or voxcpm2 is None:
            # The engine was unloaded between the gate and the lock: restart the
            # load and answer 503, never 500.
            if resp := _ensure_voxcpm2_loading():
                return resp
            return voxcpm2_503("not_loaded")
        engine = voxcpm2
        try:
            audio, elapsed = engine.synthesize(
                library_text,
                reference=reference,
                prompt_text=prompt_text,
                cfg=cfg,
                seed=req.seed,
            )
        except BadTake:
            return voxcpm2_bad_take()
        except FileNotFoundError:
            # The clone vanished between resolution and the library call.
            return _bad_request(_voxcpm2_unknown_voice(voice_id))
        except Exception:  # noqa: BLE001 — never leak engine internals
            log.exception("voxcpm2 synthesis failed")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "voxcpm2 failed to generate speech"}},
            )
        voxcpm2_last_used = time.monotonic()
        sample_rate = engine.sample_rate
    return _speech_response(
        audio, elapsed, req.response_format, engine.name, sample_rate
    )


def _speech_response(
    audio: np.ndarray, elapsed: float, fmt: str, engine_name: str, sample_rate: int
) -> Response:
    """Encode `audio` and build the four response headers, naming the engine."""
    body, media_type = _encode(audio, fmt, sample_rate)
    duration = len(audio) / sample_rate
    log.info(
        "speech engine=%s audio=%.2fs elapsed=%.3fs", engine_name, duration, elapsed
    )
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "X-Speakeasy-Engine": engine_name,
            "X-Speakeasy-Elapsed": str(round(elapsed, 3)),
            "X-Speakeasy-Audio-Duration": str(round(duration, 3)),
            "X-Speakeasy-RTF": str(round(elapsed / max(duration, 0.01), 3)),
        },
    )


# ---------------------------------------------------------------------------
# Voice clones — create a voice from a reference clip, and delete one.
# ---------------------------------------------------------------------------

MAX_REFERENCE_BYTES = 20 * 2**20
MIN_REFERENCE_SECONDS, MAX_REFERENCE_SECONDS = 1.0, 30.0  # the library warns past 15 s


def _decode_reference(data: bytes, sampling_rate: int | None = None) -> np.ndarray:
    """Mono float32 at `sampling_rate`, through PyAV (any browser codec).

    With no `sampling_rate` it decodes at the Reverb engine's own rate; VoxCPM2
    passes its 16 kHz reference rate, so its clone path never reads Reverb's.
    """
    from faster_whisper.audio import decode_audio

    rate = tts.sample_rate if sampling_rate is None else sampling_rate
    tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return decode_audio(tmp.name, sampling_rate=rate)
    finally:
        os.unlink(tmp.name)


def _missing_text_422() -> JSONResponse:
    """The 422 FastAPI raised for a missing `text` before it became optional."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "type": "missing",
                    "loc": ["body", "text"],
                    "msg": "Field required",
                    "input": None,
                }
            ]
        },
    )


def _null_voice_422() -> JSONResponse:
    """The 422 FastAPI raised for a null `voice` before the field was optional.

    `SpeechRequest.voice` is optional so a `voxcpm2` request can omit or null
    it; a Reverb request with a null voice must still answer the body the old
    `str` annotation produced, so it is hand-built here (as `_missing_text_422`
    does for `text`).
    """
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {
                    "type": "string_type",
                    "loc": ["body", "voice"],
                    "msg": "Input should be a valid string",
                    "input": None,
                }
            ]
        },
    )


def _voice_id_taken(voice_id: str) -> bool:
    """True when either engine's store already owns `voice_id` (ids are shared).

    The Reverb store object is `None` while Reverb is disabled, loading or in
    error, but its clones are still on disk, so fall back to the file the store
    would use. The path is built only after the id passes validation, and no
    store is constructed — its constructor would create the voices directory.
    """
    if voices_store is not None:
        if voices_store.exists(voice_id):
            return True
    else:
        try:
            check_voice_id(voice_id)
        except ValueError:
            return False
        if (Path(settings.voices_dir_default()) / f"{voice_id}.npz").is_file():
            return True
    return voxcpm2_store is not None and voxcpm2_store.exists(voice_id)


@app.post("/v1/audio/voices", dependencies=[Depends(require_auth)], status_code=201)
def create_voice(
    file: UploadFile = File(...),
    text: str | None = Form(default=None),
    language: str = Form("en"),
    voice_id: str | None = Form(default=None),
    model: str = Form(default=""),
):
    # A plain `def`: the engine lock blocks a threadpool worker, not the loop.
    # `text` is optional only for VoxCPM2, so required-ness is checked below.
    if model == "voxcpm2":
        return _create_voxcpm2_voice(file, text, language, voice_id)
    return _create_reverb_voice(file, text, language, voice_id)


def _create_reverb_voice(
    file: UploadFile, text: str | None, language: str, voice_id: str | None
):
    """Reverb's clone-create path — unchanged, plus the shared-id check."""
    if text is None:
        return _missing_text_422()
    if resp := tts_unavailable():
        return resp
    transcript = text.strip()
    if not transcript:
        return _bad_request("text (the reference's exact transcript) must not be empty")
    if language not in tts.languages:
        return _bad_request(
            f"unsupported language {language!r}. Supported: {', '.join(tts.languages)}"
        )
    voice_id = voice_id or f"clone-{secrets.token_hex(4)}"
    # Preset check before id validation: real preset ids use '_', which the
    # store regex rejects, yet a preset collision must still be 409, not 400.
    if any(p["id"] == voice_id for p in tts.presets()):
        return JSONResponse(
            status_code=409,
            content={"error": {"message": f"{voice_id!r} is a preset voice"}},
        )
    try:
        check_voice_id(voice_id)
    except ValueError as e:
        return _bad_request(str(e))
    if _voice_id_taken(voice_id):  # fast path; re-checked under the lock
        return JSONResponse(
            status_code=409,
            content={"error": {"message": f"voice {voice_id!r} already exists"}},
        )
    data = file.file.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": {"message": "reference audio over 20 MB"}},
        )
    # Decode and validate duration OUTSIDE the engine lock: a slow or
    # oversized upload must never stall speech inference.
    try:
        samples = _decode_reference(data)
    except Exception:  # noqa: BLE001 — any decoder failure is a bad upload
        return _bad_request("could not decode the reference audio")
    duration = len(samples) / tts.sample_rate
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        return _bad_request(
            f"reference is {duration:.1f}s; give it 3-10 s of speech "
            f"({MIN_REFERENCE_SECONDS:g}-{MAX_REFERENCE_SECONDS:g} accepted)"
        )
    # Serialise re-check → clone → save under the cross-engine create lock
    # (`voxcpm2_create_lock`) and the engine's RLock: sharing the create lock
    # with the VoxCPM2 create path is what makes a same-id create on either
    # engine exactly one 201 and one 409. `clone` re-acquires the engine lock
    # (reentrant), so the nested acquire is safe.
    with voxcpm2_create_lock, tts._lock:
        if _voice_id_taken(voice_id):
            return JSONResponse(
                status_code=409,
                content={"error": {"message": f"voice {voice_id!r} already exists"}},
            )
        voice = tts.clone(samples, tts.sample_rate, transcript, language)
        created = voices_store.save(voice_id, voice)
    log.info(
        "voice %s created (%.1fs reference, language=%s)", voice_id, duration, language
    )
    # Built from the values create already holds (and the save's `created`): no
    # read of the file after the lock, so a concurrent delete cannot break it.
    return {
        "id": voice_id,
        "object": "voice",
        "type": "clone",
        "language": language,
        "transcript": transcript,
        "created": created,
    }


VOXCPM2_REFERENCE_RATE = 16000


def _create_voxcpm2_voice(
    file: UploadFile, text: str | None, language: str, voice_id: str | None
):
    """Store a VoxCPM2 reference clone — the model never runs at create time."""
    # The engine gates its own request: `not_loaded` starts the single load and
    # answers 503; `loading`, `error` and `disabled` answer their own 503.
    if resp := _ensure_voxcpm2_loading():
        return resp
    if resp := voxcpm2_unavailable():
        return resp
    store = voxcpm2_store
    if store is None:
        return voxcpm2_503("disabled")
    transcript = text.strip() if text is not None else None
    if transcript == "":
        transcript = None  # a blank form value means "no transcript"
    voice_id = voice_id or f"clone-{secrets.token_hex(4)}"
    if tts is not None and any(p["id"] == voice_id for p in tts.presets()):
        return JSONResponse(
            status_code=409,
            content={"error": {"message": f"{voice_id!r} is a preset voice"}},
        )
    try:
        check_voice_id(voice_id)
    except ValueError as e:
        return _bad_request(str(e))
    data = file.file.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": {"message": "reference audio over 20 MB"}},
        )
    # Decode at this engine's reference rate, never Reverb's; not under any lock.
    try:
        samples = _decode_reference(data, VOXCPM2_REFERENCE_RATE)
    except Exception:  # noqa: BLE001 — any decoder failure is a bad upload
        return _bad_request("could not decode the reference audio")
    duration = len(samples) / VOXCPM2_REFERENCE_RATE
    if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
        return _bad_request(
            f"reference is {duration:.1f}s; give it 3-10 s of speech "
            f"({MIN_REFERENCE_SECONDS:g}-{MAX_REFERENCE_SECONDS:g} accepted)"
        )
    # One create lock spans the both-store existence check through the save, so
    # two creates of one id give one 201 and one 409. Never `tts_lock`.
    with voxcpm2_create_lock:
        if _voice_id_taken(voice_id):
            return JSONResponse(
                status_code=409,
                content={"error": {"message": f"voice {voice_id!r} already exists"}},
            )
        created = store.save(
            voice_id, samples, VOXCPM2_REFERENCE_RATE, transcript, language=language
        )
    log.info(
        "voxcpm2 voice %s created (%.1fs reference, language=%s)",
        voice_id,
        duration,
        language,
    )
    # Built from the values create already holds (and the save's `created`): the
    # saved files are never read back, so a concurrent delete cannot break it.
    return {
        "id": voice_id,
        "object": "voice",
        "type": "clone",
        "engine": "voxcpm2",
        "language": language,
        "transcript": transcript,
        "created": created,
    }


@app.delete("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
def delete_voice(voice_id: str):
    # Resolve against the VoxCPM2 store first — a filesystem op needing no loaded
    # engine. Any other id falls through to the Reverb path exactly as on main,
    # the `tts_unavailable()` gate included, so Reverb's status codes are unchanged.
    if voxcpm2_store is not None and voxcpm2_store.exists(voice_id):
        voxcpm2_store.delete(voice_id)
        return {
            "id": voice_id,
            "object": "voice",
            "deleted": True,
            "engine": "voxcpm2",
        }
    if resp := tts_unavailable():
        return resp
    if not voices_store.delete(voice_id):  # unknown, a preset, or an invalid id
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"no saved voice {voice_id!r}"}},
        )
    return {
        "id": voice_id,
        "object": "voice",
        "deleted": True,
        "engine": "reverb-tts",
    }


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    fast: bool = Query(
        default=False, description="Force Moonshine fast path (short English audio)"
    ),
    vad: bool = Query(default=True, description="Silero VAD filter (whisper only)"),
    word_timestamps: bool = Query(
        default=False, description="Word-level timestamps (whisper only)"
    ),
):
    if whisper is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": startup_error or "models not loaded"}},
        )

    if response_format not in FORMATS:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unsupported response_format '{response_format}'. Supported: {', '.join(FORMATS)}"
                }
            },
        )

    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    # Route: explicit fast=true forces Moonshine regardless of duration;
    # model=moonshine-base requests it by name; otherwise auto-route short
    # audio when the client didn't pin a whisper model.
    duration = probe_duration(audio_bytes)
    moonshine_eligible = moonshine is not None and (language in (None, "en"))
    use_moonshine = moonshine_eligible and (
        fast
        or model == "moonshine-base"
        or (
            not model
            and duration is not None
            and duration <= settings.fast_path_max_seconds
        )
    )

    # Write once, reuse for whichever engine runs
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()

        if use_moonshine:
            result = moonshine.transcribe(tmp.name)
        else:
            result = whisper.transcribe(
                tmp.name,
                language=language,
                vad_filter=vad,
                word_timestamps=word_timestamps,
            )
    finally:
        os.unlink(tmp.name)

    result.duration = result.duration or duration or 0.0

    formatter, media_type = FORMATS[response_format]
    body = formatter(result)

    headers = {
        "X-Speakeasy-Engine": result.engine,
        "X-Speakeasy-Elapsed": str(result.elapsed_seconds),
        "X-Speakeasy-Duration": str(result.duration),
        "X-Speakeasy-RTF": str(
            round(result.elapsed_seconds / max(result.duration, 0.01), 3)
        ),
    }

    log.info(
        "%s engine=%s duration=%.2fs elapsed=%.3fs rtf=%.3f",
        "transcribe",
        result.engine,
        result.duration,
        result.elapsed_seconds,
        result.elapsed_seconds / max(result.duration, 0.01),
    )

    if isinstance(body, dict):
        return JSONResponse(content=body, headers=headers)
    return Response(content=body, media_type=media_type, headers=headers)


def main():
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    uvicorn.run("speakeasy.server:app", host=settings.bind, port=settings.port)


if __name__ == "__main__":
    main()
