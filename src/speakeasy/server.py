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
from .engines import MoonshineEngine, ReverbEngine, WhisperEngine, probe_duration
from .responses import FORMATS
from .voices import VoiceStore, check_voice_id

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
    return ids


def _voice_objects() -> list[dict]:
    """Presets first (sorted by id), then saved clones (sorted by id)."""
    data = [
        {"id": p["id"], "object": "voice", "type": "preset", "language": p["language"]}
        for p in tts.presets()
    ]
    if voices_store is not None:
        data += [
            {"id": voice_id, "object": "voice", "type": "clone", **info}
            for voice_id, info in voices_store.listing()
        ]
    return data


@app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
def list_voices():
    if resp := tts_unavailable():
        return resp
    return {"object": "list", "languages": tts.languages, "data": _voice_objects()}


SPEECH_FORMATS = {  # response_format -> (libsndfile format, subtype, content type)
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "mp3": ("MP3", "MPEG_LAYER_III", "audio/mpeg"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "pcm": (None, None, "audio/pcm"),  # raw 24 kHz s16le mono, as OpenAI defines it
}


class SpeechRequest(BaseModel):
    input: str = Field(max_length=4096)
    voice: str = "en_male_narrator_1"
    response_format: Literal["wav", "mp3", "flac", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)  # a length factor; see README
    model: str = ""  # accepted for OpenAI clients; the server has one TTS model
    language: str | None = None  # default: the voice's language
    seed: int | None = None


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


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    snd, subtype, media = SPEECH_FORMATS[fmt]
    if snd is None:
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes(), media
    buf = io.BytesIO()
    with sf.SoundFile(
        buf, "w", samplerate=tts.sample_rate, channels=1, format=snd, subtype=subtype
    ) as f:
        f.write(audio)
    return buf.getvalue(), media


@app.post("/v1/audio/speech", dependencies=[Depends(require_auth)])
def speech(req: SpeechRequest):
    # A plain `def`, not `async def`: FastAPI runs it on the threadpool, so the
    # engine lock blocks a worker thread and never the event loop.
    if resp := tts_unavailable():
        return resp
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
    audio, elapsed = tts.synthesize(
        text, voice, language, seed=req.seed, speed=req.speed
    )
    body, media_type = _encode(audio, req.response_format)
    duration = len(audio) / tts.sample_rate
    log.info(
        "speech voice=%s language=%s audio=%.2fs elapsed=%.3fs",
        req.voice,
        language,
        duration,
        elapsed,
    )
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "X-Speakeasy-Engine": tts.name,
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


def _decode_reference(data: bytes) -> np.ndarray:
    """Mono float32 at the codec's sample rate, through PyAV (any browser codec)."""
    from faster_whisper.audio import decode_audio

    tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return decode_audio(tmp.name, sampling_rate=tts.sample_rate)
    finally:
        os.unlink(tmp.name)


@app.post("/v1/audio/voices", dependencies=[Depends(require_auth)], status_code=201)
def create_voice(
    file: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("en"),
    voice_id: str | None = Form(default=None),
):
    # A plain `def`: the engine lock blocks a threadpool worker, not the loop.
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
    if voices_store.exists(voice_id):  # fast path; re-checked under the lock
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
    # Serialise only exists-recheck → clone → save under the engine's RLock.
    # `clone` re-acquires the same lock (reentrant), so the nested acquire is safe.
    with tts._lock:
        if voices_store.exists(voice_id):
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


@app.delete("/v1/audio/voices/{voice_id}", dependencies=[Depends(require_auth)])
def delete_voice(voice_id: str):
    if resp := tts_unavailable():
        return resp
    if not voices_store.delete(voice_id):  # unknown, a preset, or an invalid id
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"no saved voice {voice_id!r}"}},
        )
    return {"id": voice_id, "object": "voice", "deleted": True}


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
