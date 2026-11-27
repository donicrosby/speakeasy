"""Speakeasy — OpenAI-compatible audio transcription server.

Endpoints:
    POST /v1/audio/transcriptions  — OpenAI-compatible transcription
    GET  /v1/audio/voices          — list preset and cloned voices (TTS)
    GET  /v1/models                — list loaded models
    GET  /health                   — liveness
    GET  /health/ready             — readiness (models loaded)
"""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

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
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, resolve_device_compute
from .engines import MoonshineEngine, ReverbEngine, WhisperEngine, probe_duration
from .responses import FORMATS
from .voices import VoiceStore

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
# never returned (SEC-16).
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
            {"id": i, "object": "voice", "type": "clone", **voices_store.info(i)}
            for i in voices_store.ids()
        ]
    return data


@app.get("/v1/audio/voices", dependencies=[Depends(require_auth)])
def list_voices():
    if resp := tts_unavailable():
        return resp
    return {"object": "list", "languages": tts.languages, "data": _voice_objects()}


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
