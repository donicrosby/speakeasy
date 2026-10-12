"""Speakeasy — OpenAI-compatible audio transcription server.

Endpoints:
    POST /v1/audio/transcriptions  — OpenAI-compatible transcription
    GET  /v1/models                — list loaded models
    GET  /health                   — liveness
    GET  /health/ready             — readiness (models loaded)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .config import Settings, resolve_device_compute
from .engines import MoonshineEngine, TranscriptionResult, WhisperEngine, probe_duration
from .responses import FORMATS

log = logging.getLogger("speakeasy")

settings = Settings()
if settings.hf_home:
    os.environ["HF_HOME"] = settings.hf_home

whisper: WhisperEngine | None = None
moonshine: MoonshineEngine | None = None
resolved_device = "unknown"
startup_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper, moonshine, resolved_device, startup_error

    device, compute = resolve_device_compute(settings)
    resolved_device = device
    log.info("device=%s compute=%s", device, compute)

    try:
        whisper = WhisperEngine(settings.whisper_model, device, compute, settings.beam_size)
        log.info("whisper loaded in %.1fs (%s)", whisper.load_seconds, settings.whisper_model)
    except Exception as e:
        startup_error = f"whisper failed to load: {e}"
        log.exception(startup_error)

    if settings.moonshine_enabled:
        try:
            moonshine = MoonshineEngine(settings.moonshine_model)
            log.info("moonshine loaded in %.1fs (%s)", moonshine.load_seconds, settings.moonshine_model)
        except Exception as e:
            # Fast path is optional — log and continue
            log.warning("moonshine failed to load, fast path disabled: %s", e)
            moonshine = None

    yield


app = FastAPI(title="speakeasy", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    if startup_error:
        return JSONResponse(status_code=503, content={"status": "error", "detail": startup_error})
    if whisper is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ready",
        "device": resolved_device,
        "models": _model_ids(),
    }


@app.get("/v1/models")
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
    return ids


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    fast: bool = Query(default=False, description="Force Moonshine fast path (short English audio)"),
    vad: bool = Query(default=True, description="Silero VAD filter (whisper only)"),
    word_timestamps: bool = Query(default=False, description="Word-level timestamps (whisper only)"),
):
    if whisper is None:
        return JSONResponse(status_code=503, content={"error": {"message": startup_error or "models not loaded"}})

    if response_format not in FORMATS:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unsupported response_format '{response_format}'. Supported: {', '.join(FORMATS)}"}},
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
    }

    if isinstance(body, dict):
        return JSONResponse(content=body, headers=headers)
    return Response(content=body, media_type=media_type, headers=headers)


def main():
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run("speakeasy.server:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
