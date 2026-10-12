"""Speakeasy spike 001: OpenAI-compatible transcription endpoint.

Validates: given faster-whisper, when we expose /v1/audio/transcriptions,
then it matches the OpenAI API contract closely enough for existing clients.
"""

import io
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup
# ---------------------------------------------------------------------------

model: Optional[WhisperModel] = None
MODEL_NAME = "large-v3-turbo"
DEVICE = "cpu"  # spike: CPU first, ROCm in the container
COMPUTE_TYPE = "int8"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    t0 = time.monotonic()
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"Model loaded in {time.monotonic() - t0:.1f}s ({MODEL_NAME}, {DEVICE}/{COMPUTE_TYPE})")
    yield
    model = None


app = FastAPI(title="speakeasy", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# OpenAI-compatible response shapes
# ---------------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def _to_verbose_json(segments: list, info) -> dict:
    """OpenAI verbose_json response format."""
    return {
        "task": "transcribe",
        "language": info.language,
        "duration": info.duration,
        "text": "".join(s.text for s in segments).strip(),
        "segments": [
            {
                "id": i,
                "seek": 0,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": s.text,
                "tokens": [],
                "temperature": s.temperature,
                "avg_logprob": round(s.avg_logprob, 4),
                "compression_ratio": round(s.compression_ratio, 4),
                "no_speech_prob": round(s.no_speech_prob, 4),
            }
            for i, s in enumerate(segments)
        ],
    }


def _to_srt(segments: list) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp(s.start)} --> {_format_timestamp(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines)


def _to_vtt(segments: list) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        start = _format_timestamp(s.start).replace(",", ".")
        end = _format_timestamp(s.end).replace(",", ".")
        lines.append(f"{start} --> {end}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "speakeasy",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("large-v3-turbo", alias="model"),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    audio_bytes = await file.read()

    # faster-whisper's decode_audio needs a file path for some formats;
    # write to temp file for reliability
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments_gen, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        segments = list(segments_gen)
    finally:
        os.unlink(tmp_path)

    if response_format == "json":
        return {"text": "".join(s.text for s in segments).strip()}
    elif response_format == "verbose_json":
        return _to_verbose_json(segments, info)
    elif response_format == "srt":
        return JSONResponse(
            content=_to_srt(segments),
            media_type="text/plain; charset=utf-8",
        )
    elif response_format == "vtt":
        return JSONResponse(
            content=_to_vtt(segments),
            media_type="text/vtt; charset=utf-8",
        )
    elif response_format == "text":
        return JSONResponse(
            content="".join(s.text for s in segments).strip(),
            media_type="text/plain; charset=utf-8",
        )
    else:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unsupported response_format: {response_format}"}},
        )


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
