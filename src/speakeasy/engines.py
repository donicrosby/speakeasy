"""Transcription engines.

Two engines behind one interface:

- WhisperEngine: faster-whisper (CTranslate2) — primary, multilingual, punctuated.
  `device="cuda"` covers both NVIDIA CUDA and AMD ROCm (the CTranslate2 ROCm
  wheel is HIP-backed but exposes the same device name).
- MoonshineEngine: transformers — fast path for short English voice notes.
  13x smaller and ~10x faster than Whisper large-v3-turbo, but English-only
  and drops punctuation.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf


@dataclass
class Segment:
    start: float
    end: float
    text: str
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    engine: str = "unknown"
    elapsed_seconds: float = 0.0


class WhisperEngine:
    def __init__(self, model_name: str, device: str, compute_type: str, beam_size: int = 5):
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.load_seconds = time.monotonic() - t0
        self.model_name = model_name
        self.beam_size = beam_size

    @property
    def name(self) -> str:
        return "whisper"

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        vad_filter: bool = True,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        t0 = time.monotonic()
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
        segments = [
            Segment(
                start=round(s.start, 2),
                end=round(s.end, 2),
                text=s.text,
                temperature=s.temperature,
                avg_logprob=round(s.avg_logprob, 4),
                compression_ratio=round(s.compression_ratio, 4),
                no_speech_prob=round(s.no_speech_prob, 4),
            )
            for s in segments_gen
        ]
        return TranscriptionResult(
            text="".join(s.text for s in segments).strip(),
            language=info.language,
            duration=round(info.duration, 2),
            segments=segments,
            engine=self.name,
            elapsed_seconds=round(time.monotonic() - t0, 3),
        )


class MoonshineEngine:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        t0 = time.monotonic()
        self._torch = torch
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.load_seconds = time.monotonic() - t0
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "moonshine"

    def transcribe(self, audio_path: str, **_: object) -> TranscriptionResult:
        torch = self._torch

        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo to mono
        if sr != 16000:
            from scipy import signal as scipy_signal

            audio = scipy_signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)

        duration = len(audio) / 16000

        t0 = time.monotonic()
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        return TranscriptionResult(
            text=text,
            language="en",
            duration=round(duration, 2),
            segments=[Segment(start=0.0, end=round(duration, 2), text=text)],
            engine=self.name,
            elapsed_seconds=round(time.monotonic() - t0, 3),
        )


def probe_duration(audio_bytes: bytes) -> float | None:
    """Duration of an audio file in seconds, or None if it can't be read."""
    tmp = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()
        info = sf.info(tmp.name)
        return round(info.duration, 2)
    except Exception:
        return None
    finally:
        os.unlink(tmp.name)
