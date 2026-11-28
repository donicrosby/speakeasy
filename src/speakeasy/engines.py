"""Transcription engines.

Two engines behind one interface:

- WhisperEngine: faster-whisper (CTranslate2) — primary, multilingual, punctuated.
  `device="cuda"` covers both NVIDIA CUDA and AMD ROCm (the CTranslate2 ROCm
  wheel is HIP-backed but exposes the same device name).
- MoonshineEngine: transformers — fast path for short English voice notes.
  13x smaller and ~10x faster than Whisper large-v3-turbo, but English-only
  and drops punctuation.

ReverbEngine (speech synthesis) lives here too, behind the optional `tts` extra.
"""

from __future__ import annotations

import gc
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
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
    def __init__(
        self, model_name: str, device: str, compute_type: str, beam_size: int = 5
    ):
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

            audio = scipy_signal.resample(audio, int(len(audio) * 16000 / sr)).astype(
                np.float32
            )

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


class ReverbEngine:
    """Reverb-TTS — zero-shot voice cloning, behind the `tts` extra.

    One model on one device: `synthesize` and `clone` share one lock, so
    concurrent requests queue instead of racing the sampler. The lock is
    reentrant because the clone-create path holds it and then calls `clone`,
    which re-acquires the same lock.
    """

    def __init__(self, model: str, variant: str, device: str):
        import torch
        from reverb_tts import LANGUAGES, TTS

        if torch.version.hip is not None:
            # Spike 004: MIOpen's default find mode costs 1.5-19 s per NEW output
            # length in the codec decoder. Disable cuDNN/MIOpen on HIP only.
            torch.backends.cudnn.enabled = False
        t0 = time.monotonic()
        self._tts = TTS.from_pretrained(
            model, variant=variant, device=None if device == "auto" else device
        )
        self.load_seconds = time.monotonic() - t0
        self.sample_rate = self._tts.sample_rate
        self.languages = sorted(LANGUAGES)
        # Fetched once here, so no request ever needs the network for a preset.
        self._presets = {
            n: self._tts.preset_voice(n) for n in self._tts.preset_voices()
        }
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "reverb-tts"

    def presets(self) -> list[dict]:
        """[{id, language, transcript}] — a preset's transcript is its reference text."""
        return [
            {"id": n, "language": v.language, "transcript": v.text}
            for n, v in sorted(self._presets.items())
        ]

    def voice(self, preset_id: str):
        try:
            return self._presets[preset_id]
        except KeyError:
            raise ValueError(f"unknown voice {preset_id!r}") from None

    def clone(self, samples: np.ndarray, sample_rate: int, text: str, language: str):
        """A voice from a reference recording and its exact transcript."""
        with self._lock:
            return self._tts.voice(samples, text, language, sample_rate=sample_rate)

    def synthesize(
        self,
        text: str,
        voice,
        language: str,
        *,
        seed: int | None = None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """float32 samples at `sample_rate`, and the seconds it took."""
        with self._lock:
            t0 = time.monotonic()
            audio = self._tts.generate(text, voice, language, seed=seed, speed=speed)
        return audio, time.monotonic() - t0

    def synthesize_streaming(
        self,
        text: str,
        voice,
        language: str,
        *,
        seed: int | None = None,
        speed: float = 1.0,
    ) -> Iterator[np.ndarray]:
        """The finished take as one chunk: `reverb_tts` has no streaming entry.

        Takes what `synthesize` takes and generates the whole take under the
        engine lock, exactly as `synthesize` does; the single chunk is handed
        over once it is ready. That is the documented behaviour of this engine,
        not an error.
        """
        audio, _ = self.synthesize(text, voice, language, seed=seed, speed=speed)
        yield audio


class BadTake(Exception):
    """A take the library itself would call bad: too much audio per text token."""


def _flatten_text(text: str) -> str:
    """The library flattens newlines and whitespace runs before tokenising."""
    return re.sub(r"\s+", " ", text.replace("\n", " "))


class VoxCPM2Engine:
    """VoxCPM2 — 48 kHz speech, voice design, style hints, reference cloning.

    One model on one device. `synthesize` holds one lock, because the library
    seeds through torch's global generator and a second call would otherwise
    race the sampler. The server loads it lazily and drops it with `unload`,
    which the caller must serialise against `synthesize`.
    """

    # One audio patch is 0.16 s; the library regenerates a take whose patch
    # count reaches six per text token. The server surfaces that instead.
    PATCH_SECONDS = 0.16
    BAD_TAKE_RATIO = 6.0

    def __init__(self, model: str, device: str):
        import torch
        from voxcpm import VoxCPM

        if torch.version.hip is not None:
            # MIOpen's default find mode costs 10-11 s per NEW output length.
            # Disable cuDNN/MIOpen on HIP only, as ReverbEngine does.
            torch.backends.cudnn.enabled = False
        self._torch = torch
        t0 = time.monotonic()
        self._model = VoxCPM.from_pretrained(
            model,
            load_denoiser=False,
            optimize=False,
            device=None if device == "auto" else device,
        )
        self.load_seconds = time.monotonic() - t0
        self.sample_rate = self._model.tts_model.sample_rate
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "voxcpm2"

    def synthesize(
        self,
        text: str,
        *,
        reference: str | None = None,
        prompt_text: str | None = None,
        cfg: float = 2.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """float32 samples at `sample_rate`, and the seconds it took.

        Raises `BadTake` when the take carries six audio patches or more per
        text token.
        """
        call = self._take_call(
            text, reference=reference, prompt_text=prompt_text, cfg=cfg
        )

        with self._lock:
            if seed is not None:
                self._torch.manual_seed(seed)
            t0 = time.monotonic()
            audio = self._model.generate(**call)
            elapsed = time.monotonic() - t0
            tokens = len(self._model.tts_model.text_tokenizer(_flatten_text(text)))

        audio = np.asarray(audio, dtype=np.float32)
        self._reject_bad_take(len(audio), tokens)
        return audio, elapsed

    def generate_streaming(
        self,
        text: str,
        *,
        reference: str | None = None,
        prompt_text: str | None = None,
        cfg: float = 2.0,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """float32 chunks of the take, yielded as the library produces them.

        Takes what `synthesize` takes and holds the same lock across the whole
        generation, so chunk N reaches the caller as soon as the library has
        made it instead of after the finished take. The bad-take bound is
        applied once the take is complete, because it needs the finished length.
        """
        call = self._take_call(
            text, reference=reference, prompt_text=prompt_text, cfg=cfg
        )
        with self._lock:
            if seed is not None:
                self._torch.manual_seed(seed)
            samples = 0
            for chunk in self._model.generate_streaming(**call):
                chunk = np.asarray(chunk, dtype=np.float32)
                samples += len(chunk)
                yield chunk
            tokens = len(self._model.tts_model.text_tokenizer(_flatten_text(text)))
        self._reject_bad_take(samples, tokens)

    def synthesize_streaming(
        self,
        text: str,
        *,
        reference: str | None = None,
        prompt_text: str | None = None,
        cfg: float = 2.0,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """The streaming take under the name every engine answers to.

        VoxCPM2 yields the library's chunks; an engine whose library cannot
        stream yields the finished take as a single chunk. Callers that do not
        care which engine they hold call this.
        """
        yield from self.generate_streaming(
            text, reference=reference, prompt_text=prompt_text, cfg=cfg, seed=seed
        )

    def _take_call(
        self,
        text: str,
        *,
        reference: str | None,
        prompt_text: str | None,
        cfg: float,
    ) -> dict:
        """The library arguments one take is generated with: both paths use these.

        The reference goes in as both reference audio (`reference_wav_path`) and
        continuation audio (`prompt_wav_path` with its `prompt_text`) when its
        transcript is known, so the reference is isolated behind its own tokens
        while the prompt audio continues in context. Without a transcript it goes
        in as voice cloning (`reference_wav_path`) alone.
        """
        call = {
            "text": text,
            "cfg_value": cfg,
            "inference_timesteps": 10,
            "min_len": 2,
            "max_len": 4096,
            "normalize": False,
            "denoise": False,
            "retry_badcase": False,
        }
        if reference is not None:
            call["reference_wav_path"] = reference
            if prompt_text is not None:
                call["prompt_wav_path"] = reference
                call["prompt_text"] = prompt_text
        return call

    def _reject_bad_take(self, samples: int, tokens: int) -> None:
        """A take with six audio patches or more per text token is refused."""
        patches = math.floor(samples / self.sample_rate / self.PATCH_SECONDS)
        if patches >= self.BAD_TAKE_RATIO * tokens:
            raise BadTake(f"take is {patches} audio patches for {tokens} text tokens")

    def unload(self) -> None:
        """Drop the model and release its memory; the caller holds the TTS lock."""
        self._model = None
        self._torch.cuda.empty_cache()
        gc.collect()


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
