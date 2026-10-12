"""OpenAI-compatible response formatters.

Reference: https://platform.openai.com/docs/api-reference/audio/createTranscription
Response formats: json, text, srt, vtt, verbose_json.
"""

from .engines import TranscriptionResult


def _ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


def to_json(result: TranscriptionResult) -> dict:
    return {"text": result.text}


def to_verbose_json(result: TranscriptionResult) -> dict:
    return {
        "task": "transcribe",
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "segments": [
            {
                "id": i,
                "seek": 0,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "tokens": [],
                "temperature": s.temperature,
                "avg_logprob": s.avg_logprob,
                "compression_ratio": s.compression_ratio,
                "no_speech_prob": s.no_speech_prob,
            }
            for i, s in enumerate(result.segments)
        ],
    }


def to_text(result: TranscriptionResult) -> str:
    return result.text


def to_srt(result: TranscriptionResult) -> str:
    lines = []
    for i, s in enumerate(result.segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(s.start)} --> {_ts_srt(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines)


def to_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for s in result.segments:
        lines.append(f"{_ts_vtt(s.start)} --> {_ts_vtt(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines)


FORMATS = {
    "json": (to_json, "application/json"),
    "verbose_json": (to_verbose_json, "application/json"),
    "text": (to_text, "text/plain; charset=utf-8"),
    "srt": (to_srt, "text/plain; charset=utf-8"),
    "vtt": (to_vtt, "text/vtt; charset=utf-8"),
}
