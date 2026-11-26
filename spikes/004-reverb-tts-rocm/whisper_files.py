"""Whisper (CPU, large-v3-turbo) over a {name: [expected, lang]} json of wavs in OUT."""

import json
import sys

OUT = sys.argv[1]
TEXTS = sys.argv[2]

from faster_whisper import WhisperModel  # noqa: E402
from oracle import edit_distance, norm  # noqa: E402

files = json.load(open(f"{OUT}/{TEXTS}"))
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
results = {}
for name, (expected, lang) in files.items():
    segments, info = model.transcribe(f"{OUT}/{name}.wav", language=lang, beam_size=5)
    text = "".join(s.text for s in segments).strip()
    hyp, ref = norm(text), norm(expected)
    edits = edit_distance(ref, hyp)
    results[name] = {
        "wer": round(edits / max(len(ref), 1), 3),
        "edits": edits,
        "words": len(ref),
        "transcribed": text,
        "audio_seconds": round(info.duration, 2),
    }
    print(
        f"[{name}] wer={results[name]['wer']} ({edits}/{len(ref)}) {info.duration:.1f}s\n  hyp: {text}"
    )
json.dump(
    results, open(f"{OUT}/variants_oracle.json", "w"), indent=1, ensure_ascii=False
)
