"""Spike 004 oracle: transcribe Reverb's output with faster-whisper on CPU.

The STT engine speakeasy already ships is the judge of the TTS output: if the
cloned voice said the text, large-v3-turbo should get the words back.
Writes out/oracle.json.
"""

import json
import re
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"

# Expected texts come from a file so this process never imports torch/reverb_tts:
# the ROCm torch libs and the PyPI ctranslate2 wheel in one process crash on free().
with open(f"{OUT}/texts.json") as f:
    FILES = json.load(f)


def norm(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main() -> None:
    import ctranslate2
    from faster_whisper import WhisperModel

    print("ctranslate2 devices: cuda=", ctranslate2.get_cuda_device_count())
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    results = {}
    total_edits = total_words = 0
    for name, (expected, lang) in FILES.items():
        t0 = time.monotonic()
        segments, info = model.transcribe(
            f"{OUT}/{name}.wav", language=lang, beam_size=5
        )
        text = "".join(s.text for s in segments).strip()
        elapsed = time.monotonic() - t0
        hyp, ref = norm(text), norm(expected)
        edits = edit_distance(ref, hyp)
        wer = edits / max(len(ref), 1)
        total_edits += edits
        total_words += len(ref)
        results[name] = {
            "expected": expected,
            "transcribed": text,
            "ref_words": len(ref),
            "edits": edits,
            "wer": round(wer, 4),
            "whisper_elapsed_s": round(elapsed, 2),
            "audio_seconds": round(info.duration, 2),
        }
        print(
            f"\n[{name}] wer={wer:.3f} ({edits}/{len(ref)}) audio={info.duration:.1f}s\n"
            f"  hyp: {text}\n  ref: {expected}"
        )
    results["aggregate_wer"] = round(total_edits / max(total_words, 1), 4)
    print(f"\naggregate WER {results['aggregate_wer']} ({total_edits}/{total_words})")
    with open(f"{OUT}/oracle.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
