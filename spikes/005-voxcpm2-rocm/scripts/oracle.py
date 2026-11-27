#!/usr/bin/env python3
"""Spike 005 oracle: transcribe VoxCPM2's output with faster-whisper on CPU.

Same judge and scoring as spike 004: large-v3-turbo, CPU int8, word error rate
after lowercasing and splitting on non-word characters. Runs in its own
process (the spike 004 venv) so it never imports the ROCm torch stack.

    /workspace/spike-004-scratch/.venv/bin/python oracle.py <out-dir>

Expects <out-dir>/oracle_texts.json: {name: [expected_text, language]}.
Writes <out-dir>/oracle.json.
"""

import json
import re
import sys
import time

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
with open(f"{OUT}/oracle_texts.json") as f:
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
    from faster_whisper import WhisperModel

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
    results["aggregate"] = f"{total_edits}/{total_words}"
    print(f"\naggregate WER {results['aggregate_wer']} ({total_edits}/{total_words})")
    with open(f"{OUT}/oracle.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
