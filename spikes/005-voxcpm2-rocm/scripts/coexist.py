#!/usr/bin/env python3
"""Spike 005 coexistence: VoxCPM2, Reverb and Whisper (HIP CTranslate2) in ONE process.

    /app/.venv/bin/python /work/coexist.py

The shipped server already runs Whisper next to Reverb. This adds VoxCPM2 and
interleaves calls in both orders. "Alive after every step" is the result; the
step log lands in /work/out/coexist.json as it goes.
"""

import json
import time

import numpy as np
import soundfile as sf
import torch

torch.backends.cudnn.enabled = False  # spike 004, for Reverb's codec

OUT = "/work/out/coexist.json"
LOG = {"steps": []}


def step(name, **kw):
    LOG["steps"].append({"name": name, **kw})
    with open(OUT, "w") as f:
        json.dump(LOG, f, indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] {name} {kw}", flush=True)


def mem():
    torch.cuda.synchronize()
    return {
        "alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


text = json.load(open("/work/out/texts.json"))["short"][0]
REF = "/work/out/reference.wav"

from faster_whisper import WhisperModel  # noqa: E402

t = time.perf_counter()
wm = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
step("whisper_load", s=round(time.perf_counter() - t, 1))

from reverb_tts import TTS  # noqa: E402

t = time.perf_counter()
rtts = TTS.from_pretrained("software-mansion/Reverb-TTS", variant="int8", device="cuda")
voice = rtts.preset_voice("en_male_narrator_1")
step("reverb_load", s=round(time.perf_counter() - t, 1), mem=mem())

from voxcpm import VoxCPM  # noqa: E402

t = time.perf_counter()
vox = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda"
)
step("voxcpm_load", s=round(time.perf_counter() - t, 1), mem=mem())

order = (
    ("reverb", "voxcpm")
    if len(__import__("sys").argv) < 2
    else tuple(__import__("sys").argv[1].split(","))
)
for r in (1, 2):
    for engine in order:
        t = time.perf_counter()
        if engine == "reverb":
            a = rtts.generate(text, voice, "en", seed=r, speed=1.0)
            path = f"/work/out/coexist_round{r}_reverb.wav"
            sf.write(path, a, rtts.sample_rate, subtype="PCM_16")
            secs = len(a) / rtts.sample_rate
        else:
            b = vox.generate(text=text, retry_badcase=False, reference_wav_path=REF)
            path = f"/work/out/coexist_round{r}_voxcpm.wav"
            sf.write(path, np.asarray(b, dtype=np.float32), 48000, subtype="PCM_16")
            secs = len(b) / 48000
        step(
            f"round{r}_{engine}",
            s=round(time.perf_counter() - t, 2),
            audio_s=round(secs, 2),
        )
        t = time.perf_counter()
        segs, _ = wm.transcribe(path, language="en", beam_size=5)
        heard = "".join(s.text for s in segs).strip()
        step(
            f"round{r}_whisper_after_{engine}",
            s=round(time.perf_counter() - t, 2),
            heard=heard[:140],
        )
        step(f"round{r}_after_{engine}_alive", mem=mem())

step("done_alive", mem=mem())
