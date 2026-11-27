#!/usr/bin/env python3
"""Spike 005, addendum: the three clone modes side by side on the gfx1151 HIP device.

    docker exec -e HF_HUB_OFFLINE=1 spike5 /app/.venv/bin/python /work/modes.py

The harness (`run.py`) timed `reference_wav_path` alone ("clone") and the prompt
pair alone, and labelled the latter "ultimate". The vendor's card (line 137)
defines Ultimate Cloning as all three arguments: the same clip passed to both
`reference_wav_path` and `prompt_wav_path`, plus `prompt_text`. That combined
mode is `model/voxcpm2.py:484-515` (mode "ref_continuation") and had never been
run here. This runs all three modes over the same texts with the same seeds,
first take (cold for that mode and length) plus two warm takes, MIOpen find
mode off as in the main run, and retry_badcase off so a silent retry cannot
inflate a timing.
"""

import json
import os
import time

import numpy as np
import soundfile as sf
import torch

torch.backends.cudnn.enabled = False
from voxcpm import VoxCPM  # noqa: E402

OUT = "/work/ab-modes"
os.makedirs(OUT, exist_ok=True)
with open("/work/out/texts.json") as f:
    TEXTS = json.load(f)
REF = "/work/out/reference.wav"
REF_TEXT = TEXTS["reference"][0]
SR = 48000


def log(msg):
    print(msg, flush=True)


def peak_gb():
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


t0 = time.perf_counter()
model = VoxCPM.from_pretrained(load_denoiser=False, optimize=False)
log(f"load {time.perf_counter() - t0:.1f}s; sample_rate {model.tts_model.sample_rate}")
assert model.tts_model.sample_rate == SR, model.tts_model.sample_rate

MODES = (
    ("ref", {"reference_wav_path": REF}),
    ("cont", {"prompt_wav_path": REF, "prompt_text": REF_TEXT}),
    (
        "comb",
        {"prompt_wav_path": REF, "prompt_text": REF_TEXT, "reference_wav_path": REF},
    ),
)


def gen(name, text, seed, **kw):
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    wav = model.generate(
        text=text, cfg_value=2.0, inference_timesteps=10, retry_badcase=False, **kw
    )
    el = time.perf_counter() - t
    wav = np.asarray(wav, dtype=np.float32)
    secs = len(wav) / SR
    sf.write(f"{OUT}/{name}.wav", wav, SR, subtype="PCM_16")
    rec = {
        "elapsed_s": round(el, 3),
        "audio_s": round(secs, 2),
        "rtf": round(el / secs, 4),
        "peak_gb": peak_gb(),
        "peak_amplitude": round(float(np.max(np.abs(wav))), 4),
    }
    log(
        f"{name:24} {el:7.2f}s  audio {secs:6.2f}s  RTF {el / secs:6.3f}  peak {rec['peak_gb']} GB  amp {rec['peak_amplitude']}"
    )
    return rec


res = {"reference_seconds": 5.63, "modes": {}}
for length in ("short", "medium", "long"):
    text = TEXTS[length][0]
    for mode, kw in MODES:
        recs = {"first": gen(f"{length}_{mode}_first", text, seed=7, **kw)}
        for i in (1, 2):
            recs[f"warm{i}"] = gen(f"{length}_{mode}_warm{i}", text, seed=100 + i, **kw)
        warm = [recs["warm1"]["rtf"], recs["warm2"]["rtf"]]
        recs["mean_warm_rtf"] = round(float(np.mean(warm)), 4)
        res["modes"][f"{length}_{mode}"] = recs
        with open(f"{OUT}/modes.json", "w") as f:
            json.dump(res, f, indent=1)
log(
    f"MODES_RC=0 {json.dumps({k: v['mean_warm_rtf'] for k, v in res['modes'].items()})}"
)
