#!/usr/bin/env python3
"""Spike 005, addendum 2: paired timing of the two clone modes the server will use.

    docker exec -e HF_HUB_OFFLINE=1 spike5 /app/.venv/bin/python /work/modes_pair.py

The first pass (modes.py) ran while two build workers shared the host, and its medium takes of the
combined mode were 1.540 and 1.108: too far apart to quote. This one alternates the two modes on the
same text with the same seed, three pairs per length, so slow drift in host load lands on both modes
alike and the paired ratio is the measurement. Continuation-only is left out: the server does not
use it (a transcript-assisted clone is the combined mode).
"""

import json
import time

import numpy as np
import torch

torch.backends.cudnn.enabled = False
from voxcpm import VoxCPM  # noqa: E402

OUT = "/work/ab-modes"
with open("/work/out/texts.json") as f:
    TEXTS = json.load(f)
REF = "/work/out/reference.wav"
REF_TEXT = TEXTS["reference"][0]
SR = 48000

t0 = time.perf_counter()
model = VoxCPM.from_pretrained(load_denoiser=False, optimize=False)
print(f"load {time.perf_counter() - t0:.1f}s", flush=True)


def gen(text, seed, **kw):
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    t = time.perf_counter()
    wav = np.asarray(
        model.generate(
            text=text, cfg_value=2.0, inference_timesteps=10, retry_badcase=False, **kw
        ),
        dtype=np.float32,
    )
    torch.cuda.synchronize()
    el = time.perf_counter() - t
    return el, len(wav) / SR


# warm every code path once so no first-take cost lands in a pair
gen(TEXTS["short"][0], 7, reference_wav_path=REF)
gen(
    TEXTS["short"][0],
    7,
    prompt_wav_path=REF,
    prompt_text=REF_TEXT,
    reference_wav_path=REF,
)

res = {}
for length in ("short", "medium", "long"):
    text = TEXTS[length][0]
    pairs = []
    for i in (1, 2, 3):
        seed = 300 + i
        a = gen(text, seed, reference_wav_path=REF)
        b = gen(
            text,
            seed,
            prompt_wav_path=REF,
            prompt_text=REF_TEXT,
            reference_wav_path=REF,
        )
        pairs.append(
            {
                "ref": {
                    "elapsed_s": round(a[0], 3),
                    "audio_s": round(a[1], 2),
                    "rtf": round(a[0] / a[1], 4),
                },
                "comb": {
                    "elapsed_s": round(b[0], 3),
                    "audio_s": round(b[1], 2),
                    "rtf": round(b[0] / b[1], 4),
                },
            }
        )
        print(
            f"{length:6s} pair {i}: ref RTF {pairs[-1]['ref']['rtf']:.3f}  comb RTF {pairs[-1]['comb']['rtf']:.3f}",
            flush=True,
        )
    ref = [p["ref"]["rtf"] for p in pairs]
    comb = [p["comb"]["rtf"] for p in pairs]
    res[length] = {
        "pairs": pairs,
        "ref_rtf": round(float(np.mean(ref)), 4),
        "comb_rtf": round(float(np.mean(comb)), 4),
        "comb_over_ref": round(float(np.mean(comb) / np.mean(ref)), 4),
        "ref_min_max": [min(ref), max(ref)],
        "comb_min_max": [min(comb), max(comb)],
    }
    with open(f"{OUT}/modes_pair.json", "w") as f:
        json.dump(res, f, indent=1)
print(
    "PAIR_RC=0 "
    + json.dumps(
        {
            k: {"ref": v["ref_rtf"], "comb": v["comb_rtf"], "ratio": v["comb_over_ref"]}
            for k, v in res.items()
        }
    )
)
