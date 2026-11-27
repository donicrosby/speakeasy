#!/usr/bin/env python3
"""Spike 005: does voxcpm's silent bad-case retry ever fire on this device?

    /app/.venv/bin/python /work/retry_probe.py

With retry_badcase=True (the library default) a take whose audio-to-text ratio
exceeds 6.0 is thrown away and regenerated, up to 3 times. Each run prints
elapsed, audio seconds and the ratio the library would compare.
"""

import json
import time

import torch

torch.backends.cudnn.enabled = False
from voxcpm import VoxCPM  # noqa: E402

m = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda"
)
texts = json.load(open("/work/out/texts.json"))
m.tts_model.text_tokenizer  # noqa: B018
out = {}
for name in ("short", "medium", "short"):
    text = texts[name][0]
    n_tok = len(m.tts_model.text_tokenizer(text))
    torch.manual_seed(11)
    t = time.perf_counter()
    w = m.generate(text=text, retry_badcase=True)
    el = time.perf_counter() - t
    secs = len(w) / 48000
    patches = round(secs * 6.25)
    out[name] = {
        "elapsed_s": round(el, 2),
        "audio_s": round(secs, 2),
        "text_tokens": n_tok,
        "audio_patches_per_text_token": round(patches / n_tok, 2),
        "threshold": 6.0,
    }
    print(
        f"{name:6} {el:6.2f}s audio {secs:5.2f}s  text tokens {n_tok:3d}  patches/token {patches / n_tok:5.2f} (retry above 6.0)",
        flush=True,
    )
json.dump(out, open("/work/out/retry_probe.json", "w"), indent=1)
