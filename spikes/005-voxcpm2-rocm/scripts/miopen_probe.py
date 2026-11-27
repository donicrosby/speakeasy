#!/usr/bin/env python3
"""Spike 005: MIOpen on vs off, first time at a new output length vs warm.

    /app/.venv/bin/python /work/miopen_probe.py default|off

The audio VAE decodes the whole utterance at the end in fp32 convs, so a never
seen output length can trigger MIOpen's kernel search (the spike 004 trap).
Same seed per run, so "short" comes back at the same length the second time
and that decode shape is warm.
"""

import json
import sys
import time

import torch

mode = sys.argv[1] if len(sys.argv) > 1 else "default"
if mode == "off":
    torch.backends.cudnn.enabled = False

from voxcpm import VoxCPM  # noqa: E402

free, total = torch.cuda.mem_get_info()
print(
    f"cudnn(miopen) enabled: {torch.backends.cudnn.enabled}  mem free {free / 2**30:.1f} / {total / 2**30:.1f} GiB",
    flush=True,
)
t0 = time.perf_counter()
m = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda"
)
texts = json.load(open("/work/out/texts.json"))
out = {
    "mode": mode,
    "cudnn_enabled": torch.backends.cudnn.enabled,
    "load_s": round(time.perf_counter() - t0, 2),
    "runs": {},
}
for name in ("short", "medium", "short"):
    for i in (1, 2):
        torch.manual_seed(7)
        t = time.perf_counter()
        w = m.generate(text=texts[name][0], retry_badcase=False)
        el = time.perf_counter() - t
        secs = len(w) / 48000
        out["runs"][f"{name}_{i}"] = {
            "elapsed_s": round(el, 2),
            "audio_s": round(secs, 2),
            "rtf": round(el / secs, 3),
        }
        print(
            f"{mode:7} {name:6} run{i}  {el:6.2f}s  audio {secs:5.2f}s  RTF {el / secs:5.3f}",
            flush=True,
        )
json.dump(out, open(f"/work/out/miopen_{mode}.json", "w"), indent=1)
