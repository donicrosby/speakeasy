#!/usr/bin/env python3
"""Spike 005: what does cfg_value do on this device? (the owner asked for cfg as an API parameter)

    /app/.venv/bin/python /work/cfg_sweep.py

Clone mode, the short sentence, ONE seed for every take, so guidance strength is
the only thing that differs between clips. Records time, loudness and peak per
take into /work/out/cfg_sweep.json and leaves the wavs in /work/out/cfg/ for the
oracle and the listening page.
"""

import json
import os
import time

import numpy as np
import soundfile as sf
import torch

torch.backends.cudnn.enabled = False  # the spike 004 fix: no MIOpen find-mode stalls
from voxcpm import VoxCPM  # noqa: E402

OUT = "/work/out/cfg"
os.makedirs(OUT, exist_ok=True)
TEXTS = json.load(open("/work/out/texts.json"))
RES = {}
t0 = time.perf_counter()
m = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2", load_denoiser=False, optimize=False, device="cuda"
)
RES["load_s"] = round(time.perf_counter() - t0, 2)
sr = m.tts_model.sample_rate


def one(name, cfg, seed=7, **kw):
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    wav = np.asarray(
        m.generate(
            text=TEXTS["short"][0],
            cfg_value=cfg,
            inference_timesteps=10,
            retry_badcase=False,
            **kw,
        ),
        dtype=np.float32,
    )
    el = time.perf_counter() - t
    sf.write(f"{OUT}/{name}.wav", wav, sr, subtype="PCM_16")
    rms_db = 20 * np.log10(np.sqrt(np.mean(wav**2)) + 1e-9)
    rec = {
        "cfg": cfg,
        "elapsed_s": round(el, 2),
        "audio_s": round(len(wav) / sr, 2),
        "rtf": round(el / (len(wav) / sr), 3),
        "peak_float": round(float(np.abs(wav).max()), 3),
        "rms_dbfs": round(float(rms_db), 1),
        "clipped_samples": int(np.sum(np.abs(wav) >= 0.999)),
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(
        f"{name:14} cfg {cfg:3.1f}  {el:6.2f}s  audio {rec['audio_s']:5.2f}s  RTF {rec['rtf']:5.3f}  "
        f"peak {rec['peak_float']:5.3f}  rms {rec['rms_dbfs']:6.1f} dBFS  clip {rec['clipped_samples']}",
        flush=True,
    )
    return rec


# warm-up (first take at a new length pays the one-time HIP/JIT warm-up)
one("warmup", 2.0)
for cfg in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
    RES[f"cfg_{cfg}"] = one(
        f"clone_cfg{cfg:g}", cfg, reference_wav_path="/work/out/reference.wav"
    )
# one repeat at the default with other seeds: how much of what we hear is seed noise
for seed in (8, 9):
    RES[f"cfg2_seed{seed}"] = one(
        f"clone_cfg2_seed{seed}",
        2.0,
        seed=seed,
        reference_wav_path="/work/out/reference.wav",
    )
json.dump(RES, open("/work/out/cfg_sweep.json", "w"), indent=1)
print("CFG_RC=0")
