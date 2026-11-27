#!/usr/bin/env python3
"""Spike 005 harness: VoxCPM2 on the gfx1151 HIP device, inside the speakeasy rocm image.

    /app/.venv/bin/python /work/run.py --out /work/out/main --cudnn off --retry 0 --phase load,smoke,timing,modes

Records every measurement into <out>/result.json as it goes, so a crash keeps
what came before it. `--optimize 1` is the torch.compile control (their default;
their docs say to turn it off on ROCm). `--cudnn off` disables MIOpen before
anything loads (the spike 004 fix). `--retry 1` leaves their bad-case retry on,
which can silently repeat a whole generation; timing runs use 0.
"""

import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--phase", default="load,smoke,timing,modes")
p.add_argument("--optimize", type=int, default=0)
p.add_argument("--cudnn", choices=["default", "off"], default="off")
p.add_argument("--retry", type=int, default=0)
p.add_argument("--steps", type=int, default=10)
p.add_argument("--cfg", type=float, default=2.0)
a = p.parse_args()

os.makedirs(a.out, exist_ok=True)
if a.cudnn == "off":
    torch.backends.cudnn.enabled = False

RES = {
    "args": vars(a),
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "cudnn_enabled": torch.backends.cudnn.enabled,
}
with open("/work/out/texts.json") as f:
    TEXTS = json.load(f)
REF = "/work/out/reference.wav"
REF_TEXT = TEXTS["reference"][0]


def save():
    with open(f"{a.out}/result.json", "w") as f:
        json.dump(RES, f, indent=1)


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return round(int(line.split()[1]) / 2**20, 2)
    return None


def mem():
    torch.cuda.synchronize()
    return {
        "alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "rss_gb": rss_gb(),
    }


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


phases = a.phase.split(",")
model = None

if "load" in phases:
    from voxcpm import VoxCPM

    t0 = time.perf_counter()
    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2", load_denoiser=False, optimize=bool(a.optimize), device="cuda"
    )
    RES["load_s"] = round(time.perf_counter() - t0, 2)
    m = model.tts_model
    RES["runtime"] = {
        "device": str(m.device),
        "dtype": str(m.config.dtype),
        "sample_rate": m.sample_rate,
        "encode_sample_rate": m._encode_sample_rate,
        "patch_size": m.patch_size,
        "chunk_size": m.chunk_size,
    }
    RES["mem_after_load"] = mem()
    log(f"loaded in {RES['load_s']}s  {RES['runtime']}  {RES['mem_after_load']}")
    save()


def gen(name, text, retries=None, seed=None, **kw):
    """One timed generate; writes <out>/<name>.wav; returns the record."""
    if seed is not None:
        torch.manual_seed(seed)
    retry = a.retry if retries is None else retries
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    wav = model.generate(
        text=text,
        cfg_value=a.cfg,
        inference_timesteps=a.steps,
        retry_badcase=bool(retry),
        **kw,
    )
    el = time.perf_counter() - t0
    sr = model.tts_model.sample_rate
    sf.write(
        f"{a.out}/{name}.wav", np.asarray(wav, dtype=np.float32), sr, subtype="PCM_16"
    )
    secs = len(wav) / sr
    rec = {
        "elapsed_s": round(el, 3),
        "audio_s": round(secs, 2),
        "rtf": round(el / secs, 4),
        "chars": len(text),
        "peak": mem(),
    }
    log(
        f"{name:28} {el:7.2f}s  audio {secs:6.2f}s  RTF {el / secs:6.3f}  peak {rec['peak']['alloc_gb']} GB"
    )
    return wav, rec


if "smoke" in phases:
    short = TEXTS["short"][0]
    RES["first_request"] = gen("first_plain", short)[1]
    RES["determinism"] = {}
    w1, _ = gen("det_seed7_a", short, seed=7)
    w2, _ = gen("det_seed7_b", short, seed=7)
    w3, _ = gen("det_seed8", short, seed=8)
    RES["determinism"] = {
        "same_seed_identical": bool(np.array_equal(w1, w2)),
        "other_seed_differs": bool(not np.array_equal(w1, w3)),
    }
    log(f"determinism: {RES['determinism']}")
    save()

if "timing" in phases:
    RES["timing"] = {}
    for name in ("short", "medium", "long"):
        text = TEXTS[name][0]
        for mode, kw in (("clone", {"reference_wav_path": REF}), ("plain", {})):
            first = gen(f"{name}_{mode}_first", text, **kw)[1]
            warm = [
                gen(f"{name}_{mode}_warm{i}", text, seed=100 + i, **kw)[1]
                for i in (1, 2)
            ]
            RES["timing"][f"{name}_{mode}"] = {
                "first": first,
                "warm": warm,
                "mean_warm_rtf": round(float(np.mean([w["rtf"] for w in warm])), 4),
            }
        save()
    # steps sweep on the short clone: the quality/speed dial
    RES["steps_sweep"] = {}
    for steps in (5, 10, 16):
        a.steps = steps
        warm = [
            gen(
                f"sweep_short_clone_s{steps}_warm{i}",
                TEXTS["short"][0],
                seed=200 + i,
                reference_wav_path=REF,
            )[1]
            for i in (1, 2)
        ]
        RES["steps_sweep"][steps] = {
            "warm": warm,
            "mean_warm_rtf": round(float(np.mean([w["rtf"] for w in warm])), 4),
        }
    a.steps = 10
    save()

if "modes" in phases:
    RES["modes"] = {}
    RES["modes"]["ultimate_short"] = gen(
        "ultimate_short",
        TEXTS["short"][0],
        seed=7,
        prompt_wav_path=REF,
        prompt_text=REF_TEXT,
    )[1]
    RES["modes"]["design"] = gen(
        "design",
        "(A warm, middle-aged man with a calm, low voice)" + TEXTS["short"][0],
        seed=7,
    )[1]
    RES["modes"]["clone_polish"] = gen(
        "polish_clone", TEXTS["polish"][0], seed=7, reference_wav_path=REF
    )[1]
    RES["modes"]["styled_clone"] = gen(
        "styled_clone",
        "(slightly faster, cheerful tone)" + TEXTS["short"][0],
        seed=7,
        reference_wav_path=REF,
    )[1]
    save()

RES["mem_final"] = mem()
save()
log(f"DONE  peak {RES['mem_final']}")
