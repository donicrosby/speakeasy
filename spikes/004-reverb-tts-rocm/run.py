"""Speakeasy spike 004: reverb-tts on gfx1151 (ROCm 10.1 pip stack).

Question: does reverb-tts 1.0.1 run unmodified on the HIP device, produce sane
audio, and how fast is it at batch 1?

Writes wavs + result.json into OUT (argv[1], default ./out).
"""

import importlib.metadata
import json
import math
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from reverb_tts import TTS
from torch.nn.attention import SDPBackend, sdpa_kernel

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
os.makedirs(OUT, exist_ok=True)

REF_TEXT = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
CASES = {
    "short": "Hello from an AMD graphics processor. This voice was cloned on ROCm.",
    "medium": (
        "Speakeasy transcribes voice notes with Whisper and Moonshine. Tonight it learned to talk back. "
        "The model you are hearing is Reverb, a three hundred million parameter voice cloning system, "
        "running on the integrated graphics processor of a Strix Halo laptop."
    ),
    "long": (
        "Masked diffusion text to speech works nothing like the older autoregressive models. "
        "Instead of emitting one audio token at a time, the model lays out a grid of codec tokens for the "
        "whole utterance and fills it in over a handful of passes, deciding the coarse codebooks first and "
        "polishing the fine ones at the end. That is what makes it fast: every pass runs in parallel over the "
        "entire target, so the cost grows with the number of passes rather than the number of frames. "
        "Reverb was pruned down from a larger teacher with distillation, retrained on seven languages, and "
        "then tuned with preference optimization. The result fits in under a gigabyte, clones a voice from a "
        "few seconds of reference audio, and speaks faster than real time even on modest hardware."
    ),
}
PL_TEXT = "Dzień dobry, jak się masz? Miło mi cię poznać."


def sdpa_mask_probe() -> dict:
    """Which SDPA kernels accept the sampler's 4D bf16 additive mask, and do they agree?"""

    b, h, s, d = 1, 12, 640, 128
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(b, h, s, d, generator=g).to("cuda", torch.bfloat16)
    k, v = q.clone(), q.clone()
    valid = torch.zeros(s, dtype=torch.bool, device="cuda")
    valid[:150] = True
    valid[300:] = True
    mask = (
        torch.where(valid[None, None, None, :], 0.0, -math.inf)
        .expand(b, 1, s, s)
        .clone()
    )
    mask[..., 0] = 0.0  # every query keeps one key, as the sampler does
    mask[..., :150] += 1.0  # the polish-pass reference bias
    mask = mask.to(torch.bfloat16)  # as the sampler does: mask.to(network dtype)
    out = {}
    ref = None
    backends = []
    for attr in (
        "FLASH_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
        "CUDNN_ATTENTION",
        "OVERRIDEABLE",
    ):
        be = getattr(SDPBackend, attr, None)
        if be is not None:
            backends.append(be)
    for be in backends:
        try:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(o).all())
            if ref is None:
                ref = o.float()
                diff = 0.0
            else:
                diff = float((o.float() - ref).abs().max())
            out[be.name] = {"finite": finite, "max_abs_diff_vs_first": round(diff, 5)}
        except Exception as e:  # noqa: BLE001
            out[be.name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.cuda.synchronize()
    out["default_dispatch"] = {
        "finite": bool(torch.isfinite(o).all()),
        "max_abs_diff_vs_first": round(float((o.float() - ref).abs().max()), 5),
    }
    return out


def stats(audio: np.ndarray) -> dict:
    return {
        "seconds": round(len(audio) / 24000, 2),
        "finite": bool(np.isfinite(audio).all()),
        "peak": round(float(np.abs(audio).max()), 3),
        "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)), 1),
    }


def main() -> None:
    info = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": importlib.metadata.version("transformers"),
        "reverb_tts": importlib.metadata.version("reverb-tts"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "device_name": torch.cuda.get_device_name(0),
        "gcnArchName": torch.cuda.get_device_properties(0).gcnArchName,
    }
    info["sdpa_mask_probe"] = sdpa_mask_probe()

    t0 = time.monotonic()
    tts = TTS.from_pretrained()  # int8, device/dtype auto
    torch.cuda.synchronize()
    info["load_seconds"] = round(time.monotonic() - t0, 1)
    info["device"] = str(tts.device)
    info["network_dtype"] = str(tts.network.audio_heads.weight.dtype)
    info["codec_dtype"] = str(next(tts.codec.parameters()).dtype)
    info["sample_rate"] = tts.sample_rate
    info["gpu_mem_after_load_mb"] = round(torch.cuda.memory_allocated() / 2**20)
    info["presets"] = tts.preset_voices()

    # A reference recording with a known transcript: synthesized from a preset,
    # then used to exercise the encode path (load, resample, codec encode).
    t0 = time.monotonic()
    preset = tts.preset_voice("en_male_narrator_1")
    ref = tts.generate(REF_TEXT, preset, language="en", seed=1)
    info["preset_generate"] = {
        "elapsed_s": round(time.monotonic() - t0, 2),
        **stats(ref),
    }
    sf.write(f"{OUT}/reference.wav", ref, tts.sample_rate)

    t0 = time.monotonic()
    voice = tts.voice(f"{OUT}/reference.wav", REF_TEXT, language="en")
    info["voice_encode"] = {
        "elapsed_s": round(time.monotonic() - t0, 2),
        "reference_seconds": round(voice.duration, 2),
        "speech_frames": voice.speech_frames,
        "codes_shape": list(voice.codes.shape),
    }
    voice.save(f"{OUT}/voice.npz")
    info["voice_encode"]["npz_bytes"] = os.path.getsize(f"{OUT}/voice.npz")

    # Timed generation: one warm-up, three timed takes per case.
    info["runs"] = {}
    for name, text in CASES.items():
        tts.generate(text, voice, language="en", seed=100)  # warm-up
        takes = []
        for i in range(3):
            t0 = time.monotonic()
            audio = tts.generate(text, voice, language="en", seed=100 + i)
            elapsed = time.monotonic() - t0
            takes.append({"elapsed_s": round(elapsed, 3), **stats(audio)})
            if i == 0:
                sf.write(f"{OUT}/{name}.wav", audio, tts.sample_rate)
        best = min(takes, key=lambda t: t["elapsed_s"])
        info["runs"][name] = {
            "words": len(text.split()),
            "takes": takes,
            "x_real_time": round(best["seconds"] / best["elapsed_s"], 1),
            "rtf": round(best["elapsed_s"] / best["seconds"], 3),
        }
        print(
            f"{name:7s} {best['seconds']:6.2f}s audio in {best['elapsed_s']:6.3f}s  "
            f"-> {best['seconds'] / best['elapsed_s']:5.1f}x real time",
            flush=True,
        )

    # Batch: four medium texts in one call.
    tts.generate([CASES["medium"]] * 4, voice, language="en", seed=200)
    t0 = time.monotonic()
    audios = tts.generate([CASES["medium"]] * 4, voice, language="en", seed=201)
    elapsed = time.monotonic() - t0
    total = sum(len(a) / tts.sample_rate for a in audios)
    info["batch4"] = {
        "elapsed_s": round(elapsed, 3),
        "audio_seconds": round(total, 2),
        "x_real_time": round(total / elapsed, 1),
    }

    # Cross-language with the same voice, and seed reproducibility on this device.
    pl = tts.generate(PL_TEXT, voice, language="pl", seed=7)
    sf.write(f"{OUT}/polish.wav", pl, tts.sample_rate)
    info["polish"] = stats(pl)
    a = tts.generate(CASES["short"], voice, language="en", seed=42)
    b = tts.generate(CASES["short"], voice, language="en", seed=42)
    info["seed_reproducible"] = bool(np.array_equal(a, b))

    info["gpu_mem_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    with open(f"{OUT}/result.json", "w") as f:
        json.dump(info, f, indent=1)
    print(
        json.dumps(
            {k: v for k, v in info.items() if k not in ("presets", "sdpa_mask_probe")},
            indent=1,
        )
    )
    print("sdpa probe:", json.dumps(info["sdpa_mask_probe"]))


if __name__ == "__main__":
    main()
