"""Spike 004: steady-state latency across DIFFERENT output lengths.

Every new target length means new convolution shapes in the codec decoder, and
MIOpen searches for kernels per shape. Each text is generated three times: the
first take pays any per-shape cost, the next two show the warm cost.

MODE=nocudnn disables MIOpen (torch.backends.cudnn) so convolutions take the
native HIP path. MIOPEN_FIND_MODE is an environment knob, set it outside.
"""

import json
import os
import sys
import time

import torch
from reverb_tts import TTS, Voice

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
MODE = os.environ.get("MODE", "default")
if MODE == "nocudnn":
    torch.backends.cudnn.enabled = False  # on ROCm "cudnn" is MIOpen

TEXTS = {
    "w5": "Five words go here now.",
    "w12a": "Hello from an AMD graphics processor. This voice was cloned on ROCm.",
    "w25": "The server keeps one process per model and routes each request to the "
    "engine that fits it best, so short notes never wait behind long ones.",
    "w40": "Speakeasy transcribes voice notes with Whisper and Moonshine. Tonight it learned to talk back. "
    "The model you are hearing is Reverb, a three hundred million parameter voice cloning system, "
    "running on the integrated graphics processor of a Strix Halo laptop.",
    "w60": "Voice notes arrive as compressed audio, get decoded once, and are then handed to whichever "
    "engine the router picks. A short English note goes to the small fast model. Anything longer, "
    "or anything in another language, goes to the large one. The transcript comes back in the "
    "shape every client already knows, with timestamps and confidence values intact.",
    "w12b": "Two takes of the same length but different words and letters entirely.",
}

voice = Voice.load(f"{OUT}/voice.npz")
t0 = time.monotonic()
tts = TTS.from_pretrained()
print(f"{MODE:9s} load from cache: {time.monotonic() - t0:.1f}s", flush=True)
t0 = time.monotonic()
tts.generate(TEXTS["w12a"], voice, language="en", seed=5)  # kernel warm-up
print(f"{MODE:9s} warm-up call: {time.monotonic() - t0:.1f}s", flush=True)
results = {}
for name, text in TEXTS.items():
    elapsed = []
    for i in range(3):
        t0 = time.monotonic()
        audio = tts.generate(text, voice, language="en", seed=5 + i)
        elapsed.append(round(time.monotonic() - t0, 3))
    seconds = round(len(audio) / 24000, 2)
    results[name] = {
        "words": len(text.split()),
        "seconds": seconds,
        "elapsed_s": elapsed,
    }
    print(f"{MODE:9s} {name:5s} {seconds:6.2f}s audio  elapsed {elapsed}", flush=True)

# Codec decode alone: a fresh length vs a repeated one.
codes = torch.randint(0, 1024, (1, 8, 373))  # ~15 s, an unusual frame count
dec = {}
for label in ("decode_new_shape", "decode_same_shape"):
    t0 = time.monotonic()
    tts.codec.decode(codes.to(tts.device))
    torch.cuda.synchronize()
    dec[label] = round(time.monotonic() - t0, 3)
results["codec_decode_s"] = dec
print(MODE, "codec decode:", dec)
with open(f"{OUT}/lengths_{MODE}.json", "w") as f:
    json.dump(results, f, indent=1)
