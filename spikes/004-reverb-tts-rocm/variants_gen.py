"""Spike 004: is the one odd word ('ROCm' heard as 'Aero Team') the take or the text?

GPU: same short text at several seeds, plus one contextual variant.
Prints what Verbalis makes of each text. Run whisper_files.py on the results.
"""

import sys

import soundfile as sf
from reverb_tts import TTS, Voice
from reverb_tts.text import prepare_text
from run import CASES

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
voice = Voice.load(f"{OUT}/voice.npz")
short = CASES["short"]
contextual = "Hello from an AMD graphics processor. This voice was cloned on ROCm, AMD's ROCm compute stack."

print("prepared short     :", prepare_text(short, "en"))
print("prepared contextual:", prepare_text(contextual, "en"))

tts = TTS.from_pretrained()
files = {}
for seed in (100, 101, 102, 103):
    audio = tts.generate(short, voice, language="en", seed=seed)
    sf.write(f"{OUT}/short_seed{seed}.wav", audio, 24000)
    files[f"short_seed{seed}"] = [short, "en"]
for seed in (100, 101):
    audio = tts.generate(contextual, voice, language="en", seed=seed)
    sf.write(f"{OUT}/contextual_seed{seed}.wav", audio, 24000)
    files[f"contextual_seed{seed}"] = [contextual, "en"]

import json  # noqa: E402

# Control: same take on the GPU in fp32. Same words -> bf16 numerics are not
# what whisper is objecting to.
del tts
import gc  # noqa: E402

import torch  # noqa: E402

gc.collect()
torch.cuda.empty_cache()
fp32 = TTS.from_pretrained(dtype=torch.float32)
print("fp32 control network dtype:", fp32.network.audio_heads.weight.dtype)
audio = fp32.generate(short, voice, language="en", seed=100)
sf.write(f"{OUT}/short_fp32.wav", audio, 24000)
files["short_fp32"] = [short, "en"]
audio = fp32.generate(contextual, voice, language="en", seed=100)
sf.write(f"{OUT}/contextual_fp32.wav", audio, 24000)
files["contextual_fp32"] = [contextual, "en"]

json.dump(files, open(f"{OUT}/variants_texts.json", "w"), indent=1, ensure_ascii=False)
print("wrote", len(files), "variant wavs")
